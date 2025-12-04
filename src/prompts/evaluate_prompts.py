# ==============================================================================
# RAG Prompt Evaluation Pipeline
# ==============================================================================
#
# Automated evaluation of prompt versions using MLflow GenAI framework.
# Compares multiple prompt versions and promotes the best to @champion.
#
# This script:
#   1. Loads evaluation dataset from DVC (versioned questions + answers)
#   2. Runs each prompt version through the RAG pipeline
#   3. Scores responses with custom metrics:
#      - concept_coverage: Checks if key concepts appear in answer
#      - answer_quality_judge: LLM-as-judge quality assessment
#   4. Calculates composite score (60% concept + 40% judge)
#   5. Promotes best prompt to @champion alias in MLflow
#
# Usage:
#   python src/prompts/evaluate_promts.py \
#       --prompt-name readme-rag-prompt \
#       --prompt-versions 1 2 3 \
#       --dvc-data-version opencloudhub-readmes-rag-evaluation-v1.0.0 \
#       --auto-promote
#
# Prerequisites:
#   - Environment variables: source .env.minikube
#   - Port-forward PostgreSQL in local dev: kubectl port-forward -n storage svc/demo-app-db-cluster-rw 5432:5432
#   - DVC configured with access to data registry
#
# =============================================================================="""

# ============================================================
# TRACING SETUP - Must be FIRST, before any other imports
# ============================================================
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

trace.set_tracer_provider(TracerProvider())

# ============================================================
# Standard imports - now safe
# ============================================================
import argparse
import io
import json

import dvc.api
import httpx
import mlflow
import pandas as pd
import urllib3
from loguru import logger as loguru_logger
from mlflow.entities import Feedback
from mlflow.genai import evaluate
from mlflow.genai.datasets import create_dataset, search_datasets
from mlflow.genai.scorers import scorer

from src.core.config import CONFIG

logger = loguru_logger.bind(name="evaluate_prompts")
urllib3.disable_warnings()

# MLflow setup
mlflow.set_tracking_uri(CONFIG.mlflow_tracking_uri)
mlflow.set_experiment("RAG Prompt Evaluation")
mlflow.langchain.autolog()


def load_eval_dataset(dvc_data_version: str):
    """Load dataset from DVC and create/reuse MLflow dataset."""
    logger.info(f"Loading eval dataset (DVC version: {dvc_data_version})...")

    exp = mlflow.get_experiment_by_name("RAG Prompt Evaluation")
    exp_id = (
        exp.experiment_id if exp else mlflow.create_experiment("RAG Prompt Evaluation")
    )

    logger.info("  Checking for existing dataset...")
    existing_datasets = search_datasets(
        experiment_ids=[exp_id],
        filter_string=f"tags.dvc_data_version = '{dvc_data_version}'",
        max_results=1,
        order_by=["created_time DESC"],
    )

    if existing_datasets:
        dataset = existing_datasets[0]
        logger.info(f"  ✓ Reusing existing dataset: {dataset.name}")
        return dataset

    logger.info("  No existing dataset found, creating new one...")

    csv_content = dvc.api.read(
        path=CONFIG.eval_data_path,
        repo=CONFIG.dvc_repo,
        rev=dvc_data_version,
        mode="r",
    )
    df = pd.read_csv(io.StringIO(csv_content))
    logger.success(f"  ✓ Loaded {len(df)} questions from DVC")

    dataset = create_dataset(
        name=f"readme-rag-eval-{dvc_data_version}",
        experiment_id=exp_id,
        tags={
            "dvc_data_version": dvc_data_version,
            "dvc_repo": CONFIG.dvc_repo,
            "source": "dvc",
            "table_name": CONFIG.db_table_name,
            "embedding_model": CONFIG.embedding_model,
            "num_questions": str(len(df)),
            "created_at": pd.Timestamp.now().isoformat(),
        },
    )

    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "inputs": {"question": row["question"]},
                "expectations": {
                    "answer": row.get("expected_answer", ""),
                    "key_concepts": json.loads(row.get("key_concepts", "[]")),
                },
                "metadata": {
                    "category": row.get("category", "general"),
                    "dvc_data_version": dvc_data_version,
                },
            }
        )

    dataset.merge_records(records)
    logger.success(f"  ✓ Created new MLflow dataset: {dataset.name}")
    return dataset


@scorer
def concept_coverage(outputs: str, expectations: dict) -> Feedback:
    """Custom scorer: Check if key concepts are covered."""
    key_concepts = expectations.get("key_concepts", [])
    if not key_concepts:
        return Feedback(value=1.0, rationale="No key concepts specified")

    output_lower = outputs.lower()
    covered = [c for c in key_concepts if c.lower() in output_lower]
    score = len(covered) / len(key_concepts)
    return Feedback(
        value=score,
        rationale=f"Covered {len(covered)}/{len(key_concepts)}: {covered}",
    )


def create_llm_judge(base_url: str, model: str):
    """Create LLM-as-a-judge using our own served model."""

    @scorer
    def answer_quality_judge(
        outputs: str, inputs: dict, expectations: dict
    ) -> Feedback:
        question = inputs.get("question", "")
        expected = expectations.get("answer", "")

        judge_prompt = f"""You are an expert evaluator for a RAG system about OpenCloudHub MLOps platform.

Question: {question}
Generated Answer: {outputs}
Expected Answer: {expected}

Evaluate if the generated answer:
1. Accurately addresses the question
2. Includes relevant technical details
3. Is consistent with the expected answer

Rate from 0.0 (completely wrong) to 1.0 (perfect).
Return ONLY a JSON with: {{"score": <float>, "rationale": "<string>"}}"""

        http_client = httpx.Client(verify=False)
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key="dummy",
            http_client=http_client,
        )

        try:
            response = llm.invoke(judge_prompt).content
            result = json.loads(response)
            return Feedback(value=float(result["score"]), rationale=result["rationale"])
        except Exception as e:
            return Feedback(value=0.5, rationale=f"Judge error: {e}")

    return answer_quality_judge


def evaluate_prompt(prompt_version: int, name: str, dataset, dvc_data_version: str):
    """Evaluate one prompt version."""
    logger.info(f"\n{'=' * 60}\nEvaluating v{prompt_version}\n{'=' * 60}")

    from src.core.database import DatabaseManager
    from src.rag.chain import RAGChain

    with DatabaseManager.create_temporary(CONFIG.db_connection_string) as db:
        rag = RAGChain(
            pg_engine=db.engine,
            db_connection_string_psycopg=CONFIG.db_connection_string_psycopg,
            table_name=CONFIG.db_table_name,
            embedding_model=CONFIG.embedding_model,
            llm_base_url=CONFIG.llm_base_url,
            llm_model=CONFIG.llm_model,
            api_key=CONFIG.api_key,
            prompt_name=name,
            prompt_version=prompt_version,
            top_k=CONFIG.db_top_k,
        )

        def predict_fn(question: str) -> str:
            return rag.invoke(question)

        with mlflow.start_run(
            run_name=f"{name}-prompt-v{prompt_version}-data-v{dvc_data_version}"
        ) as run:
            mlflow.log_param("prompt_name", name)
            mlflow.log_param("prompt_version", prompt_version)
            mlflow.log_param("embedding_model", CONFIG.embedding_model)
            mlflow.log_param("llm_model", CONFIG.llm_model)
            mlflow.log_param("table_name", CONFIG.db_table_name)
            mlflow.log_param("top_k", CONFIG.db_top_k)
            mlflow.log_param("eval_data_version", dvc_data_version)

            mlflow.set_tags(
                {
                    "prompt_version": str(prompt_version),
                    "prompt_name": name,
                    "dvc_data_version": dvc_data_version,
                }
            )

            results = evaluate(
                data=dataset,
                predict_fn=predict_fn,
                scorers=[
                    concept_coverage,
                    create_llm_judge(CONFIG.llm_base_url, CONFIG.llm_model),
                ],
            )

            metrics_dict = {}
            for key, value in results.metrics.items():
                metrics_dict[key] = value
                safe_key = key.replace("/", "_").replace(" ", "_")
                mlflow.log_metric(safe_key, value)

            logger.success("✅ Evaluation complete")
            for k, v in metrics_dict.items():
                logger.info(f"  {k}: {v:.3f}")

            return {
                "run_id": run.info.run_id,
                "prompt_version": prompt_version,
                "metrics": metrics_dict,
                "dataset_id": dataset.dataset_id,
            }


def run_evaluation(
    prompt_name: str,
    prompt_versions: list[int],
    dvc_data_version: str,
    auto_promote: bool = True,
):
    """Run evaluation with proper dataset tracking."""
    mlflow.set_experiment("RAG Prompt Evaluation")

    logger.info(f"{'=' * 60}")
    logger.info("🧪 RAG EVALUATION SETUP")
    logger.info(f"{'=' * 60}")
    logger.info(f"Prompt: {prompt_name}")
    logger.info(f"Versions: {prompt_versions}")
    logger.info(f"DVC data version: {dvc_data_version}\n")

    dataset = load_eval_dataset(dvc_data_version)

    results = []
    for version in prompt_versions:
        result = evaluate_prompt(version, prompt_name, dataset, dvc_data_version)
        results.append(result)

    logger.info(f"\n{'=' * 60}\nCOMPARISON\n{'=' * 60}")

    comparison_data = []
    for r in results:
        comparison_data.append(
            {
                "version": r["prompt_version"],
                "run_id": r["run_id"][:8],
                "concept_coverage": r["metrics"].get("concept_coverage/mean", 0),
                "llm_judge": r["metrics"].get("answer_quality_judge/mean", 0),
            }
        )

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df["composite"] = (
        comparison_df["concept_coverage"] * 0.6 + comparison_df["llm_judge"] * 0.4
    )
    logger.info(comparison_df.to_string(index=False))

    best_idx = comparison_df["composite"].idxmax()
    best_version = int(comparison_df.loc[best_idx, "version"])
    best_result = results[best_idx]

    logger.info(f"\n✓ Best version: {best_version}")
    logger.info(f"  Composite: {comparison_df.loc[best_idx, 'composite']:.3f}")

    if auto_promote:
        logger.info(f"\nPromoting version {best_version} to @champion...")
        mlflow.set_prompt_alias(prompt_name, alias="champion", version=best_version)
        logger.info("✓ Promoted!")

    return {
        "best_prompt_version": best_version,
        "best_run_id": best_result["run_id"],
        "dataset_id": best_result["dataset_id"],
        "metrics": best_result["metrics"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-name", default="readme-rag-prompt")
    parser.add_argument("--prompt-versions", type=str, default="1,2,3")  # String now
    parser.add_argument(
        "--dvc-data-version", default="opencloudhub-readmes-rag-evaluation-v1.0.0"
    )
    parser.add_argument("--auto-promote", action="store_true", default=True)

    args = parser.parse_args()

    # Parse comma-separated versions to list of ints
    prompt_versions = [int(v.strip()) for v in args.prompt_versions.split(",")]

    result = run_evaluation(
        prompt_name=args.prompt_name,
        prompt_versions=prompt_versions,
        dvc_data_version=args.dvc_data_version,
        auto_promote=args.auto_promote,
    )

    logger.info(f"\n{'=' * 60}")
    logger.success("✅ EVALUATION COMPLETE")
    logger.info(f"Best prompt version: v{result['best_prompt_version']}")
    logger.info(f"Run ID: {result['best_run_id']}")
