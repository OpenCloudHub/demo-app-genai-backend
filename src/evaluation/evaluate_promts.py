"""
MLflow evaluation with proper metrics extraction and LLM-as-judge.
"""

import argparse
import io
import json
import os

import dvc.api
import httpx
import mlflow
import pandas as pd
import urllib3
from dotenv import load_dotenv
from mlflow.entities import Feedback
from mlflow.genai import evaluate
from mlflow.genai.datasets import create_dataset, search_datasets
from mlflow.genai.scorers import scorer

from src._config import CONFIG
from src.rag.chain import RAGChain

urllib3.disable_warnings()

load_dotenv()


def load_eval_dataset(data_version: str):
    """
    Load dataset from DVC and create/reuse MLflow dataset.

    If a dataset with the same DVC version exists, reuse it.
    Otherwise, create a new one.
    """
    print(f"Loading eval dataset (DVC version: {data_version})...")

    # Get or create experiment
    exp = mlflow.get_experiment_by_name("RAG Prompt Evaluation")
    exp_id = (
        exp.experiment_id if exp else mlflow.create_experiment("RAG Prompt Evaluation")
    )

    # Search for existing dataset with same DVC version
    print("  Checking for existing dataset...")
    existing_datasets = search_datasets(
        experiment_ids=[exp_id],
        filter_string=f"tags.dvc_version = '{data_version}'",
        max_results=1,
        order_by=["created_time DESC"],  # Get most recent if multiple
    )

    if existing_datasets:
        dataset = existing_datasets[0]
        print(f"  ✓ Reusing existing dataset: {dataset.name}")
        print(f"  ✓ Dataset ID: {dataset.dataset_id}")
        print(f"  ✓ Records: {len(dataset.records)}\n")
        return dataset

    # No existing dataset found - create new one
    print("  No existing dataset found, creating new one...")

    # Load from DVC
    csv_content = dvc.api.read(
        path=CONFIG.eval_data_path,
        repo=CONFIG.dvc_repo,
        rev=data_version,
        mode="r",
    )
    df = pd.read_csv(io.StringIO(csv_content))
    print(f"  ✓ Loaded {len(df)} questions from DVC")

    # Create MLflow GenAI dataset with rich metadata
    dataset = create_dataset(
        name=f"readme-rag-eval-{data_version}",
        experiment_id=exp_id,
        tags={
            "dvc_version": data_version,
            "dvc_repo": CONFIG.dvc_repo,
            "source": "dvc",
            "table_name": CONFIG.table_name,
            "embedding_model": CONFIG.embedding_model,
            "num_questions": str(len(df)),
            "created_at": pd.Timestamp.now().isoformat(),
        },
    )

    # Convert to MLflow eval format and merge
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
                    "dvc_version": data_version,
                },
            }
        )

    dataset.merge_records(records)

    print(f"  ✓ Created new MLflow dataset: {dataset.name}")
    print(f"  ✓ Dataset ID: {dataset.dataset_id}")
    print(f"  ✓ Records: {len(records)}\n")

    return dataset


@scorer
def concept_coverage(outputs: str, expectations: dict) -> Feedback:
    """Custom scorer: Check if key concepts are covered"""
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
    """
    Create LLM-as-a-judge using your own served model!

    This uses your Qwen model as the judge instead of OpenAI.
    """

    @scorer
    def answer_quality_judge(
        outputs: str, inputs: dict, expectations: dict
    ) -> Feedback:
        """LLM judge using your own model"""
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

        # Call your own model
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
            # Parse JSON response
            result = json.loads(response)
            return Feedback(value=float(result["score"]), rationale=result["rationale"])
        except Exception as e:
            # Fallback if parsing fails
            return Feedback(value=0.5, rationale=f"Judge error: {e}")

    return answer_quality_judge


def evaluate_prompt(prompt_version: int, name: str, dataset, data_version: str):
    """Evaluate one prompt version."""
    print(f"\n{'=' * 60}\nEvaluating v{prompt_version}\n{'=' * 60}")

    rag = RAGChain(
        db_connection_string=CONFIG.connection_string,
        table_name=CONFIG.table_name,
        embedding_model=CONFIG.embedding_model,
        llm_base_url=CONFIG.llm_base_url,
        llm_model=CONFIG.llm_model,
        prompt_name=name,
        prompt_version=prompt_version,
        top_k=CONFIG.top_k,
    )

    def predict_fn(question: str) -> str:
        return rag.invoke(question)

    with mlflow.start_run(
        run_name=f"{name}-prompt-v{prompt_version}-data-v{data_version}"
    ) as run:
        # Log comprehensive parameters
        mlflow.log_param("prompt_name", name)
        mlflow.log_param("prompt_version", prompt_version)
        mlflow.log_param("embedding_model", CONFIG.embedding_model)
        mlflow.log_param("llm_model", CONFIG.llm_model)
        mlflow.log_param("table_name", CONFIG.table_name)
        mlflow.log_param("top_k", CONFIG.top_k)
        mlflow.log_param("eval_data_version", data_version)
        mlflow.log_param("dvc_repo", CONFIG.dvc_repo)
        mlflow.update_current_trace(
            tags={
                "prompt_version": str(prompt_version),
                "prompt_name": name,
                "data_version": data_version,
                "docker_image_tag": os.getenv("DOCKER_IMAGE_TAG", "local"),
            }
        )

        # Create scorers including your own LLM judge
        scorers = [
            concept_coverage,
            create_llm_judge(CONFIG.llm_base_url, CONFIG.llm_model),
        ]

        results = evaluate(
            data=dataset,
            predict_fn=predict_fn,
            scorers=scorers,
        )

        # Extract metrics properly - they have /mean suffix
        metrics_dict = {}
        for key, value in results.metrics.items():
            # Store with original key
            metrics_dict[key] = value
            # Also log to MLflow
            safe_key = key.replace("/", "_").replace(" ", "_")
            mlflow.log_metric(safe_key, value)

        # Log dataset info as artifact
        dataset_info = {
            "dataset_id": dataset.dataset_id,
            "dataset_name": dataset.name,
            "dvc_version": data_version,
            "num_questions": len(dataset.records),
        }
        mlflow.log_dict(dataset_info, "dataset_info.json")

        print("\n✓ Evaluation complete")
        print("Metrics:")
        for k, v in metrics_dict.items():
            print(f"  {k}: {v:.3f}")

        return {
            "run_id": run.info.run_id,
            "prompt_version": prompt_version,
            "metrics": metrics_dict,
            "dataset_id": dataset.dataset_id,
        }


def run_evaluation(
    prompt_name: str,
    prompt_versions: list[int],
    data_version: str,
    auto_promote: bool = True,
):
    """Run evaluation with proper dataset tracking."""
    mlflow.set_experiment("RAG Prompt Evaluation")

    print(f"\n{'=' * 60}\nRAG EVALUATION\n{'=' * 60}")
    print(f"Prompt: {prompt_name}")
    print(f"Versions: {prompt_versions}")
    print(f"DVC data version: {data_version}")
    print(f"{'=' * 60}\n")

    # Load dataset and reuse
    dataset = load_eval_dataset(data_version)

    # Evaluate all prompt_versions
    results = []
    for version in prompt_versions:
        result = evaluate_prompt(version, prompt_name, dataset, data_version)
        results.append(result)

    # Compare - use exact metric keys with /mean suffix
    print(f"\n{'=' * 60}")
    print("COMPARISON")
    print(f"{'=' * 60}")

    comparison_data = []
    for r in results:
        comparison_data.append(
            {
                "version": r["prompt_version"],
                "run_id": r["run_id"][:8],  # Shortened for display
                "concept_coverage": r["metrics"].get("concept_coverage/mean", 0),
                "llm_judge": r["metrics"].get("answer_quality_judge/mean", 0),
            }
        )

    comparison_df = pd.DataFrame(comparison_data)

    # Calculate composite score
    comparison_df["composite"] = (
        comparison_df["concept_coverage"] * 0.6 + comparison_df["llm_judge"] * 0.4
    )
    print(comparison_df.to_string(index=False))

    best_idx = comparison_df["composite"].idxmax()
    best_version = int(comparison_df.loc[best_idx, "version"])

    # Get the full result for complete info
    best_result = results[best_idx]

    print(f"\n✓ Best version: {best_version}")
    print(f"  Composite: {comparison_df.loc[best_idx, 'composite']:.3f}")
    print(f"  Concept coverage: {comparison_df.loc[best_idx, 'concept_coverage']:.3f}")
    print(f"  LLM judge: {comparison_df.loc[best_idx, 'llm_judge']:.3f}")

    if auto_promote:
        print(f"\nPromoting version {best_version} to @champion...")
        mlflow.set_prompt_alias(prompt_name, alias="champion", version=best_version)
        print("✓ Promoted!")

    return {
        "best_prompt_version": best_version,
        "best_run_id": best_result["run_id"],
        "dataset_id": best_result["dataset_id"],
        "metrics": best_result["metrics"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-name", default="readme-rag-prompt")
    parser.add_argument("--prompt-versions", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--data-version", default="opencloudhub-readmes-rag-evaluation-v1.0.0"
    )
    parser.add_argument("--auto-promote", action="store_true", default=True)

    args = parser.parse_args()

    result = run_evaluation(
        prompt_name=args.prompt_name,
        prompt_versions=args.prompt_versions,
        data_version=args.data_version,
        auto_promote=args.auto_promote,
    )

    print(f"\n{'=' * 60}")
    print("✅ EVALUATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Best prompt version: v{result['best_prompt_version']}")
    print(f"Run ID: {result['best_run_id']}")
    print(f"Dataset ID: {result['dataset_id']}")
    print("\nMetrics:")
    print(
        f"  Concept coverage: {result['metrics'].get('concept_coverage/mean', 0):.3f}"
    )
    print(f"  LLM judge: {result['metrics'].get('answer_quality_judge/mean', 0):.3f}")
    print("\nLoad in production with:")
    print(f"  prompts:/{args.prompt_name}@champion")
    print("Or specific version with:")
    print(f"  prompts:/{args.prompt_name}@{result['best_prompt_version']}")
    print(f"{'=' * 60}\n")
