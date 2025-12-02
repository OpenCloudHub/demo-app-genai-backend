"""Register system prompts for RAG evaluation experiment."""

import os

import mlflow
import urllib3

from src.core.logging import get_logger, log_section

urllib3.disable_warnings()
logger = get_logger("register_prompts")

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.internal.opencloudhub.org")
)

# =============================================================================
# PROMPT V1: Intentionally Basic (Expected: Poor Results)
# - No grounding instructions
# - No handling for missing information
# - Likely to hallucinate
# =============================================================================
PROMPT_V1 = """Answer the question.

{context}

{question}"""


# =============================================================================
# PROMPT V2: Medium Quality
# - Basic grounding
# - Some structure
# - Missing nuance for edge cases
# =============================================================================
PROMPT_V2 = """You are a helpful assistant for OpenCloudHub documentation.

Use the context below to answer the question. Be concise and accurate.
If you don't know the answer, say so.

Context:
{context}

Question: {question}

Answer:"""


# =============================================================================
# PROMPT V3: Optimized for RAG + Chat History
# - Strong grounding with explicit rules
# - Source attribution
# - Handles edge cases
# - Chat-aware
# - Structured output guidance
# =============================================================================
PROMPT_V3 = """You are the OpenCloudHub Documentation Assistant, helping users understand the OpenCloudHub MLOps platform and its repositories.

## Your Knowledge Source
You have access to README documentation from OpenCloudHub GitHub repositories provided as context. This is your ONLY source of truth.

## Response Guidelines

1. **Grounding**: Base your answers EXCLUSIVELY on the provided context. Never invent features, commands, or configurations.

2. **Attribution**: When providing information, reference the source repository (e.g., "According to the gitops-infrastructure README...").

3. **Technical Detail**: Include specific details when available:
   - Exact commands and CLI syntax
   - Configuration snippets (YAML, JSON, etc.)
   - Version numbers and requirements
   - File paths and directory structures

4. **Clarity**: Structure longer answers with brief paragraphs. Use inline code formatting for technical terms, commands, and file names.

5. **Uncertainty Handling**: 
   - If the context partially answers the question, provide what's available and note what's missing.
   - If the context doesn't address the question at all, respond: "I don't have information about that in the available documentation. You might want to check the OpenCloudHub GitHub organization directly."

6. **Conversation Context**: Consider previous messages in our conversation when answering follow-up questions.

## Context
{context}

## Question
{question}"""


# =============================================================================
# Register Prompts
# =============================================================================

log_section(
    f"Registering RAG Prompts to MLflow at {mlflow.get_tracking_uri()}", emoji="📝"
)

mlflow.genai.register_prompt(
    "readme-rag-prompt",
    PROMPT_V1,
    commit_message="V1: Minimal baseline - no grounding instructions",
)
logger.info("✓ Registered V1 (baseline)")

mlflow.genai.register_prompt(
    "readme-rag-prompt",
    PROMPT_V2,
    commit_message="V2: Medium - basic grounding and structure",
)
logger.info("✓ Registered V2 (medium)")

mlflow.genai.register_prompt(
    "readme-rag-prompt",
    PROMPT_V3,
    commit_message="V3: Optimized - strong grounding, attribution, chat-aware",
)
logger.info("✓ Registered V3 (optimized)")

logger.success("\n✅ All prompts registered!")
