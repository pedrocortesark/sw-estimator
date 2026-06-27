"""RAGAS evaluation script for Session 11.

Evaluates the RAG pipeline using 4 metrics:
- faithfulness: Are the claims in the answer supported by the contexts?
- answer_relevancy: Is the answer relevant to the question?
- context_precision: Are the retrieved contexts relevant to the question?
- context_recall: Does the answer cover the ground truth?

Uses OpenAI for embeddings (text-embedding-3-small) and as LLM judge (gpt-4o-mini).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.dataset_schema import SingleTurnSample

from src.core.config import get_settings
from src.generation.rag.context_assembler import build_context_block, truncate_to_token_budget
from src.generation.rag.retriever import search_chunks
from src.dependencies import get_embedder, get_token_encoder


GOLDEN_SET_PATH = Path(__file__).parent / "golden_set_with_ground_truth.json"


def format_estimate_as_text(estimate) -> str:
    """Convert Estimate object to human-readable text for RAGAS evaluation."""
    lines = []
    
    if estimate.total_engineer_days is not None:
        lines.append(f"Total engineer-days: {estimate.total_engineer_days}")
    
    if estimate.duration_weeks is not None:
        lines.append(f"Duration: {estimate.duration_weeks} weeks")
    
    lines.append(f"Confidence: {estimate.confidence}")
    lines.append("")
    
    if estimate.modules:
        lines.append("Modules:")
        for module in estimate.modules:
            lines.append(f"  - {module.name}")
            if module.description:
                lines.append(f"    {module.description}")
            for task in module.tasks:
                lines.append(f"    • {task.name}: {task.engineer_days} days")
                if task.sources:
                    source_ids = [str(s.chunk_id) for s in task.sources]
                    lines.append(f"      Sources: {', '.join(source_ids)}")
    
    if estimate.assumptions:
        lines.append("")
        lines.append("Assumptions:")
        for assumption in estimate.assumptions:
            lines.append(f"  - {assumption.description} (impact: {assumption.impact})")
    
    lines.append("")
    lines.append(f"Reasoning: {estimate.reasoning}")
    
    return "\n".join(lines)


def format_ground_truth_as_text(ground_truth: dict) -> str:
    """Convert ground_truth dict to human-readable text for RAGAS evaluation."""
    lines = []
    
    if "total_engineer_days" in ground_truth:
        lines.append(f"Total engineer-days: {ground_truth['total_engineer_days']}")
    
    if "modules" in ground_truth:
        lines.append("")
        lines.append("Modules:")
        for module in ground_truth["modules"]:
            lines.append(f"  - {module['name']}")
            for task in module.get("tasks", []):
                lines.append(f"    • {task['name']}: {task['engineer_days']} days")
    
    if "confidence" in ground_truth:
        lines.append("")
        lines.append(f"Confidence: {ground_truth['confidence']}")
    
    if "reasoning" in ground_truth:
        lines.append("")
        lines.append(f"Reasoning: {ground_truth['reasoning']}")
    
    return "\n".join(lines)


async def get_retrieved_contexts(query: str, relevant_budgets: list[str]) -> list[str]:
    """Retrieve contexts for a query using the pipeline."""
    settings = get_settings()
    
    # Embed the query
    embedder = get_embedder()
    if embedder is None:
        raise RuntimeError("Embedder not available (no OpenAI key)")
    
    query_embedding = await asyncio.to_thread(embedder.embed_one, query)
    
    # Retrieve chunks
    retrieval = await search_chunks(
        query_embedding,
        top_k=settings.retrieval_top_k,
        distance_threshold=settings.retrieval_distance_threshold,
    )
    
    if retrieval.low_confidence:
        return []
    
    # Return chunk contents as contexts
    return [chunk.content for chunk in retrieval.chunks]


async def get_pipeline_answer(query: str) -> tuple[str, list[str]]:
    """Run the full pipeline and return (answer_text, contexts)."""
    settings = get_settings()
    
    # Embed the query
    embedder = get_embedder()
    if embedder is None:
        raise RuntimeError("Embedder not available (no OpenAI key)")
    
    query_embedding = await asyncio.to_thread(embedder.embed_one, query)
    
    # Retrieve chunks
    retrieval = await search_chunks(
        query_embedding,
        top_k=settings.retrieval_top_k,
        distance_threshold=settings.retrieval_distance_threshold,
    )
    
    if retrieval.low_confidence:
        return "Insufficient context to generate estimate.", []
    
    # Truncate to token budget
    encoder = get_token_encoder()
    kept = truncate_to_token_budget(retrieval.chunks, settings.max_context_tokens, encoder)
    
    # Build context block
    context_block = build_context_block(kept)
    
    # Generate estimate
    from src.generation.rag.estimator import generate_estimate
    from src.generation.rag.query_reformulator import reformulate_query
    
    # Reformulate query (simplified - just use the query as-is for evaluation)
    from src.generation.rag.schemas import EstimationQuery
    structured_query = EstimationQuery(
        function=query,
        technologies=[],
        sector=None,
        scale="medium",
        country=None,
        regulations=[],
        constraints=[],
    )
    
    estimate = await generate_estimate(context_block, structured_query=structured_query)
    
    # Format as text
    answer_text = format_estimate_as_text(estimate)
    contexts = [chunk.content for chunk in kept]
    
    return answer_text, contexts


async def evaluate_with_ragas():
    """Run RAGAS evaluation on the golden set."""
    # Load golden set
    with open(GOLDEN_SET_PATH) as f:
        golden_set = json.load(f)
    
    print(f"Loaded {len(golden_set)} queries from golden set")
    
    # Configure RAGAS with OpenAI
    settings = get_settings()
    
    # Initialize LLM and embeddings
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=settings.openai_api_key,
    )
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=settings.openai_api_key,
    )
    
    # Prepare evaluation dataset
    samples = []
    
    for item in golden_set:
        query_id = item["query_id"]
        query = item["query"]
        ground_truth_dict = item["ground_truth"]
        
        print(f"\nProcessing {query_id}: {query[:60]}...")
        
        try:
            # Get pipeline answer and contexts
            answer_text, contexts = await get_pipeline_answer(query)
            
            # Format ground truth
            ground_truth_text = format_ground_truth_as_text(ground_truth_dict)
            
            # Create SingleTurnSample
            sample = SingleTurnSample(
                user_input=query,
                response=answer_text,
                retrieved_contexts=contexts,
                reference=ground_truth_text,
            )
            samples.append(sample)
            
            print(f"  ✓ Generated answer ({len(answer_text)} chars)")
            print(f"  ✓ Retrieved {len(contexts)} contexts")
        except Exception as e:
            print(f"  ✗ Failed to generate answer: {e}")
            # Use a placeholder answer to continue evaluation
            sample = SingleTurnSample(
                user_input=query,
                response="Failed to generate estimate due to pipeline error.",
                retrieved_contexts=[],
                reference=format_ground_truth_as_text(ground_truth_dict),
            )
            samples.append(sample)
    
    print(f"\n{'='*80}")
    print("Running RAGAS evaluation...")
    print(f"{'='*80}\n")
    
    # Create EvaluationDataset
    dataset = EvaluationDataset(samples=samples)
    
    # Run evaluation
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=llm,
        embeddings=embeddings,
    )
    
    # Print results
    print(f"\n{'='*80}")
    print("RAGAS EVALUATION RESULTS")
    print(f"{'='*80}\n")
    
    # Convert to pandas DataFrame to extract metrics
    df = result.to_pandas()
    
    # Calculate mean for each metric
    metrics = {}
    for col in df.columns:
        if col not in ['user_input', 'response', 'retrieved_contexts', 'reference']:
            try:
                metrics[col] = df[col].mean()
                print(f"{col}: {metrics[col]:.4f}")
            except (TypeError, ValueError):
                pass
    
    # Save detailed results
    output_path = Path(__file__).parent / "ragas_evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "metrics": metrics,
                "per_query": [
                    {
                        "question": sample.user_input[:100],
                        "answer": sample.response[:200],
                        "num_contexts": len(sample.retrieved_contexts),
                    }
                    for sample in samples
                ],
            },
            f,
            indent=2,
        )
    
    print(f"\n✓ Detailed results saved to {output_path}")
    
    return result


if __name__ == "__main__":
    asyncio.run(evaluate_with_ragas())
