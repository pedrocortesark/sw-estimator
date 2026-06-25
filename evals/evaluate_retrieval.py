"""Evaluación de las 4 configuraciones de retrieval contra el golden set.

Mide:
- Precision@5: proporción de resultados relevantes en los top-5
- Latencia: tiempo de ejecución en milisegundos

Configuraciones:
A: Vectorial + No Reranking
B: Híbrida + No Reranking
C: Vectorial + Sí Reranking
D: Híbrida + Sí Reranking
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from openai import OpenAI

from evals.golden_queries import GOLDEN_SET
from src.core.config import get_settings
from src.generation.rag.retrieval.pipeline import retrieve
from src.rag.embedding.embedder import OpenAIEmbedder


@dataclass
class EvalResult:
    """Resultado de una evaluación individual."""
    config_name: str
    query_id: str
    precision_at_5: float
    latency_ms: float
    retrieved_budgets: list[str]
    relevant_budgets: list[str]


async def evaluate_configuration(
    embedder: OpenAIEmbedder,
    query: str,
    relevant_budgets: list[str],
    search_mode: str,
    rerank: bool,
    config_name: str,
    query_id: str,
) -> EvalResult:
    """Evalúa una configuración contra una consulta."""
    query_embedding = embedder.embed_one(query)
    
    start = time.perf_counter()
    result = await retrieve(
        query_embedding=query_embedding,
        query_text=query,
        search_mode=search_mode,
        rerank=rerank,
        top_k=5,
        recall_k=50 if rerank else 5,
        rerank_top_n=5,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    
    # Extraer budget_ids de los resultados
    retrieved_budgets = []
    for chunk in result.chunks:
        if chunk.budget_id and chunk.budget_id not in retrieved_budgets:
            retrieved_budgets.append(chunk.budget_id)
    
    # Calcular precision@5
    relevant_set = set(relevant_budgets)
    retrieved_set = set(retrieved_budgets[:5])
    relevant_retrieved = len(relevant_set & retrieved_set)
    precision_at_5 = relevant_retrieved / min(5, len(relevant_budgets))
    
    return EvalResult(
        config_name=config_name,
        query_id=query_id,
        precision_at_5=precision_at_5,
        latency_ms=latency_ms,
        retrieved_budgets=retrieved_budgets[:5],
        relevant_budgets=relevant_budgets,
    )


async def run_evaluation():
    """Ejecuta la evaluación completa."""
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    embedder = OpenAIEmbedder(client=client)
    
    configurations = [
        ("A: Vector + No Rerank", "vector", False),
        ("B: Hybrid + No Rerank", "hybrid", False),
        ("C: Vector + Rerank", "vector", True),
        ("D: Hybrid + Rerank", "hybrid", True),
    ]
    
    all_results: list[EvalResult] = []
    
    print("=" * 100)
    print("EVALUACIÓN DE CONFIGURACIONES DE RETRIEVAL")
    print("=" * 100)
    print()
    
    for config_name, search_mode, rerank in configurations:
        print(f"\n{'='*100}")
        print(f"Configuración: {config_name}")
        print(f"{'='*100}")
        
        for item in GOLDEN_SET:
            result = await evaluate_configuration(
                embedder=embedder,
                query=item["query"],
                relevant_budgets=item["relevant_budgets"],
                search_mode=search_mode,
                rerank=rerank,
                config_name=config_name,
                query_id=item["query_id"],
            )
            all_results.append(result)
            
            print(f"\n{item['query_id']}: {item['description']}")
            print(f"  Relevantes: {', '.join(item['relevant_budgets'])}")
            print(f"  Recuperados: {', '.join(result.retrieved_budgets)}")
            print(f"  Precision@5: {result.precision_at_5:.2%}")
            print(f"  Latencia: {result.latency_ms:.1f}ms")
    
    # Generar tabla comparativa
    print("\n" + "=" * 100)
    print("TABLA COMPARATIVA")
    print("=" * 100)
    print()
    
    # Cabecera
    print(f"{'Configuración':<25} {'Query':<8} {'Precision@5':<12} {'Latencia':<12}")
    print("-" * 100)
    
    # Filas
    for result in all_results:
        print(f"{result.config_name:<25} {result.query_id:<8} {result.precision_at_5:<12.2%} {result.latency_ms:<12.1f}ms")
    
    # Estadísticas por configuración
    print("\n" + "=" * 100)
    print("ESTADÍSTICAS POR CONFIGURACIÓN")
    print("=" * 100)
    print()
    
    for config_name, _, _ in configurations:
        config_results = [r for r in all_results if r.config_name == config_name]
        avg_precision = sum(r.precision_at_5 for r in config_results) / len(config_results)
        avg_latency = sum(r.latency_ms for r in config_results) / len(config_results)
        
        print(f"{config_name}")
        print(f"  Precision@5 promedio: {avg_precision:.2%}")
        print(f"  Latencia promedio: {avg_latency:.1f}ms")
        print()
    
    # Tabla resumen
    print("=" * 100)
    print("RESUMEN EJECUTIVO")
    print("=" * 100)
    print()
    print(f"{'Configuración':<25} {'Precision@5':<15} {'Latencia':<15}")
    print("-" * 60)
    
    for config_name, _, _ in configurations:
        config_results = [r for r in all_results if r.config_name == config_name]
        avg_precision = sum(r.precision_at_5 for r in config_results) / len(config_results)
        avg_latency = sum(r.latency_ms for r in config_results) / len(config_results)
        
        print(f"{config_name:<25} {avg_precision:<15.2%} {avg_latency:<15.1f}ms")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
