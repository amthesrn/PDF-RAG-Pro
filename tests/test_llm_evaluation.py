"""
test_llm_evaluation.py — Structured LLM quality evaluation benchmark.
Runs curated Q&A pairs through both Groq and Gemini, scores each answer,
and saves detailed comparison results.
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger("bot2_tests.evaluation")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  CURATED EVALUATION DATASET (from Options Trading Complete Mastery Guide)
# ═══════════════════════════════════════════════════════════════════════════════

EVAL_DATASET = [
    {
        "id": "Q1",
        "question": "What is a call option and when would you use one?",
        "expected_keywords": ["right to buy", "strike price", "price rises", "premium", "underlying asset"],
        "difficulty": "easy",
    },
    {
        "id": "Q2",
        "question": "What is a put option and how does it differ from a call option?",
        "expected_keywords": ["right to sell", "price falls", "opposite", "call", "put"],
        "difficulty": "easy",
    },
    {
        "id": "Q3",
        "question": "What are the Greeks in options trading?",
        "expected_keywords": ["delta", "gamma", "theta", "vega"],
        "difficulty": "medium",
    },
    {
        "id": "Q4",
        "question": "What is the Black-Scholes model used for?",
        "expected_keywords": ["option pricing", "theoretical", "value", "model"],
        "difficulty": "medium",
    },
    {
        "id": "Q5",
        "question": "Explain the concept of implied volatility in options.",
        "expected_keywords": ["market", "expectation", "future", "volatility", "price"],
        "difficulty": "medium",
    },
    {
        "id": "Q6",
        "question": "What is a covered call strategy?",
        "expected_keywords": ["own", "stock", "sell", "call", "premium", "income"],
        "difficulty": "medium",
    },
    {
        "id": "Q7",
        "question": "What is the difference between American and European options?",
        "expected_keywords": ["exercise", "anytime", "expiration", "american", "european"],
        "difficulty": "easy",
    },
    {
        "id": "Q8",
        "question": "What is an iron condor strategy?",
        "expected_keywords": ["neutral", "range", "sell", "call", "put", "spread"],
        "difficulty": "hard",
    },
    {
        "id": "Q9",
        "question": "What are the risks of selling naked options?",
        "expected_keywords": ["unlimited", "risk", "loss", "margin", "obligation"],
        "difficulty": "hard",
    },
    {
        "id": "Q10",
        "question": "What is the purpose of quantum computing in finance?",
        "expected_keywords": [],
        "difficulty": "irrelevant",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
#  SCORING LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def score_answer(answer: str, expected_keywords: list[str], confidence: str, is_irrelevant: bool) -> dict:
    """Score a single answer against expected keywords and quality criteria."""
    answer_lower = answer.lower()

    # 1. Keyword Coverage (Completeness)
    if expected_keywords:
        found = [kw for kw in expected_keywords if kw.lower() in answer_lower]
        keyword_coverage = len(found) / len(expected_keywords)
    else:
        keyword_coverage = 0.0
        found = []

    # 2. Not-found detection for irrelevant questions
    not_found_correct = False
    if is_irrelevant:
        not_found_correct = (
            "not available" in answer_lower
            or "not found" in answer_lower
            or confidence == "not_found"
        )
        # For irrelevant questions, perfect score = correctly declining
        keyword_coverage = 1.0 if not_found_correct else 0.0

    # 3. Citation Count
    citation_pattern = r'\[doc_\d+:page_\d+\]'
    citations = re.findall(citation_pattern, answer)

    # 4. Answer Relevance (basic heuristic)
    if is_irrelevant:
        relevance = 1.0 if not_found_correct else 0.0
    elif len(answer.strip()) < 20:
        relevance = 0.2
    elif confidence == "high":
        relevance = 1.0
    elif confidence == "medium":
        relevance = 0.7
    elif confidence == "low":
        relevance = 0.4
    else:
        relevance = 0.1

    # 5. Faithfulness (presence of citations implies grounding)
    if is_irrelevant:
        faithfulness = 1.0 if not_found_correct else 0.0
    elif len(citations) > 0:
        faithfulness = min(1.0, len(citations) * 0.3 + 0.4)
    else:
        faithfulness = 0.3

    return {
        "keyword_coverage": round(keyword_coverage, 2),
        "keywords_found": found,
        "keywords_missing": [kw for kw in expected_keywords if kw.lower() not in answer_lower],
        "citation_count": len(citations),
        "citations": citations,
        "relevance": round(relevance, 2),
        "faithfulness": round(faithfulness, 2),
        "confidence": confidence,
        "not_found_correct": not_found_correct if is_irrelevant else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN EVALUATION RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_evaluation():
    """Run the full evaluation benchmark for both Groq and Gemini."""
    from src.vectorstore.chroma_store import VectorStore
    from src.retrieval.retriever import HybridRetriever
    from src.retrieval.reranker import Reranker
    from src.generation.llm_chain import RAGChain
    from src.models.schemas import RAGRequest

    store = VectorStore()
    if store.count() == 0:
        logger.error("No data indexed. Please index a PDF first.")
        return

    retriever = HybridRetriever(store)
    retriever.build_bm25_index()
    reranker = Reranker()
    chain = RAGChain(retriever, reranker)

    providers = ["gemini"]
    all_results = {}

    import json
    try:
        with open(RESULTS_DIR / "llm_evaluation_groq.json", "r", encoding="utf-8") as f:
            all_results["groq"] = json.load(f)
    except Exception:
        all_results["groq"] = []

    for provider in providers:
        logger.info("=" * 60)
        logger.info("EVALUATING PROVIDER: %s", provider.upper())
        logger.info("=" * 60)

        results = []
        for item in EVAL_DATASET:
            qid = item["id"]
            question = item["question"]
            is_irrelevant = item["difficulty"] == "irrelevant"
            logger.info("[%s] Q: %s", qid, question)

            try:
                start = time.perf_counter()
                request = RAGRequest(
                    question=question,
                    llm_provider=provider,
                    top_k=4,
                )
                response = chain.answer(request)
                elapsed_ms = (time.perf_counter() - start) * 1000

                scores = score_answer(
                    answer=response.answer,
                    expected_keywords=item["expected_keywords"],
                    confidence=response.confidence,
                    is_irrelevant=is_irrelevant,
                )

                result = {
                    "id": qid,
                    "question": question,
                    "difficulty": item["difficulty"],
                    "answer": response.answer,
                    "model": response.model_used,
                    "latency_ms": round(elapsed_ms, 1),
                    "source_pages": [c.page_number for c in response.source_chunks],
                    **scores,
                }
                results.append(result)
                logger.info("[%s] Provider=%s | Conf=%s | Coverage=%.0f%% | Citations=%d | %.0fms",
                           qid, provider, response.confidence, scores["keyword_coverage"]*100,
                           scores["citation_count"], elapsed_ms)

            except Exception as exc:
                logger.error("[%s] Provider=%s FAILED: %s", qid, provider, exc)
                results.append({
                    "id": qid,
                    "question": question,
                    "difficulty": item["difficulty"],
                    "answer": f"ERROR: {exc}",
                    "model": provider,
                    "latency_ms": 0,
                    "source_pages": [],
                    "keyword_coverage": 0,
                    "citation_count": 0,
                    "relevance": 0,
                    "faithfulness": 0,
                    "confidence": "error",
                    "error": str(exc),
                })

        # Save provider-specific results
        filename = f"llm_evaluation_{provider}.json"
        with open(RESULTS_DIR / filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Saved %s evaluation results to %s", provider, filename)

        all_results[provider] = results

    # Generate comparison report
    generate_comparison_report(all_results)


def generate_comparison_report(all_results: dict):
    """Generate a markdown comparison report summarizing both providers."""
    report_lines = [
        "# LLM Quality Evaluation Report",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n**PDF:** Options_Trading_Complete_Mastery_Guide_Claude.pdf",
        f"\n**Questions:** {len(EVAL_DATASET)}",
        "\n---\n",
        "## Summary Metrics\n",
        "| Metric | Groq (Llama 3.3) | Gemini 3 (Flash) |",
        "|--------|-------------------|------------------|",
    ]

    for provider in ["groq", "gemini"]:
        results = all_results.get(provider, [])
        if not results:
            continue

        valid = [r for r in results if r.get("confidence") != "error"]
        if not valid:
            continue

        avg_coverage = sum(r["keyword_coverage"] for r in valid) / len(valid)
        avg_relevance = sum(r["relevance"] for r in valid) / len(valid)
        avg_faithfulness = sum(r["faithfulness"] for r in valid) / len(valid)
        avg_latency = sum(r["latency_ms"] for r in valid) / len(valid)
        total_citations = sum(r["citation_count"] for r in valid)
        errors = len(results) - len(valid)

        all_results[f"{provider}_summary"] = {
            "avg_coverage": round(avg_coverage, 2),
            "avg_relevance": round(avg_relevance, 2),
            "avg_faithfulness": round(avg_faithfulness, 2),
            "avg_latency_ms": round(avg_latency, 1),
            "total_citations": total_citations,
            "errors": errors,
        }

    groq_s = all_results.get("groq_summary", {})
    gemini_s = all_results.get("gemini_summary", {})

    metrics = [
        ("Avg Keyword Coverage", "avg_coverage", "%"),
        ("Avg Relevance", "avg_relevance", ""),
        ("Avg Faithfulness", "avg_faithfulness", ""),
        ("Avg Latency", "avg_latency_ms", "ms"),
        ("Total Citations", "total_citations", ""),
        ("Errors", "errors", ""),
    ]

    for label, key, unit in metrics:
        g_val = groq_s.get(key, "N/A")
        m_val = gemini_s.get(key, "N/A")
        if isinstance(g_val, float) and unit == "%":
            g_str = f"{g_val*100:.0f}%"
            m_str = f"{m_val*100:.0f}%" if isinstance(m_val, float) else str(m_val)
        elif isinstance(g_val, float):
            g_str = f"{g_val:.2f}{unit}"
            m_str = f"{m_val:.2f}{unit}" if isinstance(m_val, float) else str(m_val)
        else:
            g_str = f"{g_val}{unit}"
            m_str = f"{m_val}{unit}"
        report_lines.append(f"| {label} | {g_str} | {m_str} |")

    report_lines.append("\n---\n")
    report_lines.append("## Per-Question Results\n")

    for item in EVAL_DATASET:
        qid = item["id"]
        report_lines.append(f"### {qid}: {item['question']}\n")
        report_lines.append(f"**Difficulty:** {item['difficulty']}\n")

        for provider in ["groq", "gemini"]:
            results = all_results.get(provider, [])
            match = next((r for r in results if r["id"] == qid), None)
            if not match:
                report_lines.append(f"**{provider.title()}:** No result\n")
                continue

            report_lines.append(f"**{provider.title()}** ({match.get('model', 'unknown')}):")
            report_lines.append(f"- Coverage: {match['keyword_coverage']*100:.0f}% | Relevance: {match['relevance']:.2f} | Faithfulness: {match['faithfulness']:.2f}")
            report_lines.append(f"- Citations: {match['citation_count']} | Latency: {match['latency_ms']:.0f}ms | Confidence: {match['confidence']}")
            answer_preview = match['answer'][:200].replace('\n', ' ')
            report_lines.append(f"- Answer: _{answer_preview}..._\n")

    report = "\n".join(report_lines)

    report_path = RESULTS_DIR / "comparison_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Comparison report saved to %s", report_path)


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(
                RESULTS_DIR.parent / "logs" / f"evaluation_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
                encoding="utf-8"
            ),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    run_evaluation()
