"""Chain that ties retrieval + reranking + generation into a single RAG step."""

from __future__ import annotations

from loguru import logger

from src.generation.generator import AnswerGenerator
from src.models.schemas import QueryInput, QueryResponse, RAGRequest
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import HybridRetriever
from src.retrieval.query_rewriter import QueryRewriter


class LLMChainError(Exception):
    """Raised when the RAG chain fails."""


class RAGChain:
    def __init__(self, retriever: HybridRetriever, reranker: Reranker) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._rewriter = QueryRewriter()

    def answer(self, request: RAGRequest) -> QueryResponse:
        """Run retrieval, reranking, and generation for a request."""
        try:
            # 1. Rewrite query contextually
            rewritten_query = self._rewriter.rewrite(
                question=request.question,
                chat_history=request.chat_history,
            )
            search_query = rewritten_query.question

            retrieval = self._retriever.retrieve(
                query=search_query,
                mode=request.retrieval_mode,
                top_k=request.top_k,
            )

            reranked = self._reranker.rerank(
                search_query,
                retrieval.chunks,
                top_k=request.top_k,
            )

            generator = AnswerGenerator(provider=request.llm_provider)
            response = generator.generate(
                QueryInput(question=request.question),
                reranked,
                retrieval_mode=request.retrieval_mode,
            )
            return response

        except Exception as exc:
            logger.error("RAG chain error: %s", exc, exc_info=True)
            raise LLMChainError(str(exc)) from exc
