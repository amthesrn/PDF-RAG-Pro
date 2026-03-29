"""Contextual Query Rewriting Agent for bot2."""

from __future__ import annotations

import time
from groq import Groq
from loguru import logger

from config.settings import settings
from src.models.schemas import ChatMessage, QueryInput

_REWRITE_SYSTEM_PROMPT = """You are a specialized query rewriting assistant for a document retrieval system.
Your task is to analyze the user's latest question and the preceding conversation history, then rewrite the final question to be fully self-contained, highly specific, and optimized for vector search.

Rules:
1. If the latest question uses pronouns (it, they, he, she) or refers to a previous topic implicitly, replace those references with the explicit entities from the history.
2. If the question is already self-contained, DO NOT change its core meaning, but you MAY enrich it with critical keywords.
3. Output ONLY the rewritten question. No conversational filler, no explanations, no prefix (like "Rewritten query:"). Just the exact search string.
"""

class QueryRewriter:
    """Agent that rewrites conversational queries into self-contained search queries."""

    def __init__(self) -> None:
        self._client = Groq(api_key=settings.groq_api_key)
        self._model = settings.llm_model 
    
    def rewrite(self, question: str, chat_history: list[ChatMessage]) -> QueryInput:
        """
        Rewrite the query contextually. If no history exists, returns it mostly as-is or slightly normalized.
        """
        if not chat_history:
            return QueryInput(question=question)

        history_text = "\n".join(
            f"{msg.role.upper()}: {msg.content}" for msg in chat_history[-6:]
        )
        
        user_message = f"CHAT HISTORY:\n{history_text}\n\nLATEST QUESTION: {question}\n\nREWRITTEN QUESTION:"

        logger.debug("Rewriting query based on chat history...")
        start_time = time.perf_counter()
        
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=150,
                temperature=0.0,
            )
            rewritten_q = response.choices[0].message.content.strip()
            
            # Safety fallback: if it hallucinates a long answer, just use original
            if len(rewritten_q) > 300 or "\n" in rewritten_q:
                 rewritten_q = question
                 
            # Strip potential prefixes LLMs sometimes leave despite instructions
            for prefix in ["Rewritten question:", "Rewritten query:", "Search query:"]:
                if rewritten_q.lower().startswith(prefix.lower()):
                    rewritten_q = rewritten_q[len(prefix):].strip()
            
            logger.info(f"Query rewritten: '{question}' -> '{rewritten_q}' ({(time.perf_counter() - start_time)*1000:.0f}ms)")
            return QueryInput(question=rewritten_q)
            
        except Exception as exc:
            logger.warning(f"Query rewrite failed, falling back to original query. Error: {exc}")
            return QueryInput(question=question)
