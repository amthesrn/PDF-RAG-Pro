"""
config/settings.py
──────────────────
Central configuration using Pydantic v2 BaseSettings.
All values are validated at startup; bad config fails fast with clear messages.

This is the Groq configuration (active) for PDF RAG Chatbot.
"""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ══════════════════════════════════════════════════════════════════
    #  API KEY — Groq (active)
    # ══════════════════════════════════════════════════════════════════
    groq_api_key: str = Field(
        ..., description="Free Groq API key. Get at: https://console.groq.com",
    )

    # ══════════════════════════════════════════════════════════════════
    #  MODEL NAMES (Groq)
    # ══════════════════════════════════════════════════════════════════
    llm_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq LLM for answer generation",
    )
    vision_model: str = Field(
        default="llama-3.2-11b-vision-preview",
        description="Groq vision model for diagram/image description",
    )

    # ══════════════════════════════════════════════════════════════════
    #  EMBEDDINGS & RERANKER  (local)
    # ══════════════════════════════════════════════════════════════════
    embedding_model: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description="Local HuggingFace embedding model (free, ~1.3 GB download once)",
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Local cross-encoder reranker (free, ~200 MB download once)",
    )

    # ══════════════════════════════════════════════════════════════════
    #  CHUNKING
    # ══════════════════════════════════════════════════════════════════
    chunk_size: int = Field(default=700, ge=100, le=2000)
    chunk_overlap: int = Field(default=120, ge=0, le=500)

    # ══════════════════════════════════════════════════════════════════
    #  RETRIEVAL
    # ══════════════════════════════════════════════════════════════════
    top_k_retrieve: int = Field(
        default=10, ge=1, le=50, description="Chunks fetched before reranking"
    )
    top_k_rerank: int = Field(
        default=4, ge=1, le=20, description="Chunks passed to LLM after reranking"
    )

    # ══════════════════════════════════════════════════════════════════
    #  PATHS
    # ══════════════════════════════════════════════════════════════════
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")
    pdf_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "input")
    vectorstore_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data" / "vectorstore")
    log_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "logs")

    # ══════════════════════════════════════════════════════════════════
    #  CHROMADB
    # ══════════════════════════════════════════════════════════════════
    collection_name: str = Field(default="bot2_collection_v2")
    chroma_api_key: str = Field(..., description="Chroma Cloud API key from .env")
    chroma_tenant: str = Field(default="ea729bbb-3ef5-4edf-8cf4-5be3ce88ebfc")
    chroma_database: str = Field(default="bot2")

    # ══════════════════════════════════════════════════════════════════
    #  GEMINI
    # ══════════════════════════════════════════════════════════════════
    gemini_api_key: str = Field(default="")
    gemini_model: str = Field(default="models/gemini-3-flash-preview")

    # ══════════════════════════════════════════════════════════════════
    #  LLM GENERATION SETTINGS
    # ══════════════════════════════════════════════════════════════════
    llm_temperature: float = Field(
        default=0.0, ge=0.0, le=1.0, description="0.0 = deterministic = no hallucination drift",
    )
    llm_max_tokens: int = Field(default=1024, ge=128, le=4096)

    # ══════════════════════════════════════════════════════════════════
    #  VALIDATORS
    # ══════════════════════════════════════════════════════════════════

    @field_validator("groq_api_key")
    @classmethod
    def validate_groq_key(cls, v: str) -> str:
        if not v or v.strip() == "your_groq_api_key_here":
            raise ValueError(
                "GROQ_API_KEY is not set.\n"
                "  → Get a free key at https://console.groq.com\n"
                "  → Add it to your .env file as: GROQ_API_KEY=gsk_..."
            )
        if not v.startswith("gsk_"):
            raise ValueError(
                f"GROQ_API_KEY looks wrong (got: '{v[:8]}...').\n"
                "  → Groq keys start with 'gsk_'. Check your key at https://console.groq.com"
            )
        return v.strip()

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_chunk(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 700)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})."
            )
        return v

    @field_validator("top_k_rerank")
    @classmethod
    def rerank_less_than_retrieve(cls, v: int, info) -> int:
        top_k_retrieve = info.data.get("top_k_retrieve", 10)
        if v > top_k_retrieve:
            raise ValueError(
                f"top_k_rerank ({v}) cannot exceed top_k_retrieve ({top_k_retrieve})."
            )
        return v

    # ── Post-init: ensure directories exist ─────────────────────────────
    def model_post_init(self, __context) -> None:
        for path in [self.data_dir, self.pdf_dir, self.vectorstore_dir, self.log_dir]:
            path.mkdir(parents=True, exist_ok=True)


# Singleton — import this everywhere in the project
settings = Settings()
