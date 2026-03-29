"""Image/diagram processing using a vision LLM (Groq).

Converts diagram/image bytes into descriptive text.
Falls back to OCR (pytesseract) if the LLM fails.
"""

from __future__ import annotations

import base64
from typing import Optional

from groq import Groq, GroqError
from loguru import logger

from config.settings import settings


_VISION_PROMPT = """You are analyzing a diagram, figure, chart, mind map, or image extracted from a study document.

Describe what you see clearly and completely. Include:
- The type of content (flowchart, mind map, table, graph, diagram, etc.)
- All visible text, labels, and numbers
- The relationships or connections shown (if any)
- Key concepts and their hierarchy

Be thorough — your description will be used to answer questions about this document.
Respond in plain text only, no markdown."""


class ImageProcessor:
    """Converts diagram/image bytes to text descriptions using a vision LLM."""

    _MIN_IMAGE_AREA: int = 5_000  # skip tiny icons/logos

    def __init__(self) -> None:
        self._client = Groq(api_key=settings.groq_api_key)
        self._model = settings.vision_model
        logger.debug(f"ImageProcessor (Groq vision) initialised | model={self._model}")

    def describe_batch(self, image_paths: list[Path], page_number: int) -> list[str]:
        """Describe multiple images from file paths."""
        descriptions = []
        for path in image_paths:
            try:
                with open(path, "rb") as f:
                    image_bytes = f.read()
                desc = self.describe(image_bytes, page_number)
                if desc:
                    descriptions.append(desc)
            except Exception as exc:
                logger.warning(f"Failed to process image {path} on page {page_number}: {exc}")
        return descriptions
        if not image_bytes:
            logger.warning(f"Empty image bytes on page {page_number} — skipping.")
            return None

        width, height = self._get_dimensions(image_bytes)
        if width * height < self._MIN_IMAGE_AREA:
            logger.debug(
                f"Skipping tiny image ({width}×{height}px) on page {page_number} — "
                f"likely an icon or logo."
            )
            return None

        description = self._describe_with_api(image_bytes, page_number)
        if description:
            return description

        logger.debug(f"Vision API failed on page {page_number} — trying OCR fallback…")
        return self._describe_with_ocr(image_bytes, page_number)

    def _describe_with_api(self, image_bytes: bytes, page_number: int) -> Optional[str]:
        try:
            b64_image = base64.b64encode(image_bytes).decode("utf-8")
            img_format = self._detect_format(image_bytes)

            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{img_format};base64,{b64_image}"
                                },
                            },
                            {"type": "text", "text": _VISION_PROMPT},
                        ],
                    }
                ],
                max_tokens=512,
                temperature=0.1,
            )

            description = response.choices[0].message.content.strip()
            if description:
                logger.debug(
                    f"  Groq vision (page {page_number}): {description[:80]}…"
                )
                return description
            return None

        except GroqError as exc:
            logger.warning(f"Groq vision API error on page {page_number}: {exc}")
            return None
        except Exception as exc:
            logger.warning(f"Unexpected vision error on page {page_number}: {exc}")
            return None

    def _describe_with_ocr(self, image_bytes: bytes, page_number: int) -> Optional[str]:
        try:
            import pytesseract
            from PIL import Image
            import io

            img = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(img).strip()

            if text:
                logger.debug(
                    f"  OCR fallback succeeded on page {page_number}: {text[:80]}…"
                )
                return f"[OCR extracted text from image]: {text}"

            logger.debug(f"  OCR found no text on page {page_number}.")
            return None

        except ImportError:
            logger.warning(
                "pytesseract is not installed — OCR fallback unavailable.\n"
                "Install: pip install pytesseract\n"
                "Also install Tesseract binary: https://github.com/UB-Mannheim/tesseract/wiki"
            )
            return None
        except Exception as exc:
            logger.warning(f"OCR fallback error on page {page_number}: {exc}")
            return None

    @staticmethod
    def _get_dimensions(image_bytes: bytes) -> tuple[int, int]:
        try:
            from PIL import Image
            import io

            with Image.open(io.BytesIO(image_bytes)) as img:
                return img.size
        except Exception:
            return 999, 999

    @staticmethod
    def _detect_format(image_bytes: bytes) -> str:
        if image_bytes[:3] == b"\xff\xd8\xff":
            return "jpeg"
        if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
            return "png"
        if image_bytes[:4] in (b"GIF8", b"GIF9"):
            return "gif"
        return "jpeg"
