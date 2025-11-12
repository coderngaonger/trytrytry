# utils/llm_client.py
from typing import List, Optional, Union
from google import genai
from google.genai.types import Content, Part

class GeminiClient:
    def __init__(self, api_key: str, endpoint: str = "https://generativelanguage.googleapis.com"):
        self.client = genai.Client(api_key=api_key, base_url=endpoint)

    # --- Text-only chat (role-based) ---
    def chat(self, model: str, messages: List[dict], temperature: float = 0.7, max_tokens: int = 2048) -> str:
        """
        messages: [{"role": "system"|"user"|"assistant", "content": "..."}, ...]
        """
        # Chuyển về dạng Content/Part theo google-genai
        contents: List[Content] = []
        for m in messages:
            role = m.get("role", "user")
            text = m.get("content", "")
            contents.append(Content(role=role, parts=[Part.from_text(text)]))

        resp = self.client.responses.generate(
            model=model,
            contents=contents,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )
        # Lấy text giản lược
        return "".join([p.text or "" for c in resp.candidates for p in (c.content.parts or [])])

    # --- Multimodal (text + image/video frames) nếu cần ---
    def multimodal(self, model: str, prompt: str, images: Optional[List[bytes]] = None,
                   temperature: float = 0.7, max_tokens: int = 2048) -> str:
        """
        images: danh sách bytes ảnh (PNG/JPEG). Video nên tách keyframes/caption tuỳ workflow.
        """
        parts: List[Part] = [Part.from_text(prompt)]
        if images:
            for data in images:
                parts.append(Part.from_bytes(data, mime_type="image/png"))
        resp = self.client.responses.generate(
            model=model,
            contents=[Content(role="user", parts=parts)],
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )
        return "".join([p.text or "" for c in resp.candidates for p in (c.content.parts or [])])
