from pydantic import BaseModel, Field
from typing import List, Optional
from agents.base import BaseAgent
from itertools import cycle
import re
import json
import google.generativeai as genai


class SubConcept(BaseModel):
    id: str
    title: str
    description: str
    dependencies: List[str] = Field(default_factory=list)
    key_points: List[str]


class ConceptAnalysis(BaseModel):
    main_concept: str
    sub_concepts: List[SubConcept]


class ConceptInterpreterAgent(BaseAgent):
    """
    Gemini-only implementation with API key rotation support.
    """

    def __init__(
        self,
        google_api_keys: List[str],  # ✅ CHANGED: List instead of single string
        google_api_endpoint: str,
        model: str,
        temperature: float = 0.5,
        reasoning_tokens: Optional[float] = None,
        reasoning_effort: Optional[str] = None
    ):
        # ✅ ADDED: Token tracking
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        
        # ✅ ADDED: API Key Rotation Setup
        if not google_api_keys or len(google_api_keys) == 0:
            raise ValueError("At least one Google API key is required")
        
        self.google_api_keys = google_api_keys
        self.api_key_cycle = cycle(google_api_keys)  # Round-robin iterator
        self.current_api_key = next(self.api_key_cycle)
        
        # Track usage per key
        self.key_usage = {key: 0 for key in google_api_keys}
        self.key_errors = {key: 0 for key in google_api_keys}
        
        # Configure first API key
        genai.configure(api_key=self.current_api_key)
        
        self.model = model
        self.temperature = temperature
        self.google_api_endpoint = google_api_endpoint

        # Initialize BaseAgent (if needed)
        try:
            super().__init__(model=model)
        except TypeError:
            pass

        # Setup logger
        if not hasattr(self, "logger"):
            import logging
            self.logger = logging.getLogger(self.__class__.__name__)
        
        self.logger.info(f"ConceptInterpreterAgent initialized with {len(google_api_keys)} API keys")

    # ✅ ADDED: Get next API key for rotation
    def _get_next_api_key(self) -> str:
        """
        Rotate to next API key in round-robin fashion
        """
        self.current_api_key = next(self.api_key_cycle)
        self.logger.debug(f"Switched to API key: {self.current_api_key[:10]}...")
        return self.current_api_key

    SYSTEM_PROMPT = """
You are the Concept Interpreter Agent in an AI-powered STEM animation generation pipeline.

PROJECT CONTEXT
You are the first step in a system that transforms STEM concepts into short, clear educational videos. Your output will be consumed by:
1) A Manim Agent (to create mathematical animations),
2) A Script Generator (to write narration),
3) An Audio Synthesizer (to generate speech), and
4) A Video Compositor (to assemble the final video).

YOUR ROLE
Analyze exactly the STEM concept requested by the user and produce a structured, animation-ready breakdown that is simple, concrete, and visually actionable.

SCOPE & CLARITY RULES (Very Important)
- Focus only on the concept asked. Do not introduce variants or closely related topics unless strictly required for understanding.
- Prefer plain language and short sentences. Avoid jargon when a simple term works.
- Use examples that are easy to picture and compute (small numbers, common shapes, everyday contexts).
- Each item must be showable on screen (diagrams, steps, equations, arrows, highlights, transformations).
- Keep the sequence tight: from basics → build-up → main result → quick checks.

ANALYSIS GUIDELINES

1) Concept Decomposition (3–8 sub-concepts)
   - Start with the most concrete foundation.
   - Build step-by-step to the main idea or result.
   - Every sub-concept must be visually representable in Manim or simple diagrams.
   - Show clear dependencies (which parts must appear before others).

2) Detailed Descriptions (per sub-concept)
   - Title: 2–6 words, specific and visual.
   - Description: 3–5 short sentences that explain:
     * What it is and why it matters for the main concept.
     * How it connects to the previous/next step.
     * How to show it on screen (shapes, axes, arrows, labels, motion).
     * The key "aha" insight in simple terms.

3) Key Points (4–6 per sub-concept)
   - Concrete, testable facts or relationships (numbers, formulas, directions, conditions).
   - Each should imply a visual (e.g., "draw …", "animate …", "label …", "arrow from … to …").
   - Include the minimal math/notation needed (no extra symbols).
   - Capture the "click" moment (e.g., "doubling the radius quadruples the area").

4) Pedagogical Flow
   - Concrete → abstract; simple → complex.
   - Use small, clean examples (e.g., triangles with 3–4–5; vectors with (1, 2); grids up to 5×5).
   - Include quick checkpoints (one-liners that a viewer could mentally verify).
   - Use brief, intuitive metaphors only if they directly aid the main concept (no tangents).

5) Animation-Friendly Structure
   - Specify what appears, where it appears (left/right/top), and how it moves or transforms.
   - Mention essential labels, colors (optional), and timing hints (e.g., "pause 1s after reveal").
   - Prefer consistent notation and positions across steps.
   - If equations evolve, show term-by-term transformations (highlight moving parts).

OUTPUT FORMAT (Strict)
Return ONLY valid JSON matching exactly this structure (no extra text, no backticks):
{
  "main_concept": "string",
  "sub_concepts": [
    {
      "id": "string",
      "title": "string",
      "description": "string",
      "dependencies": ["string"],
      "key_points": ["string"]
    }
  ]
}

EXAMPLE (Easy & Clear) for "Area of a Circle":
{
  "main_concept": "Area of a Circle",
  "sub_concepts": [
    {
      "id": "circle_basics",
      "title": "Circle and Radius",
      "description": "Introduce a circle with center O and radius r. Show radius as a line from O to the edge. Explain that all points on the circle are exactly r units from O. This sets the single measurement we need for area.",
      "dependencies": [],
      "key_points": [
        "Draw a circle centered at O with radius r",
        "Animate radius r as a segment from O to the boundary",
        "Label O, r, and the circumference",
        "Checkpoint: every boundary point is distance r from O"
      ]
    },
    {
      "id": "cut_and_unroll",
      "title": "Slice and Rearrange",
      "description": "Cut the circle into many equal wedges like pizza slices. Rearrange wedges alternating up and down to form a near-rectangle. This shows area by turning a curved shape into a simpler one.",
      "dependencies": ["circle_basics"],
      "key_points": [
        "Slice circle into N wedges (N large, e.g., 16)",
        "Alternate wedges to form a zig-zag rectangle",
        "Top/bottom approximate length equals half the circumference",
        "Height equals radius r"
      ]
    },
    {
      "id": "rectangle_link",
      "title": "Rectangle Approximation",
      "description": "Relate the rearranged shape to a rectangle with height r and width about half the circumference. As slices increase, the edges straighten. This makes the area easier to compute.",
      "dependencies": ["cut_and_unroll"],
      "key_points": [
        "Circumference is 2πr (used as total 'base')",
        "Half-circumference is πr (rectangle width)",
        "Rectangle height is r",
        "Approximate area becomes width × height = πr × r"
      ]
    },
    {
      "id": "final_formula",
      "title": "Area Formula",
      "description": "Take the limit as the number of slices grows. The rearranged shape becomes a true rectangle. This yields the exact area formula A = πr².",
      "dependencies": ["rectangle_link"],
      "key_points": [
        "Area = (πr) × r",
        "Therefore A = πr²",
        "Highlight r² to show area scales with radius squared",
        "Checkpoint: doubling r makes area 4×"
      ]
    }
  ]
}
"""

    def execute(self, concept: str) -> ConceptAnalysis:
        """
        Analyze a STEM concept and return structured breakdown
        """
        concept = concept.strip()
        if not concept:
            raise ValueError("Concept cannot be empty")
        if len(concept) > 500:
            raise ValueError("Concept description too long (max 500 characters)")

        concept = self._sanitize_input(concept)
        self.logger.info(f"Analyzing concept: {concept}")

        user_message = f"Analyze this STEM concept and provide a structured breakdown:\n\n{concept}"

        response_json = self._call_llm_structured(
            system_prompt=self.SYSTEM_PROMPT,
            user_message=user_message,
            temperature=0.5,
            max_retries=3,
        )

        analysis = ConceptAnalysis(**response_json)

        self.logger.info(f"Successfully analyzed concept: {analysis.main_concept}")
        self.logger.info(f"Generated {len(analysis.sub_concepts)} sub-concepts")

        return analysis

    def _sanitize_input(self, text: str) -> str:
        """Remove potentially harmful characters from input"""
        return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

    # ✅ MODIFIED: Added API key rotation with retry logic
    def _call_llm_structured(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.5,
        max_retries: int = 3,
    ) -> dict:
        """
        Call Gemini with API key rotation on rate limits
        """
        prompt = (
            system_prompt.strip()
            + "\n\nStrictly output ONLY valid JSON per the schema. No prose, no backticks."
            + "\nUser request:\n"
            + user_message.strip()
        )

        # Try with different API keys if rate limited
        max_key_attempts = len(self.google_api_keys)
        last_error = None
        
        for key_attempt in range(max_key_attempts):
            api_key = self._get_next_api_key()
            
            # Configure current API key
            genai.configure(api_key=api_key)
            
            # Try multiple times with same key
            for retry in range(max_retries):
                try:
                    gmodel = genai.GenerativeModel(self.model)
                    resp = gmodel.generate_content(
                        prompt,
                        generation_config={"temperature": temperature}
                    )
                    
                    # ✅ Track token usage if available
                    if hasattr(resp, 'usage_metadata'):
                        usage = resp.usage_metadata
                        self.prompt_tokens += getattr(usage, 'prompt_token_count', 0)
                        self.completion_tokens += getattr(usage, 'candidates_token_count', 0)
                        self.total_tokens += getattr(usage, 'total_token_count', 0)
                    
                    raw = getattr(resp, "text", "") or ""
                    data = self._safe_extract_json(raw)
                    
                    # ✅ Track successful call
                    self.key_usage[api_key] += 1
                    
                    return data
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    # Check if it's a rate limit error
                    if 'rate limit' in error_msg or 'quota' in error_msg or '429' in error_msg:
                        self.logger.warning(
                            f"⚠️ Rate limit hit for key {api_key[:10]}... "
                            f"(key attempt {key_attempt + 1}/{max_key_attempts}, "
                            f"retry {retry + 1}/{max_retries})"
                        )
                        self.key_errors[api_key] += 1
                        last_error = e
                        
                        # Break retry loop, try next key
                        break
                    else:
                        # Other errors - retry with same key
                        last_error = e
                        if retry < max_retries - 1:
                            self.logger.warning(f"Retry {retry + 1}/{max_retries}: {e}")
                            continue
                        else:
                            # Exhausted retries for this key, try next key
                            break
        
        # All keys exhausted
        raise ValueError(
            f"❌ All {max_key_attempts} API keys exhausted. "
            f"Last error: {last_error}"
        )

    @staticmethod
    def _safe_extract_json(text: str) -> dict:
        """
        Extract JSON from string (handle extra text around JSON)
        """
        text = text.strip()
        
        # Clean JSON already
        if text.startswith("{") and text.endswith("}"):
            return json.loads(text)

        # Find largest JSON block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            return json.loads(candidate)

        # No valid JSON found
        raise ValueError("No valid JSON found in LLM response")

    # ✅ ADDED: Get API usage statistics
    def get_api_stats(self) -> dict:
        """
        Get API key usage statistics
        """
        return {
            "total_calls": sum(self.key_usage.values()),
            "total_errors": sum(self.key_errors.values()),
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "keys": [
                {
                    "key": f"{key[:10]}...{key[-4:]}",
                    "usage": self.key_usage[key],
                    "errors": self.key_errors[key]
                }
                for key in self.google_api_keys
            ]
        }
    
    # ✅ ADDED: Print statistics
    def print_api_stats(self):
        """
        Print API key usage statistics
        """
        stats = self.get_api_stats()
        print("\n" + "="*60)
        print("📊 CONCEPT INTERPRETER - API KEY USAGE")
        print("="*60)
        print(f"Total API Calls: {stats['total_calls']}")
        print(f"Total Errors: {stats['total_errors']}")
        print(f"Total Tokens: {stats['total_tokens']:,}")
        print(f"  ├─ Prompt: {stats['prompt_tokens']:,}")
        print(f"  └─ Completion: {stats['completion_tokens']:,}")
        print("-"*60)
        
        for i, key_stat in enumerate(stats['keys'], 1):
            print(f"Key {i} ({key_stat['key']}): "
                  f"{key_stat['usage']} calls, "
                  f"{key_stat['errors']} errors")
        print("="*60 + "\n")