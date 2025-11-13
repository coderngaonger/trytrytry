import logging
import time
import re
import json
import tempfile
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle

import google.generativeai as genai

from agents.concept_interpreter import ConceptAnalysis, SubConcept
from agents.manim_models import (
    ScenePlan, SceneAction, ManimSceneCode, RenderResult,
    AnimationResult, AnimationConfig, AnimationMetadata
)
from rendering.manim_renderer import ManimRenderer


class ManimAgent:
    """
    Manim Agent: Transforms structured concept analysis into visual animations
    using Gemini API for scene planning and Manim code generation.
    """

    def __init__(
        self,
        google_api_keys: List[str],  # ✅ Multiple keys
        google_api_endpoint: Optional[str] = None,
        model: str = "gemini-1.5-pro",
        output_dir: Path = Path("output"),
        config: Optional[AnimationConfig] = None,
        reasoning_tokens: Optional[float] = None,
        reasoning_effort: Optional[str] = None
    ):
        # ✅ Token tracking
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        
        # ✅ API Key Rotation Setup
        if not google_api_keys or len(google_api_keys) == 0:
            raise ValueError("At least one Google API key is required")
        
        self.google_api_keys = google_api_keys
        self.api_key_cycle = cycle(google_api_keys)
        self.current_api_key = next(self.api_key_cycle)
        
        # Track usage per key
        self.key_usage = {key: 0 for key in google_api_keys}
        self.key_errors = {key: 0 for key in google_api_keys}
        
        # Configure first API key
        genai.configure(api_key=self.current_api_key)
        
        self.model = model
        self.google_api_endpoint = google_api_endpoint
        self.output_dir = Path(output_dir)
        self.config = config or AnimationConfig()

        # Initialize renderer
        self.renderer = ManimRenderer(
            output_dir=self.output_dir / "scenes",
            quality=self.config.quality,
            background_color=self.config.background_color,
            timeout=self.config.render_timeout,
            max_retries=self.config.max_retries_per_scene
        )

        # Ensure output directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "scene_codes").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "animations").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "scenes").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "scene_plans").mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"ManimAgent initialized with {len(google_api_keys)} API keys")

    def _get_next_api_key(self) -> str:
        """Rotate to next API key"""
        self.current_api_key = next(self.api_key_cycle)
        return self.current_api_key

    # ===== PROMPTS (giữ nguyên) =====
    SCENE_PLANNING_PROMPT = """You are a Manim Scene Planning Agent for an educational STEM animation system.

**TASK**: Create detailed scene plans for animating STEM concepts using Manim (Mathematical Animation Engine).

**INPUT CONCEPT ANALYSIS**:
{concept_analysis}

**ANIMATION GUIDELINES**:

1. **Scene Structure**:
   - Create 1–2 scenes per sub-concept (maximum 8 scenes total)
   - Each scene should be 30–45 seconds long
   - Build scenes logically following sub-concept dependencies
   - Start with foundations, progressively add complexity

2. **Visual Design**:
   - Use clear, educational visual style (dark background, bright elements)
   - Include mathematical notation, equations, diagrams
   - Show relationships and transformations visually
   - Use color coding consistently:
     - Blue (#3B82F6) for known/assumed quantities
     - Green (#22C55E) for newly introduced concepts
     - Red (#EF4444) for key/important results or warnings

3. **Consistency & Continuity (VERY IMPORTANT)**:
   - If an illustrative example is used to demonstrate the concept (e.g., a specific array for a sorting algorithm, a fixed probability scenario, a single graph), **use the exact same example and values across all scenes** unless a scene explicitly explores a variant. 
   - Keep element IDs and targets stable across scenes (e.g., "example_array", "disease_prevalence_box") to preserve continuity.
   - Reuse and transform existing elements instead of recreating them where possible.

4. **Animation Types**:
   - write / create: introduce text, equations, axes, or diagrams
   - transform / replace: mathematical transformations, substitutions, rearrangements
   - fade_in / fade_out: introduce or remove elements
   - move / highlight: focus attention
   - grow / shrink: emphasize scale or importance
   - **wait**: insert a timed pause with nothing changing on screen (used for narration beats)

5. **Pacing & Narration Cues**:
   - Animations should be **slow and deliberate**. After each significant action (write, transform, highlight, etc.), insert a **wait** action of 1.5–3.0 seconds for narration.
   - Typical durations (guideline, adjust as needed):
     - write/create (short text/equation): 4–5s
     - transform/replace (equation/diagram): 8-19s
     - move/highlight: 3-5s
     - fade_in/out: 2-5s
     - wait (narration): 2-4s

6. **Educational Flow**:
   - Start with context/overview
   - Introduce new elements step-by-step
   - Show relationships and connections visually
   - End with key takeaways or summaries, keeping the same example visible to reinforce learning

7. **Element Naming**:
   - Use descriptive, stable targets (e.g., "bayes_equation", "likelihood_label", "frequency_grid") reused across scenes.
   - When transforming, specify `"parameters": {{"from": "<old_target>", "to": "<new_target>"}}` where helpful.

**OUTPUT FORMAT**:
Return ONLY valid JSON matching this exact structure:
{{{{
    "scene_plans": [
        {{{{
            "id": "string",
            "title": "string",
            "description": "string",
            "sub_concept_id": "string",
            "actions": [
                {{{{
                    "action_type": "string",
                    "element_type": "string",
                    "description": "string",
                    "target": "string",
                    "duration": number,
                    "parameters": {{{{}}}}
                }}}}
            ],
            "scene_dependencies": ["string"]
        }}}}
    ]
}}}}

CRITICAL RULES:
- "easing" must be INSIDE "parameters" object, not at action root level
- All fields are required except "parameters" which can be empty {{{{}}}}
- "duration" must be a number, not a string
- "scene_dependencies" must be an array, even if empty []

Generate scene plans that will create clear, educational animations for the given concept, using a single consistent example across scenes, with slow pacing and explicit narration pauses after each major action."""
    CODE_GENERATION_PROMPT = """You are a Manim Code Generation Agent for creating **very simple, 2D educational STEM animations**.

    **TASK**: From the single **SCENE PLAN** below (one scene at a time), generate complete, executable Manim code for **Manim Community Edition v0.19**.

    **SCENE PLAN (SINGLE SCENE ONLY)**:
    {scene_plan}

    **CLASS NAME**: {class_name}
    **TARGET DURATION (approx.)**: {target_duration} seconds

    ============================================================
    SIMPLE 2D-ONLY MODE (STRICT)
    ============================================================

    0) **Hard Limits (Do Not Violate)**
    - **2D only**: No 3D classes/cameras/surfaces/axes3D
    - **Exactly one scene class** named **{class_name}**, inheriting from `Scene`
    - **One file**, **one class**, **one `construct(self)`** method
    - **No updaters**, **No ValueTracker / DecimalNumber**

    1) **Imports**
    - Always start with: `from manim import *`
    - Optionally: `import numpy as np` **only if actually used**

    2) **Element Types & Mapping**
    - Use **`Text`** for plain text, **`MathTex`** for math, **`Tex`** for LaTeX
    - Allowed 2D mobjects: `Dot, Line, Arrow, Vector, Circle, Square, Rectangle, Triangle, NumberPlane, Axes, Brace, SurroundingRectangle, Text, MathTex, Tex, VGroup`

    3) **Action → Code Mapping**
    - `"write"` → `self.play(Write(obj))`
    - `"create"` → `self.play(Create(obj))`
    - `"fade_in"` / `"fade_out"` → `FadeIn(obj)` / `FadeOut(obj)`
    - `"transform"` → `Transform(old, new)`
    - `"move"` → `obj.animate.shift(DIR * amount)`
    - `"highlight"` → `Indicate(obj)` or `SurroundingRectangle(obj)`
    - `"wait"` → `self.wait(duration)`

    4) **Using `target` as Variable Names**
    - Use the plan's `"target"` as Python variable name (sanitize to snake_case)
    - Reuse same variable to transform/indicate

    5) **Parameters Handling**
    - Map common fields: `{{"text": "..."}}`, `{{"equation": "..."}}`, `{{"color": BUILTIN_COLOR}}`
    - **Colors**: `WHITE, BLACK, RED, GREEN, BLUE, YELLOW, ORANGE, PURPLE, PINK` (NO CYAN)
    - Replace hex colors with closest built-in

    6) **CRITICAL: Positioning & Layout Methods (CORRECT USAGE)**
    
    **Centering objects:**
    - ✅ CORRECT: `obj.move_to(ORIGIN)` - centers object at origin
    - ✅ CORRECT: `obj.to_edge(UP)` - moves to top edge
    - ✅ CORRECT: `obj.to_edge(DOWN)` - moves to bottom edge
    - ✅ CORRECT: `obj.to_edge(LEFT)` - moves to left edge
    - ✅ CORRECT: `obj.to_edge(RIGHT)` - moves to right edge
    - ✅ CORRECT: `obj.to_corner(UL)` - moves to upper-left corner
    - ❌ WRONG: `obj.to_center()` - DOES NOT EXIST
    - ❌ WRONG: `obj.center()` - DOES NOT EXIST
    
    **Relative positioning:**
    - ✅ CORRECT: `obj2.next_to(obj1, DOWN)` - place obj2 below obj1
    - ✅ CORRECT: `obj2.next_to(obj1, RIGHT)` - place obj2 to right of obj1
    - ✅ CORRECT: `obj.shift(UP * 2)` - move up by 2 units
    - ✅ CORRECT: `obj.shift(LEFT * 1.5)` - move left by 1.5 units
    
    **Alignment:**
    - ✅ CORRECT: `obj.align_to(other, UP)` - align top edges
    - ✅ CORRECT: `VGroup(obj1, obj2).arrange(DOWN)` - stack vertically
    - ✅ CORRECT: `VGroup(obj1, obj2).arrange(RIGHT)` - arrange horizontally

    7) **CRITICAL: VGroup Methods (CORRECT USAGE)**
    
    **Checking if VGroup is empty:**
    - ✅ CORRECT: `if len(vgroup) == 0:`
    - ✅ CORRECT: `if not vgroup.submobjects:`
    - ❌ WRONG: `if vgroup.is_empty():` - DOES NOT EXIST
    
    **Adding/removing from VGroup:**
    - ✅ CORRECT: `vgroup.add(obj)`
    - ✅ CORRECT: `vgroup.remove(obj)`
    - ✅ CORRECT: `VGroup(obj1, obj2, obj3)` - create with objects
    
    **Arranging VGroup:**
    - ✅ CORRECT: `vgroup.arrange(DOWN, buff=0.5)` - vertical spacing
    - ✅ CORRECT: `vgroup.arrange(RIGHT, buff=0.5)` - horizontal spacing

    8) **CRITICAL: Axes and Graphs**
    - When creating axes: `axes = Axes(x_range=[min, max, step], y_range=[min, max, step])`
    - When plotting graphs: `graph = axes.plot(lambda x: function, x_range=[min, max], color=COLOR)`
    - **NEVER use** `axes.get_graph()` - use `axes.plot()` instead
    - Example:
    ```python
        axes = Axes(x_range=[-3, 3, 1], y_range=[-1, 9, 1])
        parabola = axes.plot(lambda x: x**2, x_range=[-3, 3], color=BLUE)
    ```

    9) **Pacing & Narration**
    - After every significant action, insert pause: `self.wait(1-3)`
    - If explicit `"wait"` action, honor its duration

    10) **Layout & Anti-overlap**
    - Keep simple, readable layout
    - Use `next_to`, `to_edge`, small `shift` values
    - Keep font sizes moderate (36–48)
    - Keep explanation text, calculations, and visualizations separate

    11) **Flow**
    - Brief title (2–3s), step-by-step reveal
    - End with `self.wait(2)`

    12) **Robustness**
    - Ensure each mobject is created before transforming
    - Don't reference objects after `FadeOut` unless re-created
    - Make sure code matches the concept being visualized
    - **ONLY use documented Manim v0.19 methods** - no invented methods!
    - When unsure, use simple basic methods like `move_to()`, `shift()`, `next_to()`

    13) **DO NOT INCLUDE BACKTICKS (``) IN YOUR CODE, EVER!**

    14) **Unicode Characters**
    - You can use Unicode math symbols (∇, ∂, ∫, etc.) in Text/MathTex
    - Always include `# -*- coding: utf-8 -*-` at top of file
    - Do NOT use special Unicode symbols (×, ÷, →, ⇒, ∙, etc.).
    - Use plain ASCII only (e.g., "*" instead of "×").


    ============================================================
    COMMON MISTAKES TO AVOID
    ============================================================
    ❌ obj.to_center() → ✅ obj.move_to(ORIGIN)
    ❌ obj.center() → ✅ obj.move_to(ORIGIN)
    ❌ vgroup.is_empty() → ✅ len(vgroup) == 0
    ❌ axes.get_graph() → ✅ axes.plot()
    ❌ obj.set_center() → ✅ obj.move_to(point)
    ❌ Using CYAN color → ✅ Use BLUE or TEAL instead

    ============================================================
    OUTPUT FORMAT (MANDATORY)
    ============================================================
    <manim>
    # -*- coding: utf-8 -*-
    from manim import *

    class {class_name}(Scene):
        def construct(self):
            # Your code here
            self.wait(2)
    </manim>
    """

    def execute(self, concept_analysis: ConceptAnalysis) -> AnimationResult:
        """Generate animation from concept analysis"""
        start_time = time.time()
        self.logger.info(f"Starting animation generation for: {concept_analysis.main_concept}")

        try:
            # Step 1: Generate scene plans
            scene_plans, response_json = self._generate_scene_plans(concept_analysis)
            self.logger.info(f"Generated {len(scene_plans)} scene plans")

            if not scene_plans:
                raise ValueError("No valid scene plans generated")

            # Save scene plans for debugging
            self._save_scene_plans(scene_plans, concept_analysis, response_json)

            # Step 2: Generate Manim code for each scene
            scene_codes = self._generate_scene_codes(scene_plans)
            self.logger.info(f"Generated code for {len(scene_codes)} scenes")

            # Step 3: Render each scene
            render_results = self._render_scenes(scene_codes)
            successful_renders = [r for r in render_results if r.success]
            self.logger.info(f"Successfully rendered {len(successful_renders)}/{len(render_results)} scenes")

            # Step 4: Concatenate scenes into single animation
            if successful_renders:
                silent_animation_path = self._concatenate_scenes(successful_renders)
            else:
                silent_animation_path = None

            # Calculate timing
            generation_time = time.time() - start_time
            total_render_time = sum(r.render_time or 0 for r in render_results)

            # Create result
            result = AnimationResult(
                success=len(successful_renders) > 0,
                concept_id=concept_analysis.main_concept.lower().replace(" ", "_"),
                total_duration=sum(r.duration for r in successful_renders if r.duration),
                scene_count=len(scene_plans),
                silent_animation_path=str(silent_animation_path) if silent_animation_path else None,
                scene_plan=scene_plans,
                scene_codes=scene_codes,
                render_results=render_results,
                generation_time=generation_time,
                total_render_time=total_render_time,
                models_used={"reasoning": self.model},
                token_usage=self.get_token_usage()
            )

            self.logger.info(f"Animation generation completed in {generation_time:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"Animation generation failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            return AnimationResult(
                success=False,
                concept_id=concept_analysis.main_concept.lower().replace(" ", "_"),
                scene_count=0,
                error_message=str(e),
                scene_plan=[],
                scene_codes=[],
                render_results=[],
                generation_time=time.time() - start_time,
                models_used={"reasoning": self.model},
                token_usage=self.get_token_usage()
            )

    def _generate_scene_plans(self, concept_analysis: ConceptAnalysis) -> tuple[List[ScenePlan], Dict[str, Any]]:
        """Generate scene plans from concept analysis using Gemini"""
        
        # ✅ Format prompt with concept analysis
        formatted_system_prompt = self.SCENE_PLANNING_PROMPT.format(
            concept_analysis=json.dumps(concept_analysis.model_dump(), indent=2)
        )
        
        user_message = (
            "Create detailed scene plans for animating this STEM concept. "
            "Follow the animation guidelines strictly. "
            "Output ONLY valid JSON with no additional text."
        )
        
        self.logger.info(f"Generating scene plans for: {concept_analysis.main_concept}")
        self.logger.info(f"Sub-concepts to animate: {len(concept_analysis.sub_concepts)}")
        for i, sc in enumerate(concept_analysis.sub_concepts, 1):
            self.logger.info(f"  {i}. {sc.title}")

        try:
            response_json = self._call_llm_structured(
                system_prompt=formatted_system_prompt,
                user_message=user_message,
                temperature=self.config.temperature,
                max_retries=3
            )

            # Parse and validate scene plans
            scene_plans = []
            raw_plans = response_json.get("scene_plans", [])
            
            self.logger.info(f"Received {len(raw_plans)} raw scene plans from LLM")
            
            for plan_data in raw_plans:
                try:
                    scene_plan = ScenePlan(**plan_data)
                    scene_plans.append(scene_plan)
                except Exception as e:
                    self.logger.warning(f"Invalid scene plan data: {e}")
                    self.logger.debug(f"Plan data: {plan_data}")
                    continue

            self.logger.info(f"Generated {len(scene_plans)} valid scene plans")
            
            if not scene_plans:
                raise ValueError("No valid scene plans generated after validation")

            return scene_plans, response_json

        except Exception as e:
            self.logger.error(f"Scene planning failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Failed to generate scene plans: {e}")

    def _call_llm_structured(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.5,
        max_retries: int = 3,
    ) -> dict:
        """
        Call Gemini API with rotation and return structured JSON
        """
        prompt = (
            system_prompt.strip()
            + "\n\nStrictly output ONLY valid JSON per the schema. No prose, no backticks."
            + "\nUser request:\n"
            + user_message.strip()
        )

        max_key_attempts = len(self.google_api_keys)
        last_error = None
        
        for key_attempt in range(max_key_attempts):
            api_key = self._get_next_api_key()
            genai.configure(api_key=api_key)
            
            for retry in range(max_retries):
                try:
                    gmodel = genai.GenerativeModel(self.model)
                    resp = gmodel.generate_content(
                        prompt,
                        generation_config={"temperature": temperature}
                    )
                    
                    # Track token usage
                    if hasattr(resp, 'usage_metadata'):
                        usage = resp.usage_metadata
                        self.prompt_tokens += getattr(usage, 'prompt_token_count', 0)
                        self.completion_tokens += getattr(usage, 'candidates_token_count', 0)
                        self.total_tokens += getattr(usage, 'total_token_count', 0)
                    
                    raw = getattr(resp, "text", "") or ""
                    data = self._safe_extract_json(raw)
                    
                    # Track successful call
                    self.key_usage[api_key] += 1
                    
                    return data
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    if 'rate limit' in error_msg or 'quota' in error_msg or '429' in error_msg:
                        self.logger.warning(f"Rate limit on key {api_key[:10]}...")
                        self.key_errors[api_key] += 1
                        last_error = e
                        break  # Try next key
                    else:
                        last_error = e
                        if retry < max_retries - 1:
                            self.logger.warning(f"Retry {retry + 1}/{max_retries}: {e}")
                            time.sleep(2 ** retry)
                            continue
                        else:
                            break
        
        raise ValueError(f"All {max_key_attempts} API keys exhausted. Last error: {last_error}")

    def _call_llm(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.5,
        max_retries: int = 3
    ) -> str:
        """Call Gemini API for text generation (non-structured)"""
        
        prompt = system_prompt.strip() + "\n\n" + user_message.strip()
        
        max_key_attempts = len(self.google_api_keys)
        last_error = None
        
        for key_attempt in range(max_key_attempts):
            api_key = self._get_next_api_key()
            genai.configure(api_key=api_key)
            
            for retry in range(max_retries):
                try:
                    gmodel = genai.GenerativeModel(self.model)
                    resp = gmodel.generate_content(
                        prompt,
                        generation_config={"temperature": temperature}
                    )
                    
                    # Track token usage
                    if hasattr(resp, 'usage_metadata'):
                        usage = resp.usage_metadata
                        self.prompt_tokens += getattr(usage, 'prompt_token_count', 0)
                        self.completion_tokens += getattr(usage, 'candidates_token_count', 0)
                        self.total_tokens += getattr(usage, 'total_token_count', 0)
                    
                    raw = getattr(resp, "text", "") or ""
                    
                    # Track successful call
                    self.key_usage[api_key] += 1
                    
                    return raw
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    if 'rate limit' in error_msg or 'quota' in error_msg:
                        self.logger.warning(f"Rate limit on key {api_key[:10]}...")
                        self.key_errors[api_key] += 1
                        last_error = e
                        break  # Try next key
                    else:
                        last_error = e
                        if retry < max_retries - 1:
                            time.sleep(2 ** retry)
                            continue
                        else:
                            break
        
        raise ValueError(f"All API keys exhausted. Last error: {last_error}")

    @staticmethod
    def _safe_extract_json(text: str) -> dict:
        """Extract JSON from response with better error handling"""
        text = text.strip()
        
        # Remove markdown code blocks
        text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
        text = text.strip()
        
        # Try direct parse
        if text.startswith("{") and text.endswith("}"):
            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                logging.warning(f"JSON parse error at line {e.lineno}, col {e.colno}: {e.msg}")
                
                # Try to fix common issues
                fixed_text = text
                
                # Fix trailing commas
                fixed_text = re.sub(r',(\s*[}\]])', r'\1', fixed_text)
                
                # Fix missing commas between } and {
                fixed_text = re.sub(r'}\s*{', r'},{', fixed_text)
                
                # Fix missing commas between ] and {
                fixed_text = re.sub(r']\s*{', r'],{', fixed_text)
                
                # Try parse again
                try:
                    return json.loads(fixed_text)
                except json.JSONDecodeError as e2:
                    # Log context around error
                    lines = fixed_text.split('\n')
                    start = max(0, e2.lineno - 3)
                    end = min(len(lines), e2.lineno + 2)
                    context = '\n'.join(f"{i+1:4d}: {lines[i]}" for i in range(start, end))
                    logging.error(f"JSON error context:\n{context}")
                    raise ValueError(f"Cannot parse JSON: {e2.msg} at line {e2.lineno}")

        # Try to find JSON block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            return json.loads(candidate)

        raise ValueError("No valid JSON found in LLM response")

    def get_token_usage(self) -> dict:
        """Get token usage statistics"""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }

    def get_api_stats(self) -> dict:
        """Get API key usage statistics"""
        return {
            "total_calls": sum(self.key_usage.values()),
            "total_errors": sum(self.key_errors.values()),
            "total_tokens": self.total_tokens,
            "keys": [
                {
                    "key": f"{key[:10]}...{key[-4:]}",
                    "usage": self.key_usage[key],
                    "errors": self.key_errors[key]
                }
                for key in self.google_api_keys
            ]
        }

    def print_api_stats(self):
        """Print API key usage statistics"""
        stats = self.get_api_stats()
        print("\n" + "="*60)
        print("MANIM AGENT - API KEY USAGE")
        print("="*60)
        print(f"Total API Calls: {stats['total_calls']}")
        print(f"Total Errors: {stats['total_errors']}")
        print(f"Total Tokens: {stats['total_tokens']:,}")
        print("-"*60)
        for i, key_stat in enumerate(stats['keys'], 1):
            print(f"Key {i} ({key_stat['key']}): {key_stat['usage']} calls, {key_stat['errors']} errors")
        print("="*60 + "\n")

    # ===== Rest of methods (giữ nguyên logic) =====
    
    def _save_scene_plans(self, scene_plans: List[ScenePlan], concept_analysis: ConceptAnalysis, response_json: Dict[str, Any]) -> Path:
        """Save raw scene plans output to JSON file for debugging"""
        safe_name = "".join(c if c.isalnum() else "_" for c in concept_analysis.main_concept.lower())
        safe_name = safe_name[:50]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name}_raw_scene_plans_{timestamp}.json"
        filepath = self.output_dir / "scene_plans" / filename

        with open(filepath, 'w') as f:
            json.dump(response_json, f, indent=2)

        self.logger.info(f"Raw scene plans output saved to {filepath}")
        return filepath

    def _generate_scene_codes(self, scene_plans: List[ScenePlan]) -> List[ManimSceneCode]:
        """Generate Manim code for each scene plan in parallel"""
        scene_codes = []
        
        if not scene_plans:
            self.logger.error("No scene plans provided for code generation")
            return []
        
        self.logger.info(f"Starting parallel code generation for {len(scene_plans)} scenes")

        def generate_single_scene_code(scene_plan: ScenePlan) -> Optional[ManimSceneCode]:
            try:
                self.logger.info(f"Generating code for scene: {scene_plan.title}")
                class_name = self._sanitize_class_name(scene_plan.id)
                scene_plan_json = json.dumps(scene_plan.model_dump(), indent=2)

                formatted_prompt = self.CODE_GENERATION_PROMPT.format(
                    scene_plan=scene_plan_json,
                    class_name=class_name,
                    target_duration="25-30"
                )

                response = self._call_llm(
                    system_prompt=formatted_prompt,
                    user_message="Generate the Manim code for the scene plan specified above.",
                    temperature=self.config.temperature,
                    max_retries=3
                )

                manim_code, extraction_method = self._extract_manim_code(response)
                
                # Ensure class name matches
                manim_code = re.sub(
                    r'class\s+\w+\s*\(\s*Scene\s*\)\s*:',
                    f'class {class_name}(Scene):',
                    manim_code,
                    count=1
                )

                if not re.search(rf'class\s+{class_name}\s*\(\s*Scene\s*\)\s*:', manim_code):
                    self.logger.error(f"Class name mismatch after normalize: expected {class_name}")
                    return None

                if manim_code:
                    self._save_scene_code(scene_plan.id, class_name, manim_code, response)
                    scene_code = ManimSceneCode(
                        scene_id=scene_plan.id,
                        scene_name=class_name,
                        manim_code=manim_code,
                        raw_llm_output=response,
                        extraction_method=extraction_method
                    )
                    self.logger.info(f"Successfully generated code for scene: {class_name}")
                    return scene_code
                else:
                    self.logger.error(f"Failed to extract Manim code from response")
                    return None

            except Exception as e:
                self.logger.error(f"Code generation failed for scene {scene_plan.id}: {e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                return None

        max_workers = max(1, min(len(scene_plans), 10))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_plan = {executor.submit(generate_single_scene_code, plan): plan for plan in scene_plans}
            for future in as_completed(future_to_plan):
                scene_plan = future_to_plan[future]
                try:
                    result = future.result()
                    if result:
                        scene_codes.append(result)
                except Exception as e:
                    self.logger.error(f"Exception in parallel code generation for {scene_plan.id}: {e}")

        scene_codes.sort(key=lambda x: scene_plans.index(next(p for p in scene_plans if p.id == x.scene_id)))
        self.logger.info(f"Parallel code generation complete: {len(scene_codes)}/{len(scene_plans)} succeeded")
        return scene_codes

    def _extract_manim_code(self, response: str) -> tuple[str, str]:
        """Extract Manim code from LLM response"""
        manim_pattern = r'<manim>(.*?)</manim>'
        matches = re.findall(manim_pattern, response, re.DOTALL)

        if matches:
            manim_code = matches[0].strip()
            manim_code = self._clean_manim_code(manim_code)
            return manim_code, "tags"

        class_pattern = r'class\s+(\w+)\s*\(\s*Scene\s*\):.*?(?=\n\nclass|\Z)'
        matches = re.findall(class_pattern, response, re.DOTALL)

        if matches:
            class_start = response.find(f"class {matches[0]}(")
            if class_start != -1:
                remaining = response[class_start:]
                next_class = re.search(r'\n\nclass\s+\w+', remaining)
                if next_class:
                    manim_code = remaining[:next_class.start()]
                else:
                    manim_code = remaining

                if "from manim import" not in manim_code:
                    manim_code = "from manim import *\n\n" + manim_code

                manim_code = self._clean_manim_code(manim_code)
                return manim_code.strip(), "parsing"

        if "class" in response and "def construct" in response:
            cleaned = response.strip()
            if not cleaned.startswith("from"):
                cleaned = "from manim import *\n\n" + cleaned
            cleaned = self._clean_manim_code(cleaned)
            return cleaned, "cleanup"

        return "", "failed"

    def _clean_manim_code(self, code: str) -> str:
        """Clean Manim code by removing backticks"""
        code = code.replace('`', '')
        code = re.sub(r'python\n', '', code, flags=re.IGNORECASE)
        code = re.sub(r'\npython', '', code, flags=re.IGNORECASE)
        code = re.sub(r'^```.*\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n```.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n{3,}', '\n\n', code)
        return code.strip()

    def _sanitize_class_name(self, scene_id: str) -> str:
        """Convert scene ID to valid Python class name"""
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '', scene_id)
        if sanitized and sanitized[0].isdigit():
            sanitized = "Scene_" + sanitized
        sanitized = sanitized.title().replace('_', '')
        if not sanitized:
            sanitized = "AnimationScene"
        return sanitized

    def _save_scene_code(self, scene_id: str, class_name: str, manim_code: str, raw_output: str) -> Path:
        """Save generated Manim code to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{scene_id}_{class_name}_{timestamp}.py"
        filepath = self.output_dir / "scene_codes" / filename

        # ✅ Use UTF-8 encoding to handle Unicode characters
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Generated Manim code for scene: {scene_id}\n")
            f.write(f"# Class: {class_name}\n")
            f.write(f"# Generated at: {timestamp}\n")
            f.write(f"# -*- coding: utf-8 -*-\n\n")
            f.write(manim_code)

        # Also save raw LLM output with UTF-8
        raw_filepath = filepath.with_suffix('.raw.txt')
        with open(raw_filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Raw LLM output for scene: {scene_id}\n")
            f.write(f"# Class: {class_name}\n")
            f.write(f"# Generated at: {timestamp}\n\n")
            f.write(raw_output)

        return filepath

    def _render_scenes(self, scene_codes: List[ManimSceneCode]) -> List[RenderResult]:
        """Render each scene using ManimRenderer"""
        render_results = []

        for scene_code in scene_codes:
            self.logger.info(f"Rendering scene: {scene_code.scene_name}")
            output_filename = f"{scene_code.scene_id}_{scene_code.scene_name}.mp4"

            try:
                render_result = self.renderer.render(
                    manim_code=scene_code.manim_code,
                    scene_name=scene_code.scene_name,
                    output_filename=output_filename
                )

                result = RenderResult(
                    scene_id=scene_code.scene_id,
                    success=render_result.success,
                    video_path=render_result.video_path,
                    error_message=render_result.error_message,
                    duration=render_result.duration,
                    resolution=render_result.resolution,
                    render_time=render_result.render_time
                )

                render_results.append(result)

                if result.success:
                    self.logger.info(f"Successfully rendered: {scene_code.scene_name}")
                    self.logger.info(f"  Video path: {result.video_path}")
                    self.logger.info(f"  Duration: {result.duration}s")
                else:
                    self.logger.error(f"Failed to render {scene_code.scene_name}: {result.error_message}")

            except Exception as e:
                self.logger.error(f"Rendering failed for {scene_code.scene_name}: {e}")
                render_results.append(RenderResult(
                    scene_id=scene_code.scene_id,
                    success=False,
                    error_message=str(e)
                ))

        return render_results

    def _concatenate_scenes(self, render_results: List[RenderResult]) -> Optional[Path]:
        """Concatenate rendered scenes into single silent animation"""
        if not render_results:
            self.logger.error("No render results to concatenate")
            return None

        video_paths = []
        for r in render_results:
            if r.success and r.video_path:
                video_path = Path(r.video_path)
                if not video_path.is_absolute():
                    video_path = (Path.cwd() / video_path).resolve()
                if video_path.exists():
                    video_paths.append(video_path)
                else:
                    self.logger.warning(f"Video path does not exist: {video_path}")

        if not video_paths:
            self.logger.error("No successfully rendered scenes with valid video paths to concatenate")
            self.logger.error(f"Render results: {[(r.scene_id, r.success, r.video_path) for r in render_results]}")
            return None

        self.logger.info(f"Found {len(video_paths)} videos to concatenate")

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"animation_{timestamp}.mp4"
            output_path = self.output_dir / "animations" / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"Concatenating {len(video_paths)} scenes into {output_filename}")

            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                for video_path in video_paths:
                    abs_path = str(video_path.resolve())
                    temp_file.write(f"file '{abs_path}'\n")
                    self.logger.debug(f"Adding to concat list: {abs_path}")
                temp_file_path = temp_file.name

            try:
                cmd = [
                    "ffmpeg",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", str(temp_file_path),
                    "-c", "copy",
                    "-y",
                    str(output_path.resolve())
                ]

                self.logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120
                )

                if result.returncode == 0 and output_path.exists():
                    self.logger.info(f"Successfully concatenated animation: {output_filename}")
                    self.logger.info(f"Final video path: {output_path}")
                    return output_path
                else:
                    self.logger.error(f"FFmpeg concatenation failed with return code {result.returncode}")
                    self.logger.error(f"STDERR: {result.stderr}")
                    self.logger.error(f"STDOUT: {result.stdout}")
                    self.logger.error(f"Output path exists: {output_path.exists()}")
                    return None

            finally:
                try:
                    import os
                    os.unlink(temp_file_path)
                except OSError:
                    pass

        except Exception as e:
            self.logger.error(f"Scene concatenation failed: {e}")
            return None

    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about animation generation performance"""
        return {
            "token_usage": self.get_token_usage(),
            "api_stats": self.get_api_stats(),
            "model_used": self.model,
            "config": self.config.model_dump(),
            "renderer_status": "ready" if self.renderer.validate_manim_installation() else "not_ready"
        }