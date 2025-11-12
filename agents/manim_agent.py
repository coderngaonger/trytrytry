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

import google.generativeai as genai  # << Gemini SDK

from agents.base import BaseAgent
from agents.concept_interpreter import ConceptAnalysis, SubConcept
from agents.manim_models import (
    ScenePlan, SceneAction, ManimSceneCode, RenderResult,
    AnimationResult, AnimationConfig, AnimationMetadata
)
from rendering.manim_renderer import ManimRenderer


class ManimAgent(BaseAgent):
    """
    Manim Agent (Gemini-only): Transforms structured concept analysis into visual animations
    using scene planning and Manim code generation with <manim> tag extraction.
    """

    def __init__(
        self,
        google_api_keys: str,
        google_api_endpoint: Optional[str],
        model: str,
        output_dir: Path,
        config: Optional[AnimationConfig] = None,
        reasoning_tokens: Optional[float] = None,  # không dùng cho Gemini, giữ để không vỡ interface
        reasoning_effort: Optional[str] = None,    # không dùng
        **kwargs,                                   # nuốt các tham số cũ như api_key/base_url nếu còn sót
    ):
        # Khởi tạo các thuộc tính token tracking
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        
        # Lưu các tham số
        self.google_api_keys = google_api_keys
        self.google_api_endpoint = google_api_endpoint
        self.model = model
        self.output_dir = output_dir
        self.config = config or AnimationConfig()

        try:
            super().__init__(model=model)
        except TypeError:
            # Nếu BaseAgent có signature khác thì vẫn an toàn bỏ qua
            pass

        # Cấu hình Gemini
        genai.configure(api_key=google_api_keys)

        self.model = model
        self.output_dir = Path(output_dir)
        self.config = config or AnimationConfig()

        if not hasattr(self, "logger"):
            self.logger = logging.getLogger(self.__class__.__name__)

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
   - Prefer easing that reads smoothly (ease-in-out). Include `"parameters": {"easing": "ease_in_out"}` when relevant.

6. **Educational Flow**:
   - Start with context/overview
   - Introduce new elements step-by-step
   - Show relationships and connections visually
   - End with key takeaways or summaries, keeping the same example visible to reinforce learning

7. **Element Naming**:
   - Use descriptive, stable targets (e.g., "bayes_equation", "likelihood_label", "frequency_grid") reused across scenes.
   - When transforming, specify `"parameters": {"from": "<old_target>", "to": "<new_target>"}` where helpful.

**OUTPUT FORMAT**:
Return ONLY valid JSON matching this exact structure:
{{
    "scene_plans": [
        {{
            "id": "string",
            "title": "string",
            "description": "string",
            "sub_concept_id": "string",
            "actions": [
                {{
                    "action_type": "string",
                    "element_type": "string",
                    "description": "string",
                    "target": "string",
                    "duration": number,
                    "parameters": {{}}
                }}
            ],
            "scene_dependencies": ["string"]
        }}
    ]
}}
...
Generate scene plans that will create clear, educational animations for the given concept, using a single consistent example across scenes, with slow pacing and explicit narration pauses after each major action."""
    # (Giữ nguyên toàn bộ nội dung prompt như bạn gửi)

    CODE_GENERATION_PROMPT = """You are a Manim Code Generation Agent for creating **very simple, 2D educational STEM animations**.
    - Never call find_highlighted_text; use Indicate(obj) or SurroundingRectangle(obj) instead.
...
</manim>
"""
    # (Giữ nguyên toàn bộ nội dung prompt như bạn gửi)

    # ====== OVERRIDES: dùng Gemini thay vì BaseAgent OpenRouter ======
    def _call_llm_structured(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.5,
        max_retries: int = 3,
    ) -> dict:
        """
        Gọi Gemini và bắt buộc trả về JSON; tự trích JSON trong trường hợp model có dính text rìa.
        """
        prompt = (
            system_prompt.strip()
            + "\n\nSTRICT OUTPUT: Return ONLY valid JSON per the schema. No prose, no backticks."
            + "\n\nUSER REQUEST:\n"
            + user_message.strip()
        )
        last_err = None
        for _ in range(max_retries):
            try:
                gmodel = genai.GenerativeModel(self.model)
                resp = gmodel.generate_content(
                    prompt,
                    generation_config={"temperature": temperature}
                )
                raw = getattr(resp, "text", "") or ""
                data = self._safe_extract_json(raw)
                return data
            except Exception as e:
                last_err = e
        raise ValueError(f"LLM returned invalid structured output: {last_err}")

    def _call_llm(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.5,
        max_retries: int = 3,
    ) -> str:
        """
        Gọi Gemini cho text/code (non-structured). Trả về string (resp.text).
        """
        prompt = system_prompt.strip() + "\n\n" + user_message.strip()
        last_err = None
        for _ in range(max_retries):
            try:
                gmodel = genai.GenerativeModel(self.model)
                resp = gmodel.generate_content(
                    prompt,
                    generation_config={"temperature": temperature}
                )
                text = getattr(resp, "text", "") or ""
                if text.strip():
                    return text
                last_err = "Empty response"
            except Exception as e:
                last_err = e
        raise ValueError(f"LLM call failed: {last_err}")

    @staticmethod
    def _safe_extract_json(text: str) -> dict:
        """
        Trích JSON từ chuỗi có thể lẫn text rìa.
        """
        s = (text or "").strip()
        if s.startswith("{") and s.endswith("}"):
            return json.loads(s)
        a = s.find("{")
        b = s.rfind("}")
        if a != -1 and b != -1 and b > a:
            return json.loads(s[a:b+1])
        raise ValueError("No valid JSON found")

    # ====== Phần dưới: GIỮ NGUYÊN (logic pipeline của bạn) ======
    def execute(self, concept_analysis: ConceptAnalysis) -> AnimationResult:
        start_time = time.time()
        self.logger.info(f"Starting animation generation for: {concept_analysis.main_concept}")
        try:
            # Step 1: Scene plans
            scene_plans, response_json = self._generate_scene_plans(concept_analysis)
            self.logger.info(f"Generated {len(scene_plans)} scene plans")
            self._save_scene_plans(scene_plans, concept_analysis, response_json)

            # Step 2: Manim code
            scene_codes = self._generate_scene_codes(scene_plans)
            self.logger.info(f"Generated code for {len(scene_codes)} scenes")

            # Step 3: Render
            render_results = self._render_scenes(scene_codes)
            successful_renders = [r for r in render_results if r.success]
            self.logger.info(f"Successfully rendered {len(successful_renders)}/{len(render_results)} scenes")

            # Step 4: Concat
            if successful_renders:
                silent_animation_path = self._concatenate_scenes(successful_renders)
            else:
                silent_animation_path = None

            generation_time = time.time() - start_time
            total_render_time = sum(r.render_time or 0 for r in render_results)

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
        user_message = f"Analyze this STEM concept and create scene plans:\n\n{json.dumps(concept_analysis.model_dump(), indent=2)}"
        try:
            response_json = self._call_llm_structured(
                system_prompt=self.SCENE_PLANNING_PROMPT,
                user_message=user_message,
                temperature=self.config.temperature,
                max_retries=3
            )
            scene_plans = []
            for plan_data in response_json.get("scene_plans", []):
                try:
                    scene_plan = ScenePlan(**plan_data)
                    scene_plans.append(scene_plan)
                except Exception as e:
                    self.logger.warning(f"Invalid scene plan data: {e}")
                    continue
            return scene_plans, response_json
        except Exception as e:
            self.logger.error(f"Scene planning failed: {e}")
            raise ValueError(f"Failed to generate scene plans: {e}")

    def _save_scene_plans(self, scene_plans: List[ScenePlan], concept_analysis: ConceptAnalysis, response_json: Dict[str, Any]) -> Path:
        safe_name = "".join(c if c.isalnum() else "_" for c in concept_analysis.main_concept.lower())[:50]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name}_raw_scene_plans_{timestamp}.json"
        filepath = self.output_dir / "scene_plans" / filename
        with open(filepath, 'w') as f:
            json.dump(response_json, f, indent=2)
        self.logger.info(f"Raw scene plans output saved to {filepath}")
        return filepath

    def _generate_scene_codes(self, scene_plans: List[ScenePlan]) -> List[ManimSceneCode]:
        scene_codes = []
        
        # ✅ FIX: Kiểm tra trước khi tạo ThreadPoolExecutor
        if len(scene_plans) == 0:
            self.logger.error("No scene plans provided for code generation")
            return []
        
        self.logger.info(f"Starting parallel code generation for {len(scene_plans)} scenes")

        def generate_single_scene_code(scene_plan: ScenePlan) -> Optional[ManimSceneCode]:
            try:
                self.logger.info(f"Generating code for scene: {scene_plan.title}")
                self.logger.debug(f"Scene ID: {scene_plan.id}, Actions count: {len(scene_plan.actions)}")
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
                # Ensure class name matches exactly what the renderer will run
                manim_code = re.sub(
                    r'class\s+\w+\s*\(\s*Scene\s*\)\s*:',
                    f'class {class_name}(Scene):',
                    manim_code,
                    count=1
                )

                # Double-check presence
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
                    self.logger.error(f"Failed to extract Manim code from response for scene: {scene_plan.id}")
                    self.logger.error(f"Response contained: {response[:500]}...")
                    return None

            except Exception as e:
                self.logger.error(f"Code generation failed for scene {scene_plan.id}: {e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                return None

        # ✅ FIX: Đảm bảo max_workers >= 1
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
        code = code.replace('`', '')
        code = re.sub(r'python\n', '', code, flags=re.IGNORECASE)
        code = re.sub(r'\npython', '', code, flags=re.IGNORECASE)
        code = re.sub(r'^```.*\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n```.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n{3,}', '\n\n', code)
        code = code.strip()
        code = re.sub(r'\.find_highlighted_text\s*\(.*?\)\s*', '', code)
        return code

    def _sanitize_class_name(self, scene_id: str) -> str:
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '', scene_id)
        if sanitized and sanitized[0].isdigit():
            sanitized = "Scene_" + sanitized
        sanitized = sanitized.title().replace('_', '')
        if not sanitized:
            sanitized = "AnimationScene"
        return sanitized

    def _save_scene_code(self, scene_id: str, class_name: str, manim_code: str, raw_output: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{scene_id}_{class_name}_{timestamp}.py"
        filepath = self.output_dir / "scene_codes" / filename
        with open(filepath, 'w') as f:
            f.write(f"# Generated Manim code for scene: {scene_id}\n")
            f.write(f"# Class: {class_name}\n")
            f.write(f"# Generated at: {timestamp}\n\n")
            f.write(manim_code)
        raw_filepath = filepath.with_suffix('.raw.txt')
        with open(raw_filepath, 'w') as f:
            f.write(f"# Raw LLM output for scene: {scene_id}\n")
            f.write(f"# Class: {class_name}\n")
            f.write(f"# Generated at: {timestamp}\n\n")
            f.write(raw_output)
        return filepath

    def _render_scenes(self, scene_codes: List[ManimSceneCode]) -> List[RenderResult]:
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
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"animation_{timestamp}.mp4"
            output_path = self.output_dir / "animations" / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                for video_path in video_paths:
                    abs_path = str(video_path.resolve())
                    temp_file.write(f"file '{abs_path}'\n")
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
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.returncode == 0 and output_path.exists():
                    self.logger.info(f"Successfully concatenated animation: {output_filename}")
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
        return {
            "token_usage": self.get_token_usage(),
            "model_used": self.model,
            "reasoning_tokens": getattr(self, "reasoning_tokens", None),
            "config": self.config.model_dump(),
            "renderer_status": "ready" if self.renderer.validate_manim_installation() else "not_ready"
        }
