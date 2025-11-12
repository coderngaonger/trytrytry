import subprocess
import tempfile
import logging
import time
import re
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import json


@dataclass
class RenderResult:
    """Result of a Manim render operation"""
    success: bool
    video_path: Optional[str] = None
    error_message: Optional[str] = None
    duration: Optional[float] = None
    resolution: Optional[Tuple[int, int]] = None
    render_time: Optional[float] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None


class ManimRenderer:
    """
    Safe Manim code execution via subprocess with comprehensive error handling
    """

    def __init__(
        self,
        output_dir: Path,
        quality: str = "1080p60",
        background_color: str = "#0f0f0f",
        timeout: int = 300,  # 5 minutes default timeout
        max_retries: int = 2
    ):
        self.output_dir = Path(output_dir)
        self.quality = quality
        self.background_color = background_color
        self.timeout = timeout
        self.max_retries = max_retries
        self.logger = logging.getLogger(self.__class__.__name__)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Quality mapping to Manim parameters (new format for v0.19+)
        self.quality_map = {
            "l": {"resolution": "854x480", "frame_rate": 15, "name": "low"},
            "m": {"resolution": "1280x720", "frame_rate": 30, "name": "medium"},
            "h": {"resolution": "1920x1080", "frame_rate": 30, "name": "high"},
            "p": {"resolution": "1920x1080", "frame_rate": 60, "name": "production"},
            "k": {"resolution": "2560x1440", "frame_rate": 60, "name": "4k"},
        }

        # Map old quality names to new format
        self.old_to_new_quality = {
            "480p15": "l",
            "720p30": "m",
            "1080p30": "h",
            "1080p60": "p",
            "1440p60": "k",
            "4k60": "k"
        }

    def render(
        self,
        manim_code: str,
        scene_name: str,
        output_filename: Optional[str] = None
    ) -> RenderResult:
        """
        Render Manim code and return video file path

        Args:
            manim_code: Complete Manim Python code as string
            scene_name: Name of the scene class to render
            output_filename: Optional custom filename for output video

        Returns:
            RenderResult with success status and video path or error details
        """

        start_time = time.time()

        for attempt in range(self.max_retries + 1):
            try:
                self.logger.info(f"Rendering Manim scene '{scene_name}' (attempt {attempt + 1}/{self.max_retries + 1})")

                # Create temporary file for Manim code
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                    temp_file.write(manim_code)
                    temp_file_path = temp_file.name

                try:
                    # Prepare Manim command
                    cmd = self._build_manim_command(temp_file_path, scene_name, output_filename)

                    # Execute Manim (run from project root, not output_dir)
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout,
                        cwd=Path.cwd()
                    )

                    # Parse output and find video file
                    if result.returncode == 0:
                        video_path = self._find_output_video(result.stdout, scene_name, output_filename)
                        if video_path:
                            # Extract metadata
                            duration = self._extract_video_duration(video_path)
                            resolution = self._get_quality_resolution()
                            render_time = time.time() - start_time

                            self.logger.info(f"Successfully rendered {scene_name} in {render_time:.2f}s")

                            return RenderResult(
                                success=True,
                                video_path=str(video_path),
                                duration=duration,
                                resolution=resolution,
                                render_time=render_time,
                                stdout=result.stdout,
                                stderr=result.stderr
                            )
                        else:
                            error_msg = "Manim completed but no output video found"
                            self.logger.error(error_msg)
                            if attempt < self.max_retries:
                                continue
                            return RenderResult(
                                success=False,
                                error_message=error_msg,
                                stdout=result.stdout,
                                stderr=result.stderr
                            )
                    else:
                        # Manim failed, analyze error
                        error_analysis = self._analyze_manim_error(result.stderr, result.stdout)
                        self.logger.warning(f"Manim render failed: {error_analysis}")
                        self.logger.warning(f"STDERR: {result.stderr[-500:]}")
                        self.logger.warning(f"STDOUT: {result.stdout[-500:]}")

                        if attempt < self.max_retries and self._is_retryable_error(error_analysis):
                            # Try with simplified code on retry
                            simplified_code = self._simplify_manim_code(manim_code)
                            if simplified_code != manim_code:
                                manim_code = simplified_code
                                continue

                        return RenderResult(
                            success=False,
                            error_message=error_analysis,
                            stdout=result.stdout,
                            stderr=result.stderr
                        )

                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_file_path)
                    except OSError:
                        pass

            except subprocess.TimeoutExpired:
                error_msg = f"Manim render timed out after {self.timeout} seconds"
                self.logger.error(error_msg)
                if attempt < self.max_retries:
                    continue
                return RenderResult(success=False, error_message=error_msg)

            except Exception as e:
                error_msg = f"Unexpected error during Manim render: {str(e)}"
                self.logger.error(error_msg)
                if attempt < self.max_retries:
                    continue
                return RenderResult(success=False, error_message=error_msg)

        # All retries exhausted
        return RenderResult(
            success=False,
            error_message=f"Failed after {self.max_retries + 1} attempts"
        )

    def _build_manim_command(
        self,
        script_path: str,
        scene_name: str,
        output_filename: Optional[str] = None
    ) -> list:
        """Build Manim command line arguments"""

        # Convert quality to new format if needed
        quality = self.old_to_new_quality.get(self.quality, self.quality)
        
        cmd = [
            "manim", "render",
            script_path,
            scene_name,
            f"-q{quality}"
        ]

        # Add custom output filename if specified
        if output_filename:
            cmd.extend(["-o", str(output_filename)])

        return cmd

    def _find_output_video(
        self,
        manim_output: str,
        scene_name: str,
        output_filename: Optional[str] = None
    ) -> Optional[Path]:
        """Find the rendered video file in Manim output"""

        # First try custom filename if specified
        if output_filename:
            custom_path = self.output_dir / output_filename
            if custom_path.exists():
                return custom_path

        # Parse Manim output for file path
        # Look for patterns like "File ready at /path/to/video.mp4"
        patterns = [
            r"File ready at (.+?\.mp4)",
            r"Output written to (.+?\.mp4)",
            r"Saved animation to (.+?\.mp4)"
        ]

        for pattern in patterns:
            matches = re.findall(pattern, manim_output)
            if matches:
                video_path = Path(matches[-1].strip())  # Use last match
                if video_path.exists():
                    return video_path

        # New Manim (v0.19+) stores files in media/videos directory
        # Search in the default media directory first (relative to cwd)
        media_dir = Path.cwd() / "media" / "videos"
        if media_dir.exists():
            # Look for the most recent video file (exclude partial_movie_files)
            video_files = [v for v in media_dir.rglob("*.mp4") if "partial_movie_files" not in str(v)]
            if video_files:
                # Filter by scene name if possible
                scene_videos = [v for v in video_files if scene_name in v.name]
                if scene_videos:
                    video_path = max(scene_videos, key=lambda x: x.stat().st_mtime)
                else:
                    video_path = max(video_files, key=lambda x: x.stat().st_mtime)

                # Copy to our output directory with proper naming
                if video_path.exists():
                    if output_filename:
                        output_path = self.output_dir / output_filename
                    else:
                        output_path = self.output_dir / f"{scene_name}.mp4"
                    import shutil
                    shutil.copy2(video_path, output_path)
                    self.logger.info(f"Copied video from {video_path} to {output_path}")
                    return output_path

        # Fallback: search for video files with scene name in output directory
        search_patterns = [
            f"{scene_name}.mp4",
            f"*{scene_name}*.mp4"
        ]

        for pattern in search_patterns:
            videos = [v for v in self.output_dir.glob(pattern) if v.is_file()]
            if videos:
                # Return the most recently modified file
                video_path = max(videos, key=lambda x: x.stat().st_mtime)
                self.logger.info(f"Found video via fallback: {video_path}")
                return video_path

        self.logger.error(f"Could not find output video for scene: {scene_name}")
        self.logger.error(f"Searched in: {self.output_dir}")
        self.logger.error(f"Available files: {list(self.output_dir.glob('*.mp4'))}")
        return None

    def _analyze_manim_error(self, stderr: str, stdout: str) -> str:
        """Analyze Manim error output and provide helpful error messages"""

        output = stderr + stdout

        # Common error patterns
        error_patterns = {
            r"No such file or directory: 'latex'": "LaTeX not installed - install LaTeX (brew install --cask mactex)",
            r"SyntaxError": "Syntax error in Manim code - invalid Python syntax",
            r"IndentationError": "Indentation error - check code spacing and alignment",
            r"NameError.*not defined": "Variable or function not defined - check for typos",
            r"AttributeError.*has no attribute": "Attribute error - method or property doesn't exist",
            r"TypeError": "Type error - incorrect data types being used",
            r"ValueError": "Value error - invalid argument values",
            r"ZeroDivisionError": "Division by zero error in mathematical calculations",
            r"IndexError": "Index error - list or array index out of range",
            r"KeyError": "Key error - dictionary key not found",
            r"ImportError": "Import error - required module not available",
            r"ModuleNotFoundError": "Module not found - check import statements",
            r"No scene named": f"Scene class not found - ensure scene name matches class definition",
            r"OpenGL is not supported": "OpenGL rendering not supported - falling back to software rendering",
            r"ffmpeg.*not found": "FFmpeg not found - ensure FFmpeg is installed and in PATH",
            r"Permission denied": "Permission error - unable to write to output directory",
            r"Disk full": "Disk full - insufficient storage space for rendering",
            r"Memory.*error": "Memory error - insufficient RAM for complex animation",
            r"Timeout": "Rendering timeout - animation too complex or taking too long"
        }

        for pattern, message in error_patterns.items():
            if re.search(pattern, output, re.IGNORECASE):
                return message

        # If no specific pattern matches, return a generic message with useful context
        if "Traceback" in output:
            # Extract the last line of the traceback for more specific error
            lines = output.strip().split('\n')
            for line in reversed(lines):
                if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                    return f"Manim render failed: {line.strip()}"

        return f"Manim render failed with unknown error. Check logs for details."

    def _is_retryable_error(self, error_message: str) -> bool:
        """Determine if an error is likely to be fixed by retrying with simplified code"""

        retryable_patterns = [
            r"Memory.*error",
            r"Timeout",
            r"OpenGL.*not supported",
            r"Complex.*animation",
            r"Too.*many.*objects"
        ]

        for pattern in retryable_patterns:
            if re.search(pattern, error_message, re.IGNORECASE):
                return True

        return False

    def _simplify_manim_code(self, code: str) -> str:
        """Attempt to simplify Manim code to avoid common rendering issues"""

        # Basic simplifications that might help with complex scenes
        simplifications = [
            # Reduce animation quality settings within code
            (r"frame_rate\s*=\s*\d+", "frame_rate = 15"),

            # Reduce particle counts or complex object counts
            (r"n_points\s*=\s*\d+", "n_points = 20"),
            (r"num_elements\s*=\s*\d+", "num_elements = 10"),

            # Simplify colors to basic ones
            (r"#[0-9a-fA-F]{6}", "#FFFFFF"),

            # Reduce animation durations
            (r"run_time\s*=\s*\d+\.?\d*", "run_time = 1.0"),
        ]

        simplified_code = code
        for pattern, replacement in simplifications:
            simplified_code = re.sub(pattern, replacement, simplified_code)

        return simplified_code

    def _extract_video_duration(self, video_path: Path) -> Optional[float]:
        """Extract video duration using FFmpeg"""
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                duration_str = result.stdout.strip()
                if duration_str:
                    return float(duration_str)

        except (subprocess.TimeoutExpired, ValueError, subprocess.SubprocessError):
            pass

        return None

    def _get_quality_resolution(self) -> Optional[Tuple[int, int]]:
        """Get resolution tuple for current quality setting"""
        quality_info = self.quality_map.get(self.quality)
        if quality_info and "resolution" in quality_info:
            width, height = map(int, quality_info["resolution"].split('x'))
            return (width, height)
        return None

    def get_supported_qualities(self) -> list:
        """Return list of supported quality presets"""
        return list(self.quality_map.keys())

    def validate_manim_installation(self) -> bool:
        """Check if Manim is properly installed and accessible"""
        try:
            result = subprocess.run(
                ["manim", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False