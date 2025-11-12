import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from .base import BaseTTSSynthesizer, AudioSegment, AudioResult

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class OpenAITTSSynthesizer(BaseTTSSynthesizer):
    """OpenAI TTS provider implementation"""

    def __init__(
        self,
        api_key: str,
        output_dir: Path,
        voice: str = "alloy",
        model: str = "tts-1",
        response_format: str = "mp3",
        speed: float = 1.0,
        base_url: Optional[str] = None,
        **kwargs
    ):
        super().__init__(api_key, output_dir, **kwargs)

        # OpenAI-specific settings
        self.voice = voice
        self.model = model
        self.response_format = response_format
        self.speed = speed
        self.base_url = base_url

        # Initialize OpenAI client
        if OpenAI is None:
            raise ImportError("OpenAI library is not installed. Install with: pip install openai")
        
        # Use custom endpoint if provided, otherwise use default
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
            self.logger.info(f"Using custom OpenAI endpoint: {base_url}")
        
        self.client = OpenAI(**client_kwargs)

    def execute(self, script_path: str, target_duration: Optional[float] = None) -> AudioResult:
        """
        Convert SRT script to synchronized audio using OpenAI

        Args:
            script_path: Path to SRT script file
            target_duration: Optional target duration to match video

        Returns:
            AudioResult with generated audio file and metadata
        """
        start_time = time.time()
        self.logger.info(f"Starting OpenAI audio synthesis for script: {script_path}")

        try:
            # Validate script file
            script_file = Path(script_path)
            if not script_file.exists():
                raise FileNotFoundError(f"Script file not found: {script_path}")

            # Parse SRT file
            subtitles = self._parse_srt_file(script_file)
            if not subtitles:
                raise ValueError("No subtitles found in SRT file")

            self.logger.info(f"Parsed {len(subtitles)} subtitles from script")

            # Generate audio for each subtitle
            audio_segments = self._generate_audio_segments(subtitles)

            if not audio_segments:
                raise ValueError("Failed to generate audio for any subtitles")

            # Concatenate audio segments
            final_audio_path = self._concatenate_audio_segments(audio_segments, script_file.stem)

            # Validate audio duration
            actual_duration = self._get_audio_duration(final_audio_path)
            if target_duration and actual_duration:
                # Add silence padding if needed
                if actual_duration < target_duration:
                    final_audio_path = self._add_silence_padding(
                        final_audio_path, target_duration - actual_duration
                    )
                    actual_duration = target_duration

            # Calculate file size
            file_size_mb = final_audio_path.stat().st_size / (1024 * 1024)

            generation_time = time.time() - start_time
            self.logger.info(f"OpenAI audio synthesis completed in {generation_time:.2f}s")
            self.logger.info(f"Generated audio: {actual_duration:.2f}s, {file_size_mb:.2f}MB")

            return AudioResult(
                success=True,
                audio_path=str(final_audio_path),
                audio_segments=audio_segments,
                total_duration=actual_duration,
                file_size_mb=file_size_mb,
                generation_time=generation_time,
                model_used=self.model,
                voice_settings={
                    "voice": self.voice,
                    "model": self.model,
                    "response_format": self.response_format,
                    "speed": self.speed
                }
            )

        except Exception as e:
            self.logger.error(f"OpenAI audio synthesis failed: {e}")
            return AudioResult(
                success=False,
                error_message=str(e),
                generation_time=time.time() - start_time,
                model_used=self.model
            )

    def _generate_audio_segments(self, subtitles: List[Dict[str, Any]]) -> List[AudioSegment]:
        """Generate audio for each subtitle using OpenAI"""

        audio_segments = []

        for subtitle in subtitles:
            self.logger.info(f"Generating OpenAI audio for subtitle {subtitle['sequence']}: {subtitle['text'][:50]}...")

            for attempt in range(self.max_retries):
                try:
                    # Generate audio using OpenAI TTS API
                    response = self.client.audio.speech.create(
                        model=self.model,
                        voice=self.voice,
                        input=subtitle['text'],
                        response_format=self.response_format,
                        speed=self.speed
                    )

                    # Save audio segment to temporary file
                    segment_path = self._save_audio_segment(response, subtitle['sequence'])

                    # Get actual duration
                    actual_duration = self._get_audio_duration(segment_path)

                    audio_segment = AudioSegment(
                        text=subtitle['text'],
                        start_time=subtitle['start_time'],
                        end_time=subtitle['end_time'],
                        audio_path=str(segment_path),
                        duration=actual_duration,
                        file_size=segment_path.stat().st_size / (1024 * 1024)
                    )

                    audio_segments.append(audio_segment)
                    self.logger.info(f"Generated OpenAI audio segment: {actual_duration:.2f}s")
                    break

                except Exception as e:
                    self.logger.warning(f"OpenAI TTS failed for subtitle {subtitle['sequence']} (attempt {attempt + 1}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        self.logger.error(f"Failed to generate OpenAI audio for subtitle {subtitle['sequence']}")
                        break

        return audio_segments

    def _save_audio_segment(self, response, sequence: int) -> Path:
        """Save audio segment to temporary file"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"segment_{sequence:03d}_{timestamp}.mp3"
        filepath = self.output_dir / "segments" / filename

        # Save audio data (OpenAI returns a response object with stream_to_file method)
        response.stream_to_file(str(filepath))

        return filepath

    def get_synthesis_stats(self) -> Dict[str, Any]:
        """Get statistics about OpenAI audio synthesis performance"""
        stats = super().get_synthesis_stats()
        stats.update({
            "provider": "openai",
            "voice": self.voice,
            "model": self.model,
            "response_format": self.response_format,
            "speed": self.speed
        })
        return stats