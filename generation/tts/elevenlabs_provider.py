import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from .base import BaseTTSSynthesizer, AudioSegment, AudioResult

try:
    from elevenlabs.client import ElevenLabs
    from elevenlabs import Voice, VoiceSettings, play
except ImportError:
    # Fallback for older elevenlabs versions
    from elevenlabs import ElevenLabs, Voice, VoiceSettings, play


class ElevenLabsTTSSynthesizer(BaseTTSSynthesizer):
    """ElevenLabs TTS provider implementation"""

    def __init__(
        self,
        api_key: str,
        output_dir: Path,
        voice_id: str = "JBFqnCBsd6RMkjVDRZzb",
        model_id: str = "eleven_multilingual_v2",
        stability: float = 0.75,
        similarity_boost: float = 0.75,
        style: float = 0.0,
        use_speaker_boost: bool = True,
        **kwargs
    ):
        super().__init__(api_key, output_dir, **kwargs)

        # ElevenLabs-specific settings
        self.voice_id = voice_id
        self.model_id = model_id
        self.stability = stability
        self.similarity_boost = similarity_boost
        self.style = style
        self.use_speaker_boost = use_speaker_boost

        # Initialize ElevenLabs client
        self.client = ElevenLabs(api_key=api_key)
        self.output_format = "mp3_44100_128"

    def execute(self, script_path: str, target_duration: Optional[float] = None) -> AudioResult:
        """
        Convert SRT script to synchronized audio using ElevenLabs

        Args:
            script_path: Path to SRT script file
            target_duration: Optional target duration to match video

        Returns:
            AudioResult with generated audio file and metadata
        """
        start_time = time.time()
        self.logger.info(f"Starting ElevenLabs audio synthesis for script: {script_path}")

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
            self.logger.info(f"ElevenLabs audio synthesis completed in {generation_time:.2f}s")
            self.logger.info(f"Generated audio: {actual_duration:.2f}s, {file_size_mb:.2f}MB")

            return AudioResult(
                success=True,
                audio_path=str(final_audio_path),
                audio_segments=audio_segments,
                total_duration=actual_duration,
                file_size_mb=file_size_mb,
                generation_time=generation_time,
                model_used=self.model_id,
                voice_settings={
                    "voice_id": self.voice_id,
                    "model_id": self.model_id,
                    "stability": self.stability,
                    "similarity_boost": self.similarity_boost,
                    "style": self.style,
                    "use_speaker_boost": self.use_speaker_boost
                }
            )

        except Exception as e:
            self.logger.error(f"ElevenLabs audio synthesis failed: {e}")
            return AudioResult(
                success=False,
                error_message=str(e),
                generation_time=time.time() - start_time,
                model_used=self.model_id
            )

    def _generate_audio_segments(self, subtitles: List[Dict[str, Any]]) -> List[AudioSegment]:
        """Generate audio for each subtitle using ElevenLabs"""

        audio_segments = []

        for subtitle in subtitles:
            self.logger.info(f"Generating ElevenLabs audio for subtitle {subtitle['sequence']}: {subtitle['text'][:50]}...")

            for attempt in range(self.max_retries):
                try:
                    # Generate audio using the correct ElevenLabs API
                    audio = self.client.text_to_speech.convert(
                        text=subtitle['text'],
                        voice_id=self.voice_id,
                        model_id=self.model_id,
                        output_format=self.output_format
                    )

                    # Save audio segment to temporary file
                    segment_path = self._save_audio_segment(audio, subtitle['sequence'])

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
                    self.logger.info(f"Generated ElevenLabs audio segment: {actual_duration:.2f}s")
                    break

                except Exception as e:
                    self.logger.warning(f"ElevenLabs TTS failed for subtitle {subtitle['sequence']} (attempt {attempt + 1}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        self.logger.error(f"Failed to generate ElevenLabs audio for subtitle {subtitle['sequence']}")
                        break

        return audio_segments

    def _save_audio_segment(self, audio, sequence: int) -> Path:
        """Save audio segment to temporary file"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"segment_{sequence:03d}_{timestamp}.mp3"
        filepath = self.output_dir / "segments" / filename

        # Save audio data (ElevenLabs returns generator of bytes chunks)
        with open(filepath, 'wb') as f:
            if hasattr(audio, '__iter__') and not isinstance(audio, (bytes, bytearray)):
                # Audio is a generator, consume it
                for chunk in audio:
                    if chunk:
                        f.write(chunk)
            else:
                # Audio is bytes directly
                f.write(audio)

        return filepath

    def get_synthesis_stats(self) -> Dict[str, Any]:
        """Get statistics about ElevenLabs audio synthesis performance"""
        stats = super().get_synthesis_stats()
        stats.update({
            "provider": "elevenlabs",
            "voice_id": self.voice_id,
            "model_id": self.model_id,
            "output_format": self.output_format
        })
        return stats