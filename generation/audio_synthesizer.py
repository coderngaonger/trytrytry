import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydub import AudioSegment as PydubAudioSegment

try:
    from elevenlabs.client import ElevenLabs
    from elevenlabs import Voice, VoiceSettings, play
except ImportError:
    # Fallback for older elevenlabs versions
    from elevenlabs import ElevenLabs, Voice, VoiceSettings, play
from pydantic import BaseModel, Field


class AudioSegment(BaseModel):
    """Single audio segment from TTS synthesis"""
    text: str
    start_time: float  # seconds
    end_time: float    # seconds
    audio_path: Optional[str] = None
    duration: Optional[float] = None
    file_size: Optional[float] = None  # MB


class AudioResult(BaseModel):
    """Result of audio synthesis"""
    success: bool
    audio_path: Optional[str] = None
    audio_segments: List[AudioSegment] = Field(default_factory=list)
    total_duration: Optional[float] = None
    file_size_mb: Optional[float] = None
    error_message: Optional[str] = None

    # Metadata
    generation_time: Optional[float] = None
    voice_settings: Optional[Dict[str, Any]] = None
    model_used: str = "eleven_monolingual_v1"


class AudioSynthesizer:
    """
    Audio Synthesizer: Converts timestamped scripts into synchronized speech audio
    using ElevenLabs Text-to-Speech API.
    """

    def __init__(
        self,
        api_key: str,
        output_dir: Path,
        voice_id: str = "default",
        model_id: str = "eleven_monolingual_v1",
        stability: float = 0.75,
        similarity_boost: float = 0.75,
        style: float = 0.0,
        use_speaker_boost: bool = True,
        max_retries: int = 3,
        timeout: int = 120
    ):
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.voice_id = voice_id
        self.model_id = model_id
        self.stability = stability
        self.similarity_boost = similarity_boost
        self.style = style
        self.use_speaker_boost = use_speaker_boost
        self.max_retries = max_retries
        self.timeout = timeout

        # Initialize ElevenLabs client
        self.client = ElevenLabs(api_key=api_key)

        # Store voice settings for API calls
        self.voice_id = voice_id
        self.model_id = model_id
        self.output_format = "mp3_44100_128"

        # Ensure output directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "segments").mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute(self, script_path: str, target_duration: Optional[float] = None) -> AudioResult:
        """
        Convert SRT script to synchronized audio

        Args:
            script_path: Path to SRT script file
            target_duration: Optional target duration to match video

        Returns:
            AudioResult with generated audio file and metadata
        """
        start_time = time.time()
        self.logger.info(f"Starting audio synthesis for script: {script_path}")

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
            self.logger.info(f"Audio synthesis completed in {generation_time:.2f}s")
            self.logger.info(f"Generated audio: {actual_duration:.2f}s, {file_size_mb:.2f}MB")

            return AudioResult(
                success=True,
                audio_path=str(final_audio_path),
                audio_segments=audio_segments,
                total_duration=actual_duration,
                file_size_mb=file_size_mb,
                generation_time=generation_time,
                model_used=self.model_id
            )

        except Exception as e:
            self.logger.error(f"Audio synthesis failed: {e}")
            return AudioResult(
                success=False,
                error_message=str(e),
                generation_time=time.time() - start_time,
                model_used=self.model_id
            )

    def _parse_srt_file(self, script_file: Path) -> List[Dict[str, Any]]:
        """Parse SRT file and extract subtitles with timing"""

        subtitles = []

        with open(script_file, 'r', encoding='utf-8') as f:
            content = f.read()

        blocks = content.strip().split('\n\n')

        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue

            try:
                # Parse sequence number
                sequence = int(lines[0].strip())

                # Parse timestamps
                timestamp_line = lines[1].strip()
                if '-->' not in timestamp_line:
                    continue

                start_time_str, end_time_str = [t.strip() for t in timestamp_line.split('-->')]

                # Convert timestamp to seconds
                start_time = self._timestamp_to_seconds(start_time_str)
                end_time = self._timestamp_to_seconds(end_time_str)

                # Parse text (may be multiple lines)
                text = ' '.join(line.strip() for line in lines[2:] if line.strip())

                subtitles.append({
                    'sequence': sequence,
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': text,
                    'duration': end_time - start_time
                })

            except Exception as e:
                self.logger.warning(f"Error parsing subtitle block: {e}")
                continue

        return subtitles

    def _normalize_timestamp(self, timestamp: str) -> str:
        """Normalize malformed timestamps to proper HH:MM:SS,mmm format"""
        
        timestamp = timestamp.strip()
        timestamp = timestamp.replace('.', ',').replace(':', ',')
        parts = timestamp.split(',')
        
        if len(parts) == 3:
            minutes, seconds, milliseconds = parts
            hours = "00"
        elif len(parts) == 4:
            hours, minutes, seconds, milliseconds = parts
        else:
            raise ValueError(f"Cannot parse timestamp: {timestamp}")
        
        hours = int(hours)
        minutes = int(minutes)
        seconds = int(seconds)
        milliseconds = int(milliseconds)
        
        if minutes >= 60:
            hours += minutes // 60
            minutes = minutes % 60
        
        if seconds >= 60:
            minutes += seconds // 60
            seconds = seconds % 60
        
        if milliseconds >= 1000:
            seconds += milliseconds // 1000
            milliseconds = milliseconds % 1000
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def _timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert SRT timestamp (HH:MM:SS,mmm) to seconds"""

        normalized = self._normalize_timestamp(timestamp)
        normalized = normalized.replace(',', ':')
        parts = normalized.split(':')

        if len(parts) != 4:
            raise ValueError(f"Invalid timestamp format: {timestamp}")

        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        milliseconds = int(parts[3])

        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000

    def _seconds_to_timestamp(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""

        hours = int(seconds // 3600)
        remaining = seconds % 3600
        minutes = int(remaining // 60)
        seconds = int(remaining % 60)
        milliseconds = int((remaining % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def _generate_audio_segments(self, subtitles: List[Dict[str, Any]]) -> List[AudioSegment]:
        """Generate audio for each subtitle using ElevenLabs with duration validation and speed adjustment"""

        audio_segments = []

        for i, subtitle in enumerate(subtitles):
            self.logger.info(f"Generating audio for subtitle {subtitle['sequence']}: {subtitle['text'][:50]}...")

            # Get next subtitle for time slot calculation
            next_subtitle = subtitles[i + 1] if i + 1 < len(subtitles) else None

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

                    # Validate and adjust audio duration if needed
                    adjusted_segment_path = self._validate_and_adjust_audio_segment(
                        segment_path, subtitle, next_subtitle
                    )

                    # Get final actual duration
                    actual_duration = self._get_audio_duration(adjusted_segment_path)

                    audio_segment = AudioSegment(
                        text=subtitle['text'],
                        start_time=subtitle['start_time'],
                        end_time=subtitle['end_time'],
                        audio_path=str(adjusted_segment_path),
                        duration=actual_duration,
                        file_size=adjusted_segment_path.stat().st_size / (1024 * 1024)
                    )

                    audio_segments.append(audio_segment)
                    self.logger.info(f"Generated audio segment: {actual_duration:.2f}s")
                    break

                except Exception as e:
                    self.logger.warning(f"TTS failed for subtitle {subtitle['sequence']} (attempt {attempt + 1}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        self.logger.error(f"Failed to generate audio for subtitle {subtitle['sequence']}")
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

    def _concatenate_audio_segments(self, audio_segments: List[AudioSegment], script_stem: str) -> Path:
        """Concatenate audio segments with proper timing using pydub"""

        if not audio_segments:
            raise ValueError("No audio segments to concatenate")

        self.logger.info(f"Concatenating {len(audio_segments)} audio segments with timing")

        audio_segments.sort(key=lambda s: s.start_time)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{script_stem}_audio_{timestamp}.mp3"
        output_path = self.output_dir / output_filename

        combined = PydubAudioSegment.silent(duration=0)
        current_time = 0.0

        for i, segment in enumerate(audio_segments):
            if not segment.audio_path:
                self.logger.warning(f"Skipping segment {i+1} - no audio path")
                continue

            gap = segment.start_time - current_time

            # Add silence gap if needed
            if gap > 0.05:
                silence_duration_ms = int(gap * 1000)
                combined += PydubAudioSegment.silent(duration=silence_duration_ms)
                self.logger.debug(f"Added {gap:.2f}s silence before segment {i+1} (seq: {segment.text[:20]}...)")

            # Load and add the audio segment
            try:
                audio_clip = PydubAudioSegment.from_mp3(segment.audio_path)
                combined += audio_clip

                # Log timing info for debugging
                actual_duration = len(audio_clip) / 1000.0
                self.logger.debug(f"Added segment {i+1} (seq: {segment.sequence}): {actual_duration:.2f}s at {segment.start_time:.2f}s")

                current_time = segment.start_time + actual_duration

                # Check for potential overlap with next segment
                if i + 1 < len(audio_segments):
                    next_segment = audio_segments[i + 1]
                    if current_time > next_segment.start_time:
                        self.logger.warning(f"Segment {i+1} (seq: {segment.sequence}) extends into next segment by {current_time - next_segment.start_time:.2f}s")

            except Exception as e:
                self.logger.error(f"Failed to load audio segment {i+1}: {e}")
                continue

        # Calculate final duration and log summary
        final_duration = len(combined) / 1000.0
        self.logger.info(f"Successfully concatenated audio with timing: {output_filename}")
        self.logger.info(f"Final audio duration: {final_duration:.2f}s")

        # Export the combined audio
        combined.export(output_path, format="mp3", bitrate="192k")

        return output_path



    def _add_silence_padding(self, audio_path: Path, padding_duration: float) -> Path:
        """Add silence padding to match target duration"""

        self.logger.info(f"Adding {padding_duration:.2f}s silence padding")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        padded_filename = f"{audio_path.stem}_padded_{timestamp}.mp3"
        padded_path = self.output_dir / padded_filename

        audio = PydubAudioSegment.from_mp3(audio_path)
        silence = PydubAudioSegment.silent(duration=int(padding_duration * 1000))
        padded_audio = audio + silence
        padded_audio.export(padded_path, format="mp3", bitrate="192k")

        self.logger.info(f"Added silence padding: {padded_filename}")
        return padded_path

    def _get_audio_duration(self, audio_path: Path) -> Optional[float]:
        """Get audio duration using pydub"""

        try:
            audio = PydubAudioSegment.from_mp3(audio_path)
            return len(audio) / 1000.0
        except Exception as e:
            self.logger.warning(f"Could not get audio duration for {audio_path}: {e}")
            return None

    def _speed_up_audio(self, audio_path: Path, speed_factor: float) -> Path:
        """Speed up audio by a given factor while preserving pitch"""

        try:
            audio = PydubAudioSegment.from_mp3(audio_path)

            # Calculate new sample rate for faster playback
            new_sample_rate = int(audio.frame_rate * speed_factor)

            # Change the frame rate to speed up audio (this also raises pitch)
            sped_up_audio = audio._spawn(audio.raw_data, overrides={'frame_rate': new_sample_rate})

            # Export with original frame rate to restore pitch
            sped_up_audio = sped_up_audio.set_frame_rate(audio.frame_rate)

            # Save sped up audio with new filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sped_up_filename = f"{audio_path.stem}_spedup_{int(speed_factor*100)}_{timestamp}.mp3"
            sped_up_path = audio_path.parent / sped_up_filename

            sped_up_audio.export(sped_up_path, format="mp3", bitrate="192k")

            self.logger.info(f"Audio sped up by {speed_factor*100:.0f}%: {sped_up_filename}")
            return sped_up_path

        except Exception as e:
            self.logger.error(f"Failed to speed up audio {audio_path}: {e}")
            return audio_path  # Return original if speedup fails

    def _validate_and_adjust_audio_segment(self, segment_path: Path, current_segment: Dict[str, Any], next_segment: Optional[Dict[str, Any]] = None) -> Path:
        """Validate audio segment duration and adjust if it exceeds allocated time"""

        actual_duration = self._get_audio_duration(segment_path)
        if actual_duration is None:
            return segment_path

        # Calculate allocated time slot
        allocated_time = current_segment['duration']

        # If next segment exists, calculate available time between segments
        if next_segment:
            allocated_time = next_segment['start_time'] - current_segment['start_time']

        self.logger.debug(f"Segment {current_segment['sequence']}: actual={actual_duration:.2f}s, allocated={allocated_time:.2f}s")

        # If audio exceeds allocated time, try speed adjustments
        if actual_duration > allocated_time:
            self.logger.warning(f"Audio segment {current_segment['sequence']} exceeds time slot: {actual_duration:.2f}s > {allocated_time:.2f}s")

            speed_factors = [1.1, 1.25, 1.5]  # 10%, 25%, 50% speed increases
            current_segment_path = segment_path

            for speed_factor in speed_factors:
                new_duration = actual_duration / speed_factor
                if new_duration <= allocated_time:
                    current_segment_path = self._speed_up_audio(segment_path, speed_factor)
                    final_duration = self._get_audio_duration(current_segment_path)
                    self.logger.info(f"Segment {current_segment['sequence']} sped up by {speed_factor*100:.0f}%: {final_duration:.2f}s <= {allocated_time:.2f}s")
                    break
                else:
                    self.logger.debug(f"Speed {speed_factor*100:.0f}% still too long: {new_duration:.2f}s > {allocated_time:.2f}s")

            # Clean up intermediate files
            if current_segment_path != segment_path:
                try:
                    segment_path.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to clean up intermediate file {segment_path}: {e}")

            return current_segment_path

        return segment_path

    def cleanup_temp_files(self):
        """Clean up temporary audio segment files"""

        segments_dir = self.output_dir / "segments"
        if segments_dir.exists():
            for file_path in segments_dir.glob("*.mp3"):
                try:
                    file_path.unlink()
                    self.logger.debug(f"Cleaned up temporary file: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to clean up {file_path}: {e}")

    def get_synthesis_stats(self) -> Dict[str, Any]:
        """Get statistics about audio synthesis performance"""
        return {
            "voice_id": self.voice_id,
            "model_id": self.model_id,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "output_dir": str(self.output_dir)
        }