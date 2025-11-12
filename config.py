import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field, model_validator
from pathlib import Path
from typing import Optional, List

# ✅ Load .env file FIRST
load_dotenv()


# ✅ Load Google API Keys function (define before Settings)
def load_google_api_keys() -> List[str]:
    """Load Google API keys from environment"""
    keys = []
    for i in range(1, 10):
        key = os.getenv(f"GOOGLE_API_KEY_{i}")
        if key and key.strip():
            keys.append(key.strip())
    
    if not keys:
        raise ValueError(
            "No Google API keys found in .env!\n"
            "Please add:\n"
            "  GOOGLE_API_KEY_1=your_first_key\n"
            "  GOOGLE_API_KEY_2=your_second_key\n"
            "  GOOGLE_API_KEY_3=your_third_key"
        )
    
    print(f"Loaded {len(keys)} Google API keys")
    return keys


class Settings(BaseSettings):
    # === GEMINI (Google Generative AI) ===
    google_api_endpoint: str = Field(
        default="https://generativelanguage.googleapis.com",
        alias="GOOGLE_API_ENDPOINT"
    )
    
    # ✅ ADD: Property to get API keys
    @property
    def google_api_keys(self) -> List[str]:
        """Get list of Google API keys"""
        return load_google_api_keys()
    
    # LLM models (Gemini)
    reasoning_model: str = Field(default="gemini-2.5-flash")
    multimodal_model: str = Field(default="gemini-2.5-pro")

    # === TTS (ElevenLabs only) ===
    tts_provider: str = "elevenlabs"
    elevenlabs_api_key: Optional[str] = None
    elevenlabs_voice_id: str = "Qggl4b0xRMiqOwhPtVWT"
    elevenlabs_model_id: str = "eleven_v3"
    elevenlabs_stability: float = 0.75
    elevenlabs_similarity_boost: float = 0.75
    elevenlabs_style: float = 0.0
    elevenlabs_use_speaker_boost: bool = True

    tts_max_retries: int = 3
    tts_timeout: int = 120

    # === Paths ===
    output_dir: Path = Path("output")

    @property
    def scenes_dir(self) -> Path: 
        return self.output_dir / "scenes"
    
    @property
    def animations_dir(self) -> Path: 
        return self.output_dir / "animations"
    
    @property
    def audio_dir(self) -> Path: 
        return self.output_dir / "audio"
    
    @property
    def scripts_dir(self) -> Path: 
        return self.output_dir / "scripts"
    
    @property
    def final_dir(self) -> Path: 
        return self.output_dir / "final"
    
    @property
    def analyses_dir(self) -> Path: 
        return self.output_dir / "analyses"
    
    @property
    def rendering_dir(self) -> Path: 
        return self.output_dir / "rendering"
    
    @property
    def generation_dir(self) -> Path: 
        return self.output_dir / "generation"

    # === Manim ===
    manim_quality: str = "p"
    manim_background_color: str = "#0f0f0f"
    manim_frame_rate: int = 60
    manim_render_timeout: int = 300
    manim_max_retries: int = 3
    manim_max_scene_duration: float = 30.0
    manim_total_video_duration_target: float = 120.0

    # === Generation params ===
    animation_temperature: float = 0.5
    animation_max_retries_per_scene: int = 3
    animation_enable_simplification: bool = True

    script_generation_temperature: float = 0.5
    script_generation_max_retries: int = 3
    script_generation_timeout: int = 180

    llm_max_retries: int = 3
    llm_timeout: int = 120

    # === Video ===
    video_codec: str = "libx264"
    video_preset: str = "medium"
    video_crf: int = 23
    audio_codec: str = "aac"
    audio_bitrate: str = "128k"

    # === Subtitle ===
    subtitle_burn_in: bool = True
    subtitle_font_size: int = 24
    subtitle_font_color: str = "white"
    subtitle_background: bool = True
    subtitle_background_opacity: float = 0.5
    subtitle_position: str = "bottom"

    # === Composition ===
    video_composition_max_retries: int = 3
    video_composition_timeout: int = 600

    # === Language ===
    target_language: str = "English"

    @model_validator(mode='after')
    def _require_elevenlabs_key_if_provider_is_el(self):
        if self.tts_provider == "elevenlabs":
            if not self.elevenlabs_api_key or not str(self.elevenlabs_api_key).strip():
                raise ValueError('elevenlabs_api_key is required when tts_provider="elevenlabs"')
        return self

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"

    def create_directories(self):
        for d in [
            self.output_dir,
            self.scenes_dir,
            self.animations_dir,
            self.audio_dir,
            self.scripts_dir,
            self.final_dir,
            self.analyses_dir,
            self.rendering_dir,
            self.generation_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


def get_settings():
    return Settings()


# ✅ Initialize
settings = get_settings()
settings.create_directories()

# ✅ Export API keys for direct import
google_api_keys = load_google_api_keys()