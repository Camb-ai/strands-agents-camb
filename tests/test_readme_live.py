"""Live API smoke tests for every README code-block example.

Calls each CAMB tool directly via ``_tool_func()`` (no LLM agent layer) and
verifies the output structure: files exist, JSON has expected keys, WAV magic
bytes, etc.

Run::

    source .env && uv run python -m pytest tests/test_readme_live.py -v -s --timeout=300

Quick check (config-only, no API calls)::

    source .env && uv run python -m pytest tests/test_readme_live.py -v -k "config or api_key or timeout"
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Load .env if CAMB_API_KEY is not already in the environment
# ---------------------------------------------------------------------------
_env_file = Path(__file__).resolve().parent.parent / ".env"
if not os.getenv("CAMB_API_KEY") and _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

from strands_camb import CambAIToolProvider  # noqa: E402

# ---------------------------------------------------------------------------
# Skip the entire module when no API key is available
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.skipif(
    not os.getenv("CAMB_API_KEY"),
    reason="CAMB_API_KEY not set (load .env or export it)",
)




# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _call(tools: dict, name: str, **kwargs) -> str:
    """Call a tool by name and return its string result."""
    return await tools[name]._tool_func(**kwargs)


def _assert_wav_file(path: str) -> None:
    """Assert *path* is a non-empty file whose first four bytes are ``RIFF``."""
    p = Path(path)
    assert p.exists(), f"File does not exist: {path}"
    assert p.stat().st_size > 0, f"File is empty: {path}"
    magic = p.read_bytes()[:4]
    assert magic == b"RIFF", f"Expected RIFF header, got {magic!r}"


def _cleanup_file(path: str | None) -> None:
    """Remove a temp file silently."""
    if path:
        try:
            os.unlink(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Module-level cache for the TTS audio file (avoid redundant API calls)
# ---------------------------------------------------------------------------
_cached_tts_path: str | None = None


def _cleanup_cached_audio() -> None:
    """Called via atexit to remove the cached TTS audio file."""
    if _cached_tts_path:
        _cleanup_file(_cached_tts_path)


import atexit  # noqa: E402

atexit.register(_cleanup_cached_audio)


# ---------------------------------------------------------------------------
# Fixtures — function-scoped so each test gets a fresh event-loop-bound client
# ---------------------------------------------------------------------------

@pytest.fixture
async def tools():
    """Load all 9 tools with a fresh provider (no shared httpx client)."""
    provider = CambAIToolProvider(timeout=120.0)
    loaded = await provider.load_tools()
    return {t.tool_name: t for t in loaded}


@pytest.fixture
async def tts_audio_path(tools):
    """Generate a short WAV via TTS for tests that need audio input (cached)."""
    global _cached_tts_path  # noqa: PLW0603
    if _cached_tts_path is None or not Path(_cached_tts_path).exists():
        result = await _call(tools, "camb_tts", text="Hello world, this is a test audio sample.")
        data = json.loads(result)
        path = data.get("file_path")
        assert path, f"TTS fixture failed: {data}"
        _cached_tts_path = path
    return _cached_tts_path


# ---------------------------------------------------------------------------
# Block 0 — Quick Start: TTS with "Hello world"
# ---------------------------------------------------------------------------

class TestReadmeLive:

    async def test_block0_quick_start_tts(self, tools):
        """Block 0: Convert 'Hello world' to speech."""
        result = await _call(tools, "camb_tts", text="Hello world")
        data = json.loads(result)
        assert data["status"] == "success"
        path = data["file_path"]
        _assert_wav_file(path)
        _cleanup_file(path)

    # ------------------------------------------------------------------
    # Block 1 — Multi-Tool Agent: selective flags + translate
    # ------------------------------------------------------------------

    async def test_block1_selective_tools_count(self):
        """Block 1: Provider with 3 tools enabled loads exactly 3."""
        provider = CambAIToolProvider(
            enable_tts=True,
            enable_translation=True,
            enable_transcription=False,
            enable_translated_tts=False,
            enable_voice_clone=False,
            enable_voice_list=True,
            enable_text_to_sound=False,
            enable_audio_separation=False,
            enable_voice_from_description=False,
        )
        loaded = await provider.load_tools()
        names = {t.tool_name for t in loaded}
        assert names == {"camb_tts", "camb_translate", "camb_list_voices"}

    async def test_block1_translate_good_morning(self, tools):
        """Block 1: Translate 'Good morning' to French (lang 76)."""
        result = await _call(
            tools,
            "camb_translate",
            text="Good morning",
            source_language=1,
            target_language=76,
        )
        data = json.loads(result)
        assert data["status"] == "success"
        assert len(data["translated_text"]) > 0

    # ------------------------------------------------------------------
    # Block 2 — TTS (3 variants)
    # ------------------------------------------------------------------

    async def test_block2_tts_basic(self, tools):
        """Block 2a: Basic TTS with default voice."""
        result = await _call(
            tools,
            "camb_tts",
            text="Hello, how are you today?",
            voice_id=147320,
            language="en-us",
        )
        data = json.loads(result)
        assert data["status"] == "success"
        _assert_wav_file(data["file_path"])
        _cleanup_file(data["file_path"])

    async def test_block2_tts_mars_pro(self, tools):
        """Block 2b: TTS with mars-pro model."""
        result = await _call(
            tools,
            "camb_tts",
            text="Welcome to our podcast",
            speech_model="mars-pro",
        )
        data = json.loads(result)
        assert data["status"] == "success"
        _assert_wav_file(data["file_path"])
        _cleanup_file(data["file_path"])

    async def test_block2_tts_speed(self, tools):
        """Block 2c: TTS with speed=1.5."""
        result = await _call(
            tools,
            "camb_tts",
            text="Breaking news",
            speed=1.5,
        )
        data = json.loads(result)
        assert data["status"] == "success"
        _assert_wav_file(data["file_path"])
        _cleanup_file(data["file_path"])

    # ------------------------------------------------------------------
    # Block 3 — Translation
    # ------------------------------------------------------------------

    async def test_block3_translation_en_es(self, tools):
        """Block 3: Translate English to Spanish, formality=1."""
        result = await _call(
            tools,
            "camb_translate",
            text="Hello, how are you?",
            source_language=1,
            target_language=54,
            formality=1,
        )
        data = json.loads(result)
        assert data["status"] == "success"
        assert len(data["translated_text"]) > 0

    # ------------------------------------------------------------------
    # Block 4 — Transcription
    # ------------------------------------------------------------------

    async def test_block4_transcription(self, tools, tts_audio_path):
        """Block 4: Transcribe a TTS-generated WAV file."""
        result = await _call(
            tools,
            "camb_transcribe",
            language=1,
            audio_file_path=tts_audio_path,
        )
        data = json.loads(result)
        assert data.get("text") or data.get("segments"), f"No transcription output: {data}"

    # ------------------------------------------------------------------
    # Block 5 — Translated TTS
    # ------------------------------------------------------------------

    async def test_block5_translated_tts(self, tools):
        """Block 5: Translate English→French and generate speech."""
        result = await _call(
            tools,
            "camb_translated_tts",
            text="Good morning everyone",
            source_language=1,
            target_language=76,
        )
        data = json.loads(result)
        assert data["status"] == "success"
        path = data["file_path"]
        p = Path(path)
        assert p.exists()
        assert p.stat().st_size > 0
        _cleanup_file(path)

    # ------------------------------------------------------------------
    # Block 6 — Voice Cloning
    # ------------------------------------------------------------------

    async def test_block6_voice_clone(self, tools, tts_audio_path):
        """Block 6: Clone a voice from a TTS-generated audio sample."""
        unique = uuid.uuid4().hex[:8]
        result = await _call(
            tools,
            "camb_clone_voice",
            voice_name=f"readme_test_{unique}",
            audio_file_path=tts_audio_path,
            gender=0,
        )
        data = json.loads(result)
        assert data["status"] == "created"
        assert data["voice_id"] is not None

    # ------------------------------------------------------------------
    # Block 7 — List Voices
    # ------------------------------------------------------------------

    async def test_block7_list_voices(self, tools):
        """Block 7: List all available voices."""
        result = await _call(tools, "camb_list_voices")
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) > 0
        assert "id" in data[0]
        assert "voice_name" in data[0]

    # ------------------------------------------------------------------
    # Block 8 — Text-to-Sound (music + sound effect)
    # ------------------------------------------------------------------

    async def test_block8_text_to_sound_music(self, tools):
        """Block 8a: Generate music from a text prompt."""
        result = await _call(
            tools,
            "camb_text_to_sound",
            prompt="upbeat electronic music with a driving beat",
            duration=5.0,
            audio_type="music",
        )
        data = json.loads(result)
        assert data["status"] == "success"
        _assert_wav_file(data["file_path"])
        _cleanup_file(data["file_path"])

    async def test_block8_text_to_sound_effect(self, tools):
        """Block 8b: Generate a sound effect."""
        result = await _call(
            tools,
            "camb_text_to_sound",
            prompt="rain on a tin roof",
            audio_type="sound",
        )
        data = json.loads(result)
        assert data["status"] == "success"
        _assert_wav_file(data["file_path"])
        _cleanup_file(data["file_path"])

    # ------------------------------------------------------------------
    # Block 9 — Audio Separation
    # ------------------------------------------------------------------

    async def test_block9_audio_separation(self, tools, tts_audio_path):
        """Block 9: Separate vocals from background in a TTS WAV."""
        result = await _call(
            tools,
            "camb_audio_separation",
            audio_file_path=tts_audio_path,
        )
        data = json.loads(result)
        assert data.get("foreground_audio_url") or data.get("background_audio_url"), (
            f"Expected separation URLs: {data}"
        )

    # ------------------------------------------------------------------
    # Block 10 — Voice from Description
    # ------------------------------------------------------------------

    async def test_block10_voice_from_description(self, tools):
        """Block 10: Generate a voice from a text description."""
        result = await _call(
            tools,
            "camb_voice_from_description",
            text=(
                "Welcome to the show! Today we are exploring the amazing world "
                "of AI-generated voices and how they can transform content creation."
            ),
            voice_description="warm, friendly female narrator",
        )
        data = json.loads(result)
        assert data["status"] == "completed"
        assert isinstance(data["previews"], list)

    # ------------------------------------------------------------------
    # Block 11 — TTS-Focused config
    # ------------------------------------------------------------------

    async def test_block11_tts_focused_config(self):
        """Block 11: TTS-focused provider loads exactly 4 tools."""
        provider = CambAIToolProvider(
            enable_tts=True,
            enable_voice_list=True,
            enable_voice_clone=True,
            enable_voice_from_description=True,
            enable_translation=False,
            enable_transcription=False,
            enable_translated_tts=False,
            enable_text_to_sound=False,
            enable_audio_separation=False,
        )
        loaded = await provider.load_tools()
        assert len(loaded) == 4
        names = {t.tool_name for t in loaded}
        assert names == {
            "camb_tts",
            "camb_list_voices",
            "camb_clone_voice",
            "camb_voice_from_description",
        }

    # ------------------------------------------------------------------
    # Block 12 — Translation-Focused config
    # ------------------------------------------------------------------

    async def test_block12_translation_focused_config(self):
        """Block 12: Translation-focused provider loads exactly 3 tools."""
        provider = CambAIToolProvider(
            enable_translation=True,
            enable_translated_tts=True,
            enable_transcription=True,
            enable_tts=False,
            enable_voice_list=False,
            enable_voice_clone=False,
            enable_text_to_sound=False,
            enable_audio_separation=False,
            enable_voice_from_description=False,
        )
        loaded = await provider.load_tools()
        assert len(loaded) == 3
        names = {t.tool_name for t in loaded}
        assert names == {"camb_translate", "camb_translated_tts", "camb_transcribe"}

    # ------------------------------------------------------------------
    # Block 13 — API Key config: env var pickup
    # ------------------------------------------------------------------

    async def test_block13_api_key_from_env(self):
        """Block 13: Default constructor picks up CAMB_API_KEY from env."""
        provider = CambAIToolProvider()
        assert provider._helpers._api_key == os.environ["CAMB_API_KEY"]

    # ------------------------------------------------------------------
    # Block 14 — Timeouts config
    # ------------------------------------------------------------------

    async def test_block14_timeout_config(self):
        """Block 14: Constructor passes timeout/polling params to helpers."""
        provider = CambAIToolProvider(
            timeout=60.0,
            max_poll_attempts=60,
            poll_interval=2.0,
        )
        h = provider._helpers
        assert h._timeout == 60.0
        assert h._max_poll_attempts == 60
        assert h._poll_interval == 2.0

    # ------------------------------------------------------------------
    # Block 15 — Agent Integration: 4-step workflow
    # ------------------------------------------------------------------

    async def test_block15_workflow_list_voices(self, tools):
        """Block 15, step 1: List available voices."""
        result = await _call(tools, "camb_list_voices")
        data = json.loads(result)
        assert isinstance(data, list) and len(data) > 0

    async def test_block15_workflow_tts(self, tools):
        """Block 15, step 2: Generate speech saying 'Hello world'."""
        result = await _call(tools, "camb_tts", text="Hello world")
        data = json.loads(result)
        assert data["status"] == "success"
        _assert_wav_file(data["file_path"])
        _cleanup_file(data["file_path"])

    async def test_block15_workflow_translated_tts(self, tools):
        """Block 15, step 3: Translate to Spanish and generate speech."""
        result = await _call(
            tools,
            "camb_translated_tts",
            text="Hello world",
            source_language=1,
            target_language=54,
        )
        data = json.loads(result)
        assert data["status"] == "success"
        path = data["file_path"]
        assert Path(path).exists()
        _cleanup_file(path)

    async def test_block15_workflow_separation(self, tools, tts_audio_path):
        """Block 15, step 4: Separate vocals from the last audio file."""
        result = await _call(
            tools,
            "camb_audio_separation",
            audio_file_path=tts_audio_path,
        )
        data = json.loads(result)
        assert data.get("foreground_audio_url") or data.get("background_audio_url")
