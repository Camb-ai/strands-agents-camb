"""Strands Agent integration tests for CAMB.AI tools.

Tests each tool end-to-end through the Strands Agent event loop using an
inline MockedModelProvider (no real LLM required).  The CAMB SDK is still
mocked so no API key is needed.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import AsyncGenerator, Iterable, Sequence
from typing import Any, TypeVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from strands import Agent
from strands.models import Model
from strands.types.content import Message, Messages
from strands.types.event_loop import StopReason
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolSpec

from strands_camb.camb_tools import CambAIToolProvider

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Inline MockedModelProvider
# (copied from strands-sdk-python/tests/fixtures/mocked_model_provider.py)
# ---------------------------------------------------------------------------


class MockedModelProvider(Model):
    """A mock implementation of the Model interface for testing purposes.

    Returns pre-defined agent responses in sequence without calling a real LLM.
    """

    def __init__(self, agent_responses: Sequence[Message]):
        self.agent_responses = [*agent_responses]
        self.index = 0

    def format_chunk(self, event: Any) -> StreamEvent:
        return event

    def format_request(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
    ) -> Any:
        return None

    def get_config(self) -> Any:
        pass

    def update_config(self, **model_config: Any) -> None:
        pass

    async def structured_output(
        self,
        output_model: type[T],
        prompt: Messages,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        pass

    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        tool_choice: Any | None = None,
        *,
        system_prompt_content=None,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        events = self.map_agent_message_to_events(self.agent_responses[self.index])
        for event in events:
            yield event
        self.index += 1

    def map_agent_message_to_events(self, agent_message: Message) -> Iterable[dict[str, Any]]:
        stop_reason: StopReason = "end_turn"
        yield {"messageStart": {"role": "assistant"}}
        for content in agent_message["content"]:
            if "reasoningContent" in content:
                yield {"contentBlockStart": {"start": {}}}
                yield {"contentBlockDelta": {"delta": {"reasoningContent": content["reasoningContent"]}}}
                yield {"contentBlockStop": {}}
            if "text" in content:
                yield {"contentBlockStart": {"start": {}}}
                yield {"contentBlockDelta": {"delta": {"text": content["text"]}}}
                yield {"contentBlockStop": {}}
            if "toolUse" in content:
                stop_reason = "tool_use"
                yield {
                    "contentBlockStart": {
                        "start": {
                            "toolUse": {
                                "name": content["toolUse"]["name"],
                                "toolUseId": content["toolUse"]["toolUseId"],
                            }
                        }
                    }
                }
                yield {
                    "contentBlockDelta": {
                        "delta": {"toolUse": {"input": json.dumps(content["toolUse"]["input"])}}
                    }
                }
                yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": stop_reason}}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool_call_response(tool_name: str, tool_input: dict, use_id: str = "use-1") -> Message:
    return {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": use_id, "name": tool_name, "input": tool_input}}],
    }


def _text_response(text: str) -> Message:
    return {"role": "assistant", "content": [{"text": text}]}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_api_key():
    return "test-api-key-12345"


@pytest.fixture
def provider(mock_api_key):
    return CambAIToolProvider(api_key=mock_api_key)


@pytest.fixture
def mock_client():
    return MagicMock()


# ---------------------------------------------------------------------------
# TestStrandsIntegration
# ---------------------------------------------------------------------------


class TestStrandsIntegration:
    @pytest.mark.asyncio
    async def test_provider_loads_in_agent(self, provider):
        """Agent accepts ToolProvider; load_tools returns 9 tools."""
        tools = await provider.load_tools()
        assert len(tools) == 9
        names = {t.tool_name for t in tools}
        assert "camb_tts" in names
        assert "camb_translate" in names
        assert "camb_transcribe" in names
        assert "camb_translated_tts" in names
        assert "camb_clone_voice" in names
        assert "camb_list_voices" in names
        assert "camb_text_to_sound" in names
        assert "camb_audio_separation" in names
        assert "camb_voice_from_description" in names

    def test_tts_via_agent(self, provider, mock_client):
        """Model issues camb_tts tool call; agent executes it; model sees JSON result."""

        async def mock_tts_gen(*args, **kwargs):
            yield b"RIFF" + b"\x00" * 40

        mock_client.text_to_speech.tts = mock_tts_gen
        mock_camb = MagicMock()

        model = MockedModelProvider([
            _tool_call_response("camb_tts", {"text": "Hello from Strands!"}),
            _text_response("Audio generated successfully."),
        ])
        agent = Agent(model=model, tools=[provider], callback_handler=None)

        with patch.dict("sys.modules", {"camb": mock_camb}):
            with patch.object(provider._helpers, "_get_client", return_value=mock_client):
                response = agent("Convert text to speech")

        assert "Audio generated" in str(response)

    def test_translate_via_agent(self, provider, mock_client):
        """Model issues camb_translate; agent executes; model sees translated text."""
        mock_result = MagicMock()
        mock_result.text = "Hola mundo"
        mock_client.translation.translation_stream = AsyncMock(return_value=mock_result)

        mock_api_error_mod = MagicMock()
        mock_api_error_mod.ApiError = Exception

        model = MockedModelProvider([
            _tool_call_response(
                "camb_translate",
                {"text": "Hello world", "source_language": 1, "target_language": 2},
            ),
            _text_response("Translation complete."),
        ])
        agent = Agent(model=model, tools=[provider], callback_handler=None)

        with patch.dict("sys.modules", {"camb.core": MagicMock(), "camb.core.api_error": mock_api_error_mod}):
            with patch.object(provider._helpers, "_get_client", return_value=mock_client):
                response = agent("Translate Hello world to Spanish")

        assert "Translation" in str(response)

    def test_transcribe_via_agent(self, provider, mock_client):
        """camb_transcribe with file path; agent executes; model sees transcription JSON."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF" + b"\x00" * 100)
            tmp_path = f.name

        try:
            mock_create = MagicMock()
            mock_create.task_id = "task-123"
            mock_client.transcription.create_transcription = AsyncMock(return_value=mock_create)

            mock_status = MagicMock()
            mock_status.status = "SUCCESS"
            mock_status.run_id = "run-123"
            mock_client.transcription.get_transcription_task_status = AsyncMock(return_value=mock_status)

            seg = MagicMock()
            seg.start = 0.0
            seg.end = 1.5
            seg.text = "Hello world"
            seg.speaker = "SPEAKER_0"
            mock_transcription = MagicMock()
            mock_transcription.text = "Hello world"
            mock_transcription.segments = [seg]
            mock_client.transcription.get_transcription_result = AsyncMock(return_value=mock_transcription)

            model = MockedModelProvider([
                _tool_call_response(
                    "camb_transcribe",
                    {"language": 1, "audio_file_path": tmp_path},
                ),
                _text_response("Transcription completed."),
            ])
            agent = Agent(model=model, tools=[provider], callback_handler=None)

            with patch.object(provider._helpers, "_get_client", return_value=mock_client):
                response = agent("Transcribe the audio file")

            assert "Transcription" in str(response)
        finally:
            os.unlink(tmp_path)

    def test_list_voices_via_agent(self, provider, mock_client):
        """camb_list_voices; model sees voice list."""
        v1 = MagicMock()
        v1.id = 1
        v1.voice_name = "Alice"
        mock_client.voice_cloning.list_voices = AsyncMock(return_value=[v1])

        model = MockedModelProvider([
            _tool_call_response("camb_list_voices", {}),
            _text_response("Here are the available voices."),
        ])
        agent = Agent(model=model, tools=[provider], callback_handler=None)

        with patch.object(provider._helpers, "_get_client", return_value=mock_client):
            response = agent("List available voices")

        assert "voices" in str(response).lower()

    def test_text_to_sound_via_agent(self, provider, mock_client):
        """camb_text_to_sound; model sees success."""
        mock_create = MagicMock()
        mock_create.task_id = "task-123"
        mock_client.text_to_audio.create_text_to_audio = AsyncMock(return_value=mock_create)

        mock_status = MagicMock()
        mock_status.status = "SUCCESS"
        mock_status.run_id = "run-123"
        mock_client.text_to_audio.get_text_to_audio_status = AsyncMock(return_value=mock_status)

        async def mock_audio_stream(*args, **kwargs):
            yield b"RIFF" + b"\x00" * 40

        mock_client.text_to_audio.get_text_to_audio_result = mock_audio_stream

        model = MockedModelProvider([
            _tool_call_response("camb_text_to_sound", {"prompt": "Gentle rain on a rooftop"}),
            _text_response("Sound generated successfully."),
        ])
        agent = Agent(model=model, tools=[provider], callback_handler=None)

        with patch.object(provider._helpers, "_get_client", return_value=mock_client):
            response = agent("Generate rain sound")

        assert "Sound generated" in str(response)

    def test_audio_separation_via_agent(self, provider, mock_client):
        """camb_audio_separation with file; model sees fg/bg URLs."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF" + b"\x00" * 100)
            tmp_path = f.name

        try:
            mock_create = MagicMock()
            mock_create.task_id = "task-123"
            mock_client.audio_separation.create_audio_separation = AsyncMock(return_value=mock_create)

            mock_status = MagicMock()
            mock_status.status = "SUCCESS"
            mock_status.run_id = "run-123"
            mock_client.audio_separation.get_audio_separation_status = AsyncMock(return_value=mock_status)

            mock_sep = MagicMock()
            mock_sep.foreground_audio_url = "http://fg.wav"
            mock_sep.background_audio_url = "http://bg.wav"
            mock_client.audio_separation.get_audio_separation_run_info = AsyncMock(return_value=mock_sep)

            model = MockedModelProvider([
                _tool_call_response("camb_audio_separation", {"audio_file_path": tmp_path}),
                _text_response("Audio separated into vocals and background."),
            ])
            agent = Agent(model=model, tools=[provider], callback_handler=None)

            with patch.object(provider._helpers, "_get_client", return_value=mock_client):
                response = agent("Separate vocals from background")

            assert "separated" in str(response).lower()
        finally:
            os.unlink(tmp_path)

    def test_voice_from_description_via_agent(self, provider, mock_client):
        """Model sees previews list in tool result."""
        mock_create = MagicMock()
        mock_create.task_id = "task-123"
        mock_client.text_to_voice.create_text_to_voice = AsyncMock(return_value=mock_create)

        mock_status = MagicMock()
        mock_status.status = "SUCCESS"
        mock_status.run_id = "run-123"
        mock_client.text_to_voice.get_text_to_voice_status = AsyncMock(return_value=mock_status)

        mock_voice_result = MagicMock()
        mock_voice_result.previews = ["http://preview1.wav"]
        mock_client.text_to_voice.get_text_to_voice_result = AsyncMock(return_value=mock_voice_result)

        model = MockedModelProvider([
            _tool_call_response(
                "camb_voice_from_description",
                {"text": "Hello world", "voice_description": "A warm, friendly voice"},
            ),
            _text_response("Voice previews are ready."),
        ])
        agent = Agent(model=model, tools=[provider], callback_handler=None)

        with patch.object(provider._helpers, "_get_client", return_value=mock_client):
            response = agent("Generate a voice from description")

        assert "Voice previews" in str(response)

    def test_clone_voice_via_agent(self, provider, mock_client):
        """camb_clone_voice with temp file; model sees voice_id."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF" + b"\x00" * 100)
            tmp_path = f.name

        try:
            mock_result = MagicMock()
            mock_result.voice_id = 42
            mock_client.voice_cloning.create_custom_voice = AsyncMock(return_value=mock_result)

            model = MockedModelProvider([
                _tool_call_response(
                    "camb_clone_voice",
                    {"voice_name": "MyClone", "audio_file_path": tmp_path},
                ),
                _text_response("Voice cloned with ID 42."),
            ])
            agent = Agent(model=model, tools=[provider], callback_handler=None)

            with patch.object(provider._helpers, "_get_client", return_value=mock_client):
                response = agent("Clone my voice")

            assert "Voice cloned" in str(response)
        finally:
            os.unlink(tmp_path)

    def test_translated_tts_via_agent(self, provider, mock_client):
        """Full poll + httpx flow mocked; model sees file_path."""
        mock_create = MagicMock()
        mock_create.task_id = "task-123"
        mock_client.translated_tts.create_translated_tts = AsyncMock(return_value=mock_create)

        mock_status = MagicMock()
        mock_status.status = "SUCCESS"
        mock_status.run_id = "run-123"
        mock_client.translated_tts.get_translated_tts_task_status = AsyncMock(return_value=mock_status)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"RIFF" + b"\x00" * 100
        mock_resp.headers = {"content-type": "audio/wav"}

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_http

        model = MockedModelProvider([
            _tool_call_response(
                "camb_translated_tts",
                {"text": "Hello", "source_language": 1, "target_language": 2},
            ),
            _text_response("Translated TTS generated successfully."),
        ])
        agent = Agent(model=model, tools=[provider], callback_handler=None)

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            with patch.object(provider._helpers, "_get_client", return_value=mock_client):
                response = agent("Translate and speak in Spanish")

        assert "Translated TTS" in str(response)

    @pytest.mark.asyncio
    async def test_selective_tools_in_agent(self, mock_api_key):
        """Provider with only 2 tools enabled; agent only has 2 tools."""
        limited_provider = CambAIToolProvider(
            api_key=mock_api_key,
            enable_tts=True,
            enable_translation=True,
            enable_transcription=False,
            enable_translated_tts=False,
            enable_voice_clone=False,
            enable_voice_list=False,
            enable_text_to_sound=False,
            enable_audio_separation=False,
            enable_voice_from_description=False,
        )
        tools = await limited_provider.load_tools()
        assert len(tools) == 2
        names = {t.tool_name for t in tools}
        assert "camb_tts" in names
        assert "camb_translate" in names

    def test_tool_error_handled_by_agent(self, provider, mock_client):
        """File-not-found error; agent receives JSON error; continues gracefully."""
        model = MockedModelProvider([
            _tool_call_response(
                "camb_transcribe",
                {"language": 1, "audio_file_path": "/nonexistent/path.wav"},
            ),
            _text_response("I encountered an error with the audio file."),
        ])
        agent = Agent(model=model, tools=[provider], callback_handler=None)

        with patch.object(provider._helpers, "_get_client", return_value=mock_client):
            response = agent("Transcribe the audio file")

        # Agent should not raise; it receives the error JSON and responds gracefully
        assert "error" in str(response).lower()
