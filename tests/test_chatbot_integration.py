"""All-encompassing chatbot integration test for CAMB.AI tools.

Simulates a 9-turn "Podcast Production Assistant" conversation where all 9
CAMB tools are exercised in sequence by a single Agent instance, maintaining
shared conversation history across turns — just like a real chatbot session.
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
# (identical to test_strands_integration.py)
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


# ---------------------------------------------------------------------------
# TestChatbotIntegration
# ---------------------------------------------------------------------------


class TestChatbotIntegration:
    def test_chatbot_audio_production_assistant(self, mock_api_key):
        """9-turn Podcast Production Assistant chatbot exercising all 9 CAMB tools.

        A single Agent instance maintains conversation history across all turns,
        with each turn invoking a different CAMB tool — simulating a real chatbot
        session end-to-end.
        """
        # ------------------------------------------------------------------
        # Temp files for tools that require a real file path on disk
        # ------------------------------------------------------------------
        audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio_file.write(b"RIFF" + b"\x00" * 100)
        audio_file.close()
        audio_path = audio_file.name

        clone_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        clone_file.write(b"RIFF" + b"\x00" * 100)
        clone_file.close()
        clone_path = clone_file.name

        try:
            # ------------------------------------------------------------------
            # Mock setup — all upfront, shared across all 9 turns
            # ------------------------------------------------------------------
            mock_client = MagicMock()

            # camb module mocks (for tts / translate imports)
            mock_camb = MagicMock()
            mock_api_error_mod = MagicMock()
            mock_api_error_mod.ApiError = Exception

            # httpx mock (for translated_tts)
            mock_resp = MagicMock(
                status_code=200,
                content=b"RIFF" + b"\x00" * 100,
                headers={"content-type": "audio/wav"},
            )
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_resp)
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_httpx = MagicMock()
            mock_httpx.AsyncClient.return_value = mock_http

            # Turn 1 — list_voices
            v1 = MagicMock(id=1, voice_name="Alice")
            v2 = MagicMock(id=2, voice_name="Bob")
            mock_client.voice_cloning.list_voices = AsyncMock(return_value=[v1, v2])

            # Turn 2 — tts (async generator)
            tts_calls: list[dict] = []

            async def mock_tts_gen(*args, **kwargs):
                tts_calls.append(kwargs)
                yield b"RIFF" + b"\x00" * 40

            mock_client.text_to_speech.tts = mock_tts_gen

            # Turn 3 — translate
            mock_tr = MagicMock()
            mock_tr.text = "¡Bienvenidos a nuestro podcast!"
            mock_client.translation.translation_stream = AsyncMock(return_value=mock_tr)

            # Turn 4 — translated_tts
            mc4 = MagicMock(task_id="t4")
            ms4 = MagicMock(status="SUCCESS", run_id="r4")
            mock_client.translated_tts.create_translated_tts = AsyncMock(return_value=mc4)
            mock_client.translated_tts.get_translated_tts_task_status = AsyncMock(return_value=ms4)

            # Turn 5 — transcribe
            mc5 = MagicMock(task_id="t5")
            ms5 = MagicMock(status="SUCCESS", run_id="r5")
            seg = MagicMock(start=0.0, end=1.5, text="Welcome to our podcast!", speaker="SPEAKER_0")
            mt5 = MagicMock(text="Welcome to our podcast!", segments=[seg])
            mock_client.transcription.create_transcription = AsyncMock(return_value=mc5)
            mock_client.transcription.get_transcription_task_status = AsyncMock(return_value=ms5)
            mock_client.transcription.get_transcription_result = AsyncMock(return_value=mt5)

            # Turn 6 — audio_separation
            mc6 = MagicMock(task_id="t6")
            ms6 = MagicMock(status="SUCCESS", run_id="r6")
            msep = MagicMock(
                foreground_audio_url="http://fg.wav",
                background_audio_url="http://bg.wav",
            )
            mock_client.audio_separation.create_audio_separation = AsyncMock(return_value=mc6)
            mock_client.audio_separation.get_audio_separation_status = AsyncMock(return_value=ms6)
            mock_client.audio_separation.get_audio_separation_run_info = AsyncMock(return_value=msep)

            # Turn 7 — text_to_sound
            mc7 = MagicMock(task_id="t7")
            ms7 = MagicMock(status="SUCCESS", run_id="r7")

            async def mock_sound_stream(*args, **kwargs):
                yield b"RIFF" + b"\x00" * 40

            mock_client.text_to_audio.create_text_to_audio = AsyncMock(return_value=mc7)
            mock_client.text_to_audio.get_text_to_audio_status = AsyncMock(return_value=ms7)
            mock_client.text_to_audio.get_text_to_audio_result = mock_sound_stream

            # Turn 8 — clone_voice
            mv8 = MagicMock(voice_id=42)
            mock_client.voice_cloning.create_custom_voice = AsyncMock(return_value=mv8)

            # Turn 9 — voice_from_description
            mc9 = MagicMock(task_id="t9")
            ms9 = MagicMock(status="SUCCESS", run_id="r9")
            mvr9 = MagicMock(previews=["http://preview1.wav"])
            mock_client.text_to_voice.create_text_to_voice = AsyncMock(return_value=mc9)
            mock_client.text_to_voice.get_text_to_voice_status = AsyncMock(return_value=ms9)
            mock_client.text_to_voice.get_text_to_voice_result = AsyncMock(return_value=mvr9)

            # ------------------------------------------------------------------
            # MockedModelProvider — 18 responses (2 per turn: tool call + text)
            # ------------------------------------------------------------------
            model = MockedModelProvider([
                # Turn 1: list voices
                _tool_call_response("camb_list_voices", {}, use_id="use-1"),
                _text_response("Here are 2 voices: Alice (1) and Bob (2). Which would you like to use?"),

                # Turn 2: tts
                _tool_call_response(
                    "camb_tts",
                    {"text": "Welcome to our podcast!", "language": "en-us", "voice_id": 1},
                    use_id="use-2",
                ),
                _text_response("Audio generated and saved to disk. Ready for the next step!"),

                # Turn 3: translate
                _tool_call_response(
                    "camb_translate",
                    {"text": "Welcome to our podcast!", "source_language": 1, "target_language": 2},
                    use_id="use-3",
                ),
                _text_response("Translation: '¡Bienvenidos a nuestro podcast!' Done!"),

                # Turn 4: translated_tts
                _tool_call_response(
                    "camb_translated_tts",
                    {"text": "Welcome to our podcast!", "source_language": 1, "target_language": 2},
                    use_id="use-4",
                ),
                _text_response("Spanish audio created and saved successfully."),

                # Turn 5: transcribe
                _tool_call_response(
                    "camb_transcribe",
                    {"language": 1, "audio_file_path": audio_path},
                    use_id="use-5",
                ),
                _text_response("Transcription complete: 'Welcome to our podcast!'"),

                # Turn 6: audio_separation
                _tool_call_response(
                    "camb_audio_separation",
                    {"audio_file_path": audio_path},
                    use_id="use-6",
                ),
                _text_response("Vocals and background audio have been separated."),

                # Turn 7: text_to_sound
                _tool_call_response(
                    "camb_text_to_sound",
                    {"prompt": "ambient podcast background music"},
                    use_id="use-7",
                ),
                _text_response("Background music generated successfully."),

                # Turn 8: clone_voice
                _tool_call_response(
                    "camb_clone_voice",
                    {"voice_name": "PodcastHost", "audio_file_path": clone_path},
                    use_id="use-8",
                ),
                _text_response("Voice cloned! Your voice ID is 42."),

                # Turn 9: voice_from_description
                _tool_call_response(
                    "camb_voice_from_description",
                    {"text": "Hello welcome!", "voice_description": "professional podcast host"},
                    use_id="use-9",
                ),
                _text_response("Voice preview ready at: http://preview1.wav"),
            ])

            # ------------------------------------------------------------------
            # Agent setup — single instance, shared across all 9 turns
            # ------------------------------------------------------------------
            provider = CambAIToolProvider(api_key=mock_api_key)
            agent = Agent(model=model, tools=[provider], callback_handler=None)

            # ------------------------------------------------------------------
            # 9-turn conversation
            # ------------------------------------------------------------------
            with patch.dict(
                "sys.modules",
                {
                    "camb": mock_camb,
                    "camb.core": MagicMock(),
                    "camb.core.api_error": mock_api_error_mod,
                    "httpx": mock_httpx,
                },
            ):
                with patch.object(provider._helpers, "_get_client", return_value=mock_client):
                    r1 = agent("What voices are available?")
                    r2 = agent("Generate speech saying 'Welcome to our podcast!'")
                    r3 = agent("Translate that greeting to Spanish")
                    r4 = agent("Create a Spanish audio version of that")
                    r5 = agent(f"Transcribe this audio sample: {audio_path}")
                    r6 = agent(f"Separate the vocals from this recording: {audio_path}")
                    r7 = agent("Generate ambient podcast background music")
                    r8 = agent(f"Clone my voice from this sample: {clone_path}")
                    r9 = agent("Create a voice matching: professional podcast host, warm and clear")

            # ------------------------------------------------------------------
            # Assertions — per-response content
            # ------------------------------------------------------------------
            assert "voices" in str(r1).lower() or "Alice" in str(r1)
            assert "generated" in str(r2).lower() or "audio" in str(r2).lower()
            assert "Translation" in str(r3) or "Spanish" in str(r3)
            assert "Spanish audio" in str(r4) or "created" in str(r4).lower()
            assert "Transcription" in str(r5) or "transcript" in str(r5).lower()
            assert "separated" in str(r6).lower() or "vocals" in str(r6).lower()
            assert "music" in str(r7).lower() or "generated" in str(r7).lower()
            assert "cloned" in str(r8).lower() or "42" in str(r8)
            assert "preview" in str(r9).lower() or "voice" in str(r9).lower()

            # ------------------------------------------------------------------
            # Assertions — each tool was invoked exactly once
            # ------------------------------------------------------------------
            mock_client.voice_cloning.list_voices.assert_called_once()
            mock_client.translation.translation_stream.assert_called_once()
            mock_client.translated_tts.create_translated_tts.assert_called_once()
            mock_client.transcription.create_transcription.assert_called_once()
            mock_client.audio_separation.create_audio_separation.assert_called_once()
            mock_client.text_to_audio.create_text_to_audio.assert_called_once()
            mock_client.voice_cloning.create_custom_voice.assert_called_once()
            mock_client.text_to_voice.create_text_to_voice.assert_called_once()
            assert len(tts_calls) == 1  # TTS uses async generator, not AsyncMock

            # ------------------------------------------------------------------
            # Assertions — conversation history accumulated across all turns
            # ------------------------------------------------------------------
            assert len(agent.messages) >= 18  # 9 user + at least 9 assistant messages

        finally:
            os.unlink(audio_path)
            os.unlink(clone_path)
