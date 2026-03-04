"""Tests for CAMB.AI Strands tools."""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strands_camb._helpers import CambHelpers
from strands_camb.camb_tools import CambAIToolProvider


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_api_key():
    return "test-api-key-12345"


@pytest.fixture
def helpers(mock_api_key):
    return CambHelpers(api_key=mock_api_key)


@pytest.fixture
def provider(mock_api_key):
    return CambAIToolProvider(api_key=mock_api_key)


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def make_poll_status():
    def _make(status="SUCCESS", run_id="run-123"):
        s = MagicMock()
        s.status = status
        s.run_id = run_id
        return s

    return _make


@pytest.fixture
def mock_httpx_ctx():
    """Returns (mock_httpx_module, mock_resp) with WAV bytes pre-loaded."""
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
    return mock_httpx, mock_resp


# ---------------------------------------------------------------------------
# TestCambHelpers
# ---------------------------------------------------------------------------


class TestCambHelpers:
    def test_init_with_api_key(self, mock_api_key):
        h = CambHelpers(api_key=mock_api_key)
        assert h._api_key == mock_api_key

    def test_init_without_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="CAMB_API_KEY"):
                CambHelpers(api_key=None)

    def test_init_from_env(self):
        with patch.dict("os.environ", {"CAMB_API_KEY": "env-key"}):
            h = CambHelpers()
            assert h._api_key == "env-key"

    def test_detect_audio_format_wav(self, helpers):
        assert helpers._detect_audio_format(b"RIFF" + b"\x00" * 100) == "wav"

    def test_detect_audio_format_mp3(self, helpers):
        assert helpers._detect_audio_format(b"\xff\xfb" + b"\x00" * 100) == "mp3"

    def test_detect_audio_format_mp3_id3(self, helpers):
        assert helpers._detect_audio_format(b"ID3" + b"\x00" * 100) == "mp3"

    def test_detect_audio_format_flac(self, helpers):
        assert helpers._detect_audio_format(b"fLaC" + b"\x00" * 100) == "flac"

    def test_detect_audio_format_ogg(self, helpers):
        assert helpers._detect_audio_format(b"OggS" + b"\x00" * 100) == "ogg"

    def test_detect_audio_format_pcm(self, helpers):
        assert helpers._detect_audio_format(b"\x00" * 100) == "pcm"

    def test_detect_audio_format_content_type(self, helpers):
        assert helpers._detect_audio_format(b"\x00", "audio/mpeg") == "mp3"
        assert helpers._detect_audio_format(b"\x00", "audio/wav") == "wav"

    def test_add_wav_header(self, helpers):
        pcm = b"\x00\x01" * 100
        wav = helpers._add_wav_header(pcm)
        assert wav.startswith(b"RIFF")
        assert b"WAVE" in wav[:12]
        assert wav.endswith(pcm)

    def test_save_audio(self, helpers, tmp_path):
        data = b"fake audio data"
        path = helpers._save_audio(data, ".wav")
        assert path.endswith(".wav")
        with open(path, "rb") as f:
            assert f.read() == data

    def test_extract_translation_string(self, helpers):
        assert helpers._extract_translation("hello") == "hello"

    def test_extract_translation_object(self, helpers):
        """MagicMock with .text should return .text, not enter iterable branch."""
        obj = MagicMock()
        obj.text = "translated"
        assert helpers._extract_translation(obj) == "translated"

    def test_extract_translation_iterable(self, helpers):
        """Iterable of chunk objects with .text should be joined."""
        chunk1 = MagicMock(spec=[])
        chunk1.text = "hello "
        chunk2 = MagicMock(spec=[])
        chunk2.text = "world"
        assert helpers._extract_translation([chunk1, chunk2]) == "hello world"

    def test_extract_translation_plain_object(self, helpers):
        """Object without .text or __iter__ falls through to str()."""
        obj = object()
        result = helpers._extract_translation(obj)
        assert isinstance(result, str)

    def test_format_transcription_with_segments(self, helpers):
        """Handle SDK response with .segments attribute."""
        seg = MagicMock()
        seg.start = 0.0
        seg.end = 1.5
        seg.text = "hello"
        seg.speaker = "SPEAKER_0"
        transcription = MagicMock()
        transcription.text = "hello"
        transcription.segments = [seg]
        # Remove .transcript so .segments is used
        del transcription.transcript
        result = json.loads(helpers._format_transcription(transcription))
        assert result["text"] == "hello"
        assert len(result["segments"]) == 1
        assert result["segments"][0]["speaker"] == "SPEAKER_0"
        assert "SPEAKER_0" in result["speakers"]

    def test_format_transcription_speakers_list(self, helpers):
        """Speakers list should contain unique speaker IDs."""
        seg1 = MagicMock()
        seg1.start, seg1.end, seg1.text, seg1.speaker = 0.0, 1.0, "hi", "SPEAKER_0"
        seg2 = MagicMock()
        seg2.start, seg2.end, seg2.text, seg2.speaker = 1.0, 2.0, "hello", "SPEAKER_1"
        seg3 = MagicMock()
        seg3.start, seg3.end, seg3.text, seg3.speaker = 2.0, 3.0, "bye", "SPEAKER_0"
        transcription = MagicMock()
        transcription.text = "hi hello bye"
        transcription.segments = [seg1, seg2, seg3]
        del transcription.transcript
        result = json.loads(helpers._format_transcription(transcription))
        assert sorted(result["speakers"]) == ["SPEAKER_0", "SPEAKER_1"]

    def test_format_transcription_with_transcript(self, helpers):
        """Handle SDK response with .transcript attribute (fallback)."""
        seg = MagicMock()
        seg.start = 0.0
        seg.end = 1.5
        seg.text = "hello"
        seg.speaker = "SPEAKER_0"
        transcription = MagicMock()
        transcription.text = "hello"
        transcription.transcript = [seg]
        # Remove .segments so .transcript fallback is used
        del transcription.segments
        result = json.loads(helpers._format_transcription(transcription))
        assert result["text"] == "hello"
        assert len(result["segments"]) == 1

    def test_format_voices(self, helpers):
        voices = [{"id": 1, "voice_name": "Alice"}, {"id": 2, "voice_name": "Bob"}]
        result = json.loads(helpers._format_voices(voices))
        assert len(result) == 2
        assert result[0]["id"] == 1
        assert result[0]["voice_name"] == "Alice"

    def test_format_voices_objects(self, helpers):
        v = MagicMock()
        v.id = 42
        v.voice_name = "TestVoice"
        result = json.loads(helpers._format_voices([v]))
        assert result[0]["id"] == 42
        assert result[0]["voice_name"] == "TestVoice"

    def test_format_separation(self, helpers):
        sep = MagicMock()
        sep.foreground_audio_url = "http://fg.wav"
        sep.background_audio_url = "http://bg.wav"
        result = json.loads(helpers._format_separation(sep))
        assert result["foreground_audio_url"] == "http://fg.wav"
        assert result["background_audio_url"] == "http://bg.wav"


# ---------------------------------------------------------------------------
# TestCambAIToolProvider
# ---------------------------------------------------------------------------


class TestCambAIToolProvider:
    def test_init(self, provider):
        assert provider._helpers._api_key == "test-api-key-12345"

    @pytest.mark.asyncio
    async def test_load_tools_all_enabled(self, provider):
        tools = await provider.load_tools()
        assert len(tools) == 9

    @pytest.mark.asyncio
    async def test_load_tools_selective(self, mock_api_key):
        provider = CambAIToolProvider(
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
        tools = await provider.load_tools()
        assert len(tools) == 2

    def test_add_remove_consumer(self, provider):
        provider.add_consumer("agent-1")
        assert "agent-1" in provider._consumers
        provider.remove_consumer("agent-1")
        assert "agent-1" not in provider._consumers

    def test_remove_consumer_idempotent(self, provider):
        provider.remove_consumer("nonexistent")  # Should not raise

    @pytest.mark.asyncio
    async def test_tts_tool(self, provider):
        tools = await provider.load_tools()
        tts_tool = tools[0]
        assert tts_tool.tool_name == "camb_tts"

    @pytest.mark.asyncio
    async def test_translate_tool(self, provider):
        tools = await provider.load_tools()
        translate_tool = tools[1]
        assert translate_tool.tool_name == "camb_translate"

    @pytest.mark.asyncio
    async def test_list_voices_tool(self, provider):
        tools = await provider.load_tools()
        voice_list = [t for t in tools if t.tool_name == "camb_list_voices"]
        assert len(voice_list) == 1

    @pytest.mark.asyncio
    async def test_voice_from_description_tool(self, provider):
        tools = await provider.load_tools()
        vfd = [t for t in tools if t.tool_name == "camb_voice_from_description"]
        assert len(vfd) == 1

    @pytest.mark.asyncio
    async def test_all_tool_names(self, provider):
        tools = await provider.load_tools()
        names = {t.tool_name for t in tools}
        expected = {
            "camb_tts",
            "camb_translate",
            "camb_transcribe",
            "camb_translated_tts",
            "camb_clone_voice",
            "camb_list_voices",
            "camb_text_to_sound",
            "camb_audio_separation",
            "camb_voice_from_description",
        }
        assert names == expected


# ---------------------------------------------------------------------------
# TestCambHelpersPollAsync
# ---------------------------------------------------------------------------


class TestCambHelpersPollAsync:
    @pytest.mark.asyncio
    async def test_poll_success(self, helpers):
        status_result = MagicMock()
        status_result.status = "SUCCESS"
        status_result.run_id = "run-123"
        get_status = AsyncMock(return_value=status_result)

        result = await helpers._poll_async(get_status, "task-1")
        assert result.run_id == "run-123"

    @pytest.mark.asyncio
    async def test_poll_failure(self, helpers):
        helpers._max_poll_attempts = 1
        status_result = MagicMock()
        status_result.status = "FAILED"
        status_result.error = "Something went wrong"
        get_status = AsyncMock(return_value=status_result)

        with pytest.raises(RuntimeError, match="CAMB.AI task failed"):
            await helpers._poll_async(get_status, "task-1")

    @pytest.mark.asyncio
    async def test_poll_timeout(self, helpers):
        helpers._max_poll_attempts = 2
        helpers._poll_interval = 0.01
        status_result = MagicMock()
        status_result.status = "PENDING"
        get_status = AsyncMock(return_value=status_result)

        with pytest.raises(TimeoutError, match="did not complete"):
            await helpers._poll_async(get_status, "task-1")

    @pytest.mark.asyncio
    async def test_poll_transient_error_then_success(self, helpers):
        """Transient API errors should be retried, not propagated."""
        helpers._max_poll_attempts = 3
        helpers._poll_interval = 0.01
        success_status = MagicMock()
        success_status.status = "SUCCESS"
        success_status.run_id = "run-ok"
        get_status = AsyncMock(side_effect=[ConnectionError("network"), success_status])

        result = await helpers._poll_async(get_status, "task-1")
        assert result.run_id == "run-ok"
        assert get_status.call_count == 2

    @pytest.mark.asyncio
    async def test_poll_all_transient_errors_timeout(self, helpers):
        """If every poll attempt fails with transient errors, should timeout."""
        helpers._max_poll_attempts = 2
        helpers._poll_interval = 0.01
        get_status = AsyncMock(side_effect=ConnectionError("network"))

        with pytest.raises(TimeoutError, match="did not complete"):
            await helpers._poll_async(get_status, "task-1")


# ---------------------------------------------------------------------------
# TestCambTts
# ---------------------------------------------------------------------------


class TestCambTts:
    @pytest.mark.asyncio
    async def test_tts_happy_path(self, provider, mock_client):
        """Async generator yields 2 WAV chunks → success with file_path."""
        tools = await provider.load_tools()
        tts = [t for t in tools if t.tool_name == "camb_tts"][0]

        async def mock_tts_gen(*args, **kwargs):
            yield b"RIFF" + b"\x00" * 40
            yield b"\x00" * 40

        mock_client.text_to_speech.tts = mock_tts_gen
        mock_camb = MagicMock()

        with patch.dict("sys.modules", {"camb": mock_camb}):
            with patch.object(provider._helpers, "_get_client", return_value=mock_client):
                result = await tts(text="Hello world")

        body = json.loads(result)
        assert body["status"] == "success"
        assert "file_path" in body
        assert body["file_path"].endswith(".wav")

    @pytest.mark.asyncio
    async def test_tts_empty_audio(self, provider, mock_client):
        """Generator yields nothing → error: TTS returned no audio data."""
        tools = await provider.load_tools()
        tts = [t for t in tools if t.tool_name == "camb_tts"][0]

        async def mock_tts_empty(*args, **kwargs):
            if False:
                yield b""

        mock_client.text_to_speech.tts = mock_tts_empty
        mock_camb = MagicMock()

        with patch.dict("sys.modules", {"camb": mock_camb}):
            with patch.object(provider._helpers, "_get_client", return_value=mock_client):
                result = await tts(text="Hello world")

        body = json.loads(result)
        assert "error" in body
        assert "no audio data" in body["error"]

    @pytest.mark.asyncio
    async def test_tts_with_speed(self, provider, mock_client):
        """speed parameter → StreamTtsVoiceSettings included in kwargs."""
        tools = await provider.load_tools()
        tts = [t for t in tools if t.tool_name == "camb_tts"][0]

        captured_kwargs: dict = {}

        async def mock_tts_gen(*args, **kwargs):
            captured_kwargs.update(kwargs)
            yield b"RIFF" + b"\x00" * 40

        mock_client.text_to_speech.tts = mock_tts_gen
        mock_camb = MagicMock()

        with patch.dict("sys.modules", {"camb": mock_camb}):
            with patch.object(provider._helpers, "_get_client", return_value=mock_client):
                await tts(text="Hello world", speed=1.5)

        assert "voice_settings" in captured_kwargs

    @pytest.mark.asyncio
    async def test_tts_user_instructions_mars_instruct(self, provider, mock_client):
        """mars-instruct model → user_instructions kwarg forwarded."""
        tools = await provider.load_tools()
        tts = [t for t in tools if t.tool_name == "camb_tts"][0]

        captured_kwargs: dict = {}

        async def mock_tts_gen(*args, **kwargs):
            captured_kwargs.update(kwargs)
            yield b"RIFF" + b"\x00" * 40

        mock_client.text_to_speech.tts = mock_tts_gen
        mock_camb = MagicMock()

        with patch.dict("sys.modules", {"camb": mock_camb}):
            with patch.object(provider._helpers, "_get_client", return_value=mock_client):
                await tts(
                    text="Hello world",
                    speech_model="mars-instruct",
                    user_instructions="Speak slowly and clearly",
                )

        assert "user_instructions" in captured_kwargs
        assert captured_kwargs["user_instructions"] == "Speak slowly and clearly"

    @pytest.mark.asyncio
    async def test_tts_user_instructions_ignored_non_instruct(self, provider, mock_client):
        """Non-instruct model → user_instructions kwarg NOT forwarded."""
        tools = await provider.load_tools()
        tts = [t for t in tools if t.tool_name == "camb_tts"][0]

        captured_kwargs: dict = {}

        async def mock_tts_gen(*args, **kwargs):
            captured_kwargs.update(kwargs)
            yield b"RIFF" + b"\x00" * 40

        mock_client.text_to_speech.tts = mock_tts_gen
        mock_camb = MagicMock()

        with patch.dict("sys.modules", {"camb": mock_camb}):
            with patch.object(provider._helpers, "_get_client", return_value=mock_client):
                await tts(
                    text="Hello world",
                    speech_model="mars-flash",
                    user_instructions="Speak slowly",
                )

        assert "user_instructions" not in captured_kwargs


# ---------------------------------------------------------------------------
# TestCambTranslate
# ---------------------------------------------------------------------------


class TestCambTranslate:
    @pytest.mark.asyncio
    async def test_translate_happy_path(self, provider, mock_client):
        """translation_stream returns object with .text → success."""
        tools = await provider.load_tools()
        translate = [t for t in tools if t.tool_name == "camb_translate"][0]

        mock_result = MagicMock()
        mock_result.text = "Hola mundo"
        mock_client.translation.translation_stream = AsyncMock(return_value=mock_result)

        mock_api_error_mod = MagicMock()
        mock_api_error_mod.ApiError = Exception  # Won't be raised in happy path

        with patch.dict("sys.modules", {"camb.core": MagicMock(), "camb.core.api_error": mock_api_error_mod}):
            with patch.object(provider._helpers, "_get_client", return_value=mock_client):
                result = await translate(text="Hello world", source_language=1, target_language=2)

        body = json.loads(result)
        assert body["status"] == "success"
        assert body["translated_text"] == "Hola mundo"

    @pytest.mark.asyncio
    async def test_translate_with_formality(self, provider, mock_client):
        """formality kwarg passes through to translation_stream."""
        tools = await provider.load_tools()
        translate = [t for t in tools if t.tool_name == "camb_translate"][0]

        mock_result = MagicMock()
        mock_result.text = "Bonjour"
        mock_client.translation.translation_stream = AsyncMock(return_value=mock_result)

        mock_api_error_mod = MagicMock()
        mock_api_error_mod.ApiError = Exception

        with patch.dict("sys.modules", {"camb.core": MagicMock(), "camb.core.api_error": mock_api_error_mod}):
            with patch.object(provider._helpers, "_get_client", return_value=mock_client):
                await translate(text="Hello", source_language=1, target_language=3, formality=1)

        call_kwargs = mock_client.translation.translation_stream.call_args.kwargs
        assert call_kwargs.get("formality") == 1

    @pytest.mark.asyncio
    async def test_translate_api_error_200(self, provider, mock_client):
        """ApiError(status_code=200, body='txt') treated as success."""
        tools = await provider.load_tools()
        translate = [t for t in tools if t.tool_name == "camb_translate"][0]

        class FakeApiError(Exception):
            def __init__(self, status_code, body):
                self.status_code = status_code
                self.body = body

        mock_client.translation.translation_stream = AsyncMock(
            side_effect=FakeApiError(200, "Hola mundo")
        )
        mock_api_error_mod = MagicMock()
        mock_api_error_mod.ApiError = FakeApiError

        with patch.dict("sys.modules", {"camb.core": MagicMock(), "camb.core.api_error": mock_api_error_mod}):
            with patch.object(provider._helpers, "_get_client", return_value=mock_client):
                result = await translate(text="Hello world", source_language=1, target_language=2)

        body = json.loads(result)
        assert body["status"] == "success"
        assert body["translated_text"] == "Hola mundo"

    @pytest.mark.asyncio
    async def test_translate_api_error_non_200(self, provider, mock_client):
        """ApiError with status_code != 200 propagates."""
        tools = await provider.load_tools()
        translate = [t for t in tools if t.tool_name == "camb_translate"][0]

        class FakeApiError(Exception):
            def __init__(self, status_code, body):
                self.status_code = status_code
                self.body = body

        mock_client.translation.translation_stream = AsyncMock(
            side_effect=FakeApiError(500, "Server error")
        )
        mock_api_error_mod = MagicMock()
        mock_api_error_mod.ApiError = FakeApiError

        with patch.dict("sys.modules", {"camb.core": MagicMock(), "camb.core.api_error": mock_api_error_mod}):
            with patch.object(provider._helpers, "_get_client", return_value=mock_client):
                with pytest.raises(FakeApiError):
                    await translate(text="Hello world", source_language=1, target_language=2)


# ---------------------------------------------------------------------------
# TestCambTranscribe  (absorbs TestTranscribeErrorHandling)
# ---------------------------------------------------------------------------


class TestCambTranscribe:
    @pytest.mark.asyncio
    async def test_transcribe_file_happy_path(self, provider, mock_client, make_poll_status):
        """Real temp file → create/poll/get → text in response."""
        tools = await provider.load_tools()
        transcribe = [t for t in tools if t.tool_name == "camb_transcribe"][0]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF" + b"\x00" * 100)
            tmp_path = f.name

        try:
            mock_create = MagicMock()
            mock_create.task_id = "task-123"
            mock_client.transcription.create_transcription = AsyncMock(return_value=mock_create)

            mock_status = make_poll_status()
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

            with patch.object(provider._helpers, "_get_client", return_value=mock_client):
                result = await transcribe(language=1, audio_file_path=tmp_path)

            body = json.loads(result)
            assert "text" in body
            assert body["text"] == "Hello world"
            assert len(body["segments"]) == 1
        finally:
            os.unlink(tmp_path)

    @pytest.mark.asyncio
    async def test_transcribe_url_happy_path(self, provider, mock_client, mock_httpx_ctx, make_poll_status):
        """httpx download + create/poll/get → text in response."""
        tools = await provider.load_tools()
        transcribe = [t for t in tools if t.tool_name == "camb_transcribe"][0]

        mock_httpx, mock_resp = mock_httpx_ctx
        mock_resp.content = b"RIFF" + b"\x00" * 100

        mock_create = MagicMock()
        mock_create.task_id = "task-123"
        mock_client.transcription.create_transcription = AsyncMock(return_value=mock_create)

        mock_status = make_poll_status()
        mock_client.transcription.get_transcription_task_status = AsyncMock(return_value=mock_status)

        seg = MagicMock()
        seg.start = 0.0
        seg.end = 1.0
        seg.text = "Hello"
        seg.speaker = "SPEAKER_0"
        mock_transcription = MagicMock()
        mock_transcription.text = "Hello"
        mock_transcription.segments = [seg]
        mock_client.transcription.get_transcription_result = AsyncMock(return_value=mock_transcription)

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            with patch.object(provider._helpers, "_get_client", return_value=mock_client):
                result = await transcribe(language=1, audio_url="https://example.com/audio.wav")

        body = json.loads(result)
        assert "text" in body
        assert body["text"] == "Hello"

    @pytest.mark.asyncio
    async def test_transcribe_no_source(self, provider, mock_client):
        """No url or path → error: Provide either."""
        tools = await provider.load_tools()
        transcribe = [t for t in tools if t.tool_name == "camb_transcribe"][0]

        with patch.object(provider._helpers, "_get_client", return_value=mock_client):
            result = await transcribe(language=1)

        body = json.loads(result)
        assert "error" in body
        assert "Provide either" in body["error"]

    @pytest.mark.asyncio
    async def test_transcribe_httpx_not_installed(self, provider, mock_client):
        """httpx not available → error mentioning httpx."""
        tools = await provider.load_tools()
        transcribe = [t for t in tools if t.tool_name == "camb_transcribe"][0]

        with patch.dict("sys.modules", {"httpx": None}):
            with patch.object(provider._helpers, "_get_client", return_value=mock_client):
                result = await transcribe(language=1, audio_url="https://example.com/audio.wav")

        body = json.loads(result)
        assert "error" in body
        assert "httpx" in body["error"]

    @pytest.mark.asyncio
    async def test_transcribe_file_not_found(self, provider):
        tools = await provider.load_tools()
        transcribe = [t for t in tools if t.tool_name == "camb_transcribe"][0]
        mock_client = MagicMock()
        with patch.object(provider._helpers, "_get_client", return_value=mock_client):
            result = await transcribe(language=1, audio_file_path="/nonexistent/file.wav")
        body = json.loads(result)
        assert "error" in body
        assert "/nonexistent/file.wav" in body["error"]

    @pytest.mark.asyncio
    async def test_transcribe_url_download_failure(self, provider):
        tools = await provider.load_tools()
        transcribe = [t for t in tools if t.tool_name == "camb_transcribe"][0]

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("404 Not Found")

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_http

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            with patch.object(provider._helpers, "_get_client", return_value=mock_client):
                result = await transcribe(language=1, audio_url="https://example.com/audio.wav")
        body = json.loads(result)
        assert "error" in body
        assert "Failed to download audio" in body["error"]


# ---------------------------------------------------------------------------
# TestCambTranslatedTts  (absorbs TestTranslatedTtsErrorHandling)
# ---------------------------------------------------------------------------


class TestCambTranslatedTts:
    @pytest.mark.asyncio
    async def test_translated_tts_happy_path(self, provider, mock_client, mock_httpx_ctx, make_poll_status):
        """Full mock: create→poll→httpx WAV → success with file_path."""
        tools = await provider.load_tools()
        tts = [t for t in tools if t.tool_name == "camb_translated_tts"][0]

        mock_httpx, mock_resp = mock_httpx_ctx

        mock_create = MagicMock()
        mock_create.task_id = "task-123"
        mock_client.translated_tts.create_translated_tts = AsyncMock(return_value=mock_create)

        mock_status = make_poll_status()
        mock_client.translated_tts.get_translated_tts_task_status = AsyncMock(return_value=mock_status)

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            with patch.object(provider._helpers, "_get_client", return_value=mock_client):
                result = await tts(text="Hello", source_language=1, target_language=2)

        body = json.loads(result)
        assert body["status"] == "success"
        assert "file_path" in body
        assert body["file_path"].endswith(".wav")

    @pytest.mark.asyncio
    async def test_translated_tts_pcm_gets_wav_header(self, provider, mock_client, make_poll_status):
        """PCM bytes returned → saved as .wav file with RIFF header."""
        tools = await provider.load_tools()
        tts = [t for t in tools if t.tool_name == "camb_translated_tts"][0]

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"\x00" * 200  # PCM data (no RIFF header)
        mock_resp.headers = {"content-type": "audio/pcm"}

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_http

        mock_create = MagicMock()
        mock_create.task_id = "task-123"
        mock_client.translated_tts.create_translated_tts = AsyncMock(return_value=mock_create)

        mock_status = make_poll_status()
        mock_client.translated_tts.get_translated_tts_task_status = AsyncMock(return_value=mock_status)

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            with patch.object(provider._helpers, "_get_client", return_value=mock_client):
                result = await tts(text="Hello", source_language=1, target_language=2)

        body = json.loads(result)
        assert body["status"] == "success"
        assert body["file_path"].endswith(".wav")
        with open(body["file_path"], "rb") as f:
            assert f.read(4) == b"RIFF"

    @pytest.mark.asyncio
    async def test_translated_tts_httpx_not_installed(self, provider, mock_client):
        """httpx not available → error mentioning httpx."""
        tools = await provider.load_tools()
        tts = [t for t in tools if t.tool_name == "camb_translated_tts"][0]

        with patch.dict("sys.modules", {"httpx": None}):
            with patch.object(provider._helpers, "_get_client", return_value=mock_client):
                result = await tts(text="Hello", source_language=1, target_language=2)

        body = json.loads(result)
        assert "error" in body
        assert "httpx" in body["error"]

    @pytest.mark.asyncio
    async def test_translated_tts_no_run_id(self, provider):
        """Should return error when run_id is None."""
        tools = await provider.load_tools()
        tts = [t for t in tools if t.tool_name == "camb_translated_tts"][0]

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.task_id = "task-1"
        mock_client.translated_tts.create_translated_tts = AsyncMock(return_value=mock_result)

        mock_status = MagicMock()
        mock_status.status = "SUCCESS"
        mock_status.run_id = None  # No run_id
        mock_client.translated_tts.get_translated_tts_task_status = AsyncMock(return_value=mock_status)

        mock_httpx = MagicMock()

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            with patch.object(provider._helpers, "_get_client", return_value=mock_client):
                result = await tts(
                    text="hello",
                    source_language=1,
                    target_language=2,
                )
        body = json.loads(result)
        assert "error" in body
        assert "no run_id" in body["error"]

    @pytest.mark.asyncio
    async def test_translated_tts_empty_audio(self, provider):
        """Should return error when audio data is empty."""
        tools = await provider.load_tools()
        tts = [t for t in tools if t.tool_name == "camb_translated_tts"][0]

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.task_id = "task-1"
        mock_client.translated_tts.create_translated_tts = AsyncMock(return_value=mock_result)

        mock_status = MagicMock()
        mock_status.status = "SUCCESS"
        mock_status.run_id = "run-123"
        mock_client.translated_tts.get_translated_tts_task_status = AsyncMock(return_value=mock_status)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b""  # Empty audio
        mock_resp.headers = {"content-type": "audio/wav"}

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_http

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            with patch.object(provider._helpers, "_get_client", return_value=mock_client):
                result = await tts(
                    text="hello",
                    source_language=1,
                    target_language=2,
                )
        body = json.loads(result)
        assert "error" in body
        assert "no audio data" in body["error"]


# ---------------------------------------------------------------------------
# TestCambCloneVoice  (absorbs TestCloneVoiceErrorHandling)
# ---------------------------------------------------------------------------


class TestCambCloneVoice:
    @pytest.mark.asyncio
    async def test_clone_voice_happy_path(self, provider, mock_client):
        """Real temp file + mock API → {"status":"created","voice_id":42}."""
        tools = await provider.load_tools()
        clone = [t for t in tools if t.tool_name == "camb_clone_voice"][0]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF" + b"\x00" * 100)
            tmp_path = f.name

        try:
            mock_result = MagicMock()
            mock_result.voice_id = 42
            mock_client.voice_cloning.create_custom_voice = AsyncMock(return_value=mock_result)

            with patch.object(provider._helpers, "_get_client", return_value=mock_client):
                result = await clone(voice_name="MyVoice", audio_file_path=tmp_path)

            body = json.loads(result)
            assert body["status"] == "created"
            assert body["voice_id"] == 42
            assert body["voice_name"] == "MyVoice"
        finally:
            os.unlink(tmp_path)

    @pytest.mark.asyncio
    async def test_clone_voice_optional_params(self, provider, mock_client):
        """description/age/language pass through to API call."""
        tools = await provider.load_tools()
        clone = [t for t in tools if t.tool_name == "camb_clone_voice"][0]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF" + b"\x00" * 100)
            tmp_path = f.name

        try:
            mock_result = MagicMock()
            mock_result.voice_id = 99
            mock_client.voice_cloning.create_custom_voice = AsyncMock(return_value=mock_result)

            with patch.object(provider._helpers, "_get_client", return_value=mock_client):
                await clone(
                    voice_name="MyVoice",
                    audio_file_path=tmp_path,
                    description="A warm voice",
                    age=30,
                    language=1,
                )

            call_kwargs = mock_client.voice_cloning.create_custom_voice.call_args.kwargs
            assert call_kwargs.get("description") == "A warm voice"
            assert call_kwargs.get("age") == 30
            assert call_kwargs.get("language") == 1
        finally:
            os.unlink(tmp_path)

    @pytest.mark.asyncio
    async def test_clone_voice_file_not_found(self, provider):
        tools = await provider.load_tools()
        clone = [t for t in tools if t.tool_name == "camb_clone_voice"][0]
        mock_client = MagicMock()
        with patch.object(provider._helpers, "_get_client", return_value=mock_client):
            result = await clone(voice_name="test", audio_file_path="/nonexistent/voice.wav")
        body = json.loads(result)
        assert "error" in body
        assert "/nonexistent/voice.wav" in body["error"]


# ---------------------------------------------------------------------------
# TestCambListVoices
# ---------------------------------------------------------------------------


class TestCambListVoices:
    @pytest.mark.asyncio
    async def test_list_voices_happy_path(self, provider, mock_client):
        """2 voice objects → JSON with id and voice_name."""
        tools = await provider.load_tools()
        list_voices = [t for t in tools if t.tool_name == "camb_list_voices"][0]

        v1 = MagicMock()
        v1.id = 1
        v1.voice_name = "Alice"
        v2 = MagicMock()
        v2.id = 2
        v2.voice_name = "Bob"
        mock_client.voice_cloning.list_voices = AsyncMock(return_value=[v1, v2])

        with patch.object(provider._helpers, "_get_client", return_value=mock_client):
            result = await list_voices()

        body = json.loads(result)
        assert len(body) == 2
        assert body[0]["id"] == 1
        assert body[0]["voice_name"] == "Alice"
        assert body[1]["id"] == 2
        assert body[1]["voice_name"] == "Bob"

    @pytest.mark.asyncio
    async def test_list_voices_empty(self, provider, mock_client):
        """Empty list → []."""
        tools = await provider.load_tools()
        list_voices = [t for t in tools if t.tool_name == "camb_list_voices"][0]

        mock_client.voice_cloning.list_voices = AsyncMock(return_value=[])

        with patch.object(provider._helpers, "_get_client", return_value=mock_client):
            result = await list_voices()

        body = json.loads(result)
        assert body == []


# ---------------------------------------------------------------------------
# TestCambTextToSound
# ---------------------------------------------------------------------------


class TestCambTextToSound:
    @pytest.mark.asyncio
    async def test_text_to_sound_happy_path(self, provider, mock_client, make_poll_status):
        """create/poll/stream 2 chunks → success with file_path."""
        tools = await provider.load_tools()
        sound = [t for t in tools if t.tool_name == "camb_text_to_sound"][0]

        mock_create = MagicMock()
        mock_create.task_id = "task-123"
        mock_client.text_to_audio.create_text_to_audio = AsyncMock(return_value=mock_create)

        mock_status = make_poll_status()
        mock_client.text_to_audio.get_text_to_audio_status = AsyncMock(return_value=mock_status)

        async def mock_audio_stream(*args, **kwargs):
            yield b"RIFF" + b"\x00" * 40
            yield b"\x00" * 40

        mock_client.text_to_audio.get_text_to_audio_result = mock_audio_stream

        with patch.object(provider._helpers, "_get_client", return_value=mock_client):
            result = await sound(prompt="Gentle rain on a rooftop")

        body = json.loads(result)
        assert body["status"] == "success"
        assert "file_path" in body

    @pytest.mark.asyncio
    async def test_text_to_sound_optional_params(self, provider, mock_client, make_poll_status):
        """duration/audio_type pass through to create_text_to_audio."""
        tools = await provider.load_tools()
        sound = [t for t in tools if t.tool_name == "camb_text_to_sound"][0]

        mock_create = MagicMock()
        mock_create.task_id = "task-123"
        mock_client.text_to_audio.create_text_to_audio = AsyncMock(return_value=mock_create)

        mock_status = make_poll_status()
        mock_client.text_to_audio.get_text_to_audio_status = AsyncMock(return_value=mock_status)

        async def mock_audio_stream(*args, **kwargs):
            yield b"RIFF" + b"\x00" * 40

        mock_client.text_to_audio.get_text_to_audio_result = mock_audio_stream

        with patch.object(provider._helpers, "_get_client", return_value=mock_client):
            await sound(prompt="Rain", duration=5.0, audio_type="sound")

        call_kwargs = mock_client.text_to_audio.create_text_to_audio.call_args.kwargs
        assert call_kwargs.get("duration") == 5.0
        assert call_kwargs.get("audio_type") == "sound"

    @pytest.mark.asyncio
    async def test_text_to_sound_empty_audio(self, provider, mock_client, make_poll_status):
        """Empty stream → error: no audio data."""
        tools = await provider.load_tools()
        sound = [t for t in tools if t.tool_name == "camb_text_to_sound"][0]

        mock_create = MagicMock()
        mock_create.task_id = "task-123"
        mock_client.text_to_audio.create_text_to_audio = AsyncMock(return_value=mock_create)

        mock_status = make_poll_status()
        mock_client.text_to_audio.get_text_to_audio_status = AsyncMock(return_value=mock_status)

        async def empty_stream(*args, **kwargs):
            if False:
                yield b""

        mock_client.text_to_audio.get_text_to_audio_result = empty_stream

        with patch.object(provider._helpers, "_get_client", return_value=mock_client):
            result = await sound(prompt="Rain")

        body = json.loads(result)
        assert "error" in body
        assert "no audio data" in body["error"]


# ---------------------------------------------------------------------------
# TestCambAudioSeparation  (absorbs TestAudioSeparationErrorHandling)
# ---------------------------------------------------------------------------


class TestCambAudioSeparation:
    @pytest.mark.asyncio
    async def test_audio_separation_file_happy_path(self, provider, mock_client, make_poll_status):
        """Temp file + mock create/poll/get_run_info → fg/bg URLs."""
        tools = await provider.load_tools()
        sep = [t for t in tools if t.tool_name == "camb_audio_separation"][0]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF" + b"\x00" * 100)
            tmp_path = f.name

        try:
            mock_create = MagicMock()
            mock_create.task_id = "task-123"
            mock_client.audio_separation.create_audio_separation = AsyncMock(return_value=mock_create)

            mock_status = make_poll_status()
            mock_client.audio_separation.get_audio_separation_status = AsyncMock(return_value=mock_status)

            mock_sep = MagicMock()
            mock_sep.foreground_audio_url = "http://fg.wav"
            mock_sep.background_audio_url = "http://bg.wav"
            mock_client.audio_separation.get_audio_separation_run_info = AsyncMock(return_value=mock_sep)

            with patch.object(provider._helpers, "_get_client", return_value=mock_client):
                result = await sep(audio_file_path=tmp_path)

            body = json.loads(result)
            assert body["foreground_audio_url"] == "http://fg.wav"
            assert body["background_audio_url"] == "http://bg.wav"
        finally:
            os.unlink(tmp_path)

    @pytest.mark.asyncio
    async def test_audio_separation_url_happy_path(
        self, provider, mock_client, mock_httpx_ctx, make_poll_status
    ):
        """httpx mock + full flow → fg/bg URLs."""
        tools = await provider.load_tools()
        sep = [t for t in tools if t.tool_name == "camb_audio_separation"][0]

        mock_httpx, _ = mock_httpx_ctx

        mock_create = MagicMock()
        mock_create.task_id = "task-123"
        mock_client.audio_separation.create_audio_separation = AsyncMock(return_value=mock_create)

        mock_status = make_poll_status()
        mock_client.audio_separation.get_audio_separation_status = AsyncMock(return_value=mock_status)

        mock_sep_result = MagicMock()
        mock_sep_result.foreground_audio_url = "http://fg.wav"
        mock_sep_result.background_audio_url = "http://bg.wav"
        mock_client.audio_separation.get_audio_separation_run_info = AsyncMock(return_value=mock_sep_result)

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            with patch.object(provider._helpers, "_get_client", return_value=mock_client):
                result = await sep(audio_url="https://example.com/audio.wav")

        body = json.loads(result)
        assert body["foreground_audio_url"] == "http://fg.wav"
        assert body["background_audio_url"] == "http://bg.wav"

    @pytest.mark.asyncio
    async def test_audio_separation_no_source(self, provider, mock_client):
        """No url or path → error: Provide either."""
        tools = await provider.load_tools()
        sep = [t for t in tools if t.tool_name == "camb_audio_separation"][0]

        with patch.object(provider._helpers, "_get_client", return_value=mock_client):
            result = await sep()

        body = json.loads(result)
        assert "error" in body
        assert "Provide either" in body["error"]

    @pytest.mark.asyncio
    async def test_audio_separation_httpx_not_installed(self, provider, mock_client):
        """httpx not available → error mentioning httpx."""
        tools = await provider.load_tools()
        sep = [t for t in tools if t.tool_name == "camb_audio_separation"][0]

        with patch.dict("sys.modules", {"httpx": None}):
            with patch.object(provider._helpers, "_get_client", return_value=mock_client):
                result = await sep(audio_url="https://example.com/audio.wav")

        body = json.loads(result)
        assert "error" in body
        assert "httpx" in body["error"]

    @pytest.mark.asyncio
    async def test_audio_separation_file_not_found(self, provider):
        tools = await provider.load_tools()
        sep = [t for t in tools if t.tool_name == "camb_audio_separation"][0]
        mock_client = MagicMock()
        with patch.object(provider._helpers, "_get_client", return_value=mock_client):
            result = await sep(audio_file_path="/nonexistent/audio.wav")
        body = json.loads(result)
        assert "error" in body
        assert "/nonexistent/audio.wav" in body["error"]

    @pytest.mark.asyncio
    async def test_audio_separation_url_download_failure(self, provider):
        tools = await provider.load_tools()
        sep = [t for t in tools if t.tool_name == "camb_audio_separation"][0]

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("Server error")

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_http

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            with patch.object(provider._helpers, "_get_client", return_value=mock_client):
                result = await sep(audio_url="https://example.com/audio.wav")
        body = json.loads(result)
        assert "error" in body
        assert "Failed to download audio" in body["error"]


# ---------------------------------------------------------------------------
# TestCambVoiceFromDescription
# ---------------------------------------------------------------------------


class TestCambVoiceFromDescription:
    @pytest.mark.asyncio
    async def test_voice_from_description_happy_path(self, provider, mock_client, make_poll_status):
        """create/poll/get_result with previews → {"previews":[...],"status":"completed"}."""
        tools = await provider.load_tools()
        vfd = [t for t in tools if t.tool_name == "camb_voice_from_description"][0]

        mock_create = MagicMock()
        mock_create.task_id = "task-123"
        mock_client.text_to_voice.create_text_to_voice = AsyncMock(return_value=mock_create)

        mock_status = make_poll_status()
        mock_client.text_to_voice.get_text_to_voice_status = AsyncMock(return_value=mock_status)

        mock_voice_result = MagicMock()
        mock_voice_result.previews = ["http://preview1.wav", "http://preview2.wav"]
        mock_client.text_to_voice.get_text_to_voice_result = AsyncMock(return_value=mock_voice_result)

        with patch.object(provider._helpers, "_get_client", return_value=mock_client):
            result = await vfd(text="Hello world", voice_description="A warm, friendly voice")

        body = json.loads(result)
        assert body["status"] == "completed"
        assert body["previews"] == ["http://preview1.wav", "http://preview2.wav"]

    @pytest.mark.asyncio
    async def test_voice_from_description_empty_previews(self, provider, mock_client, make_poll_status):
        """previews=[] passes through → {"previews":[],"status":"completed"}."""
        tools = await provider.load_tools()
        vfd = [t for t in tools if t.tool_name == "camb_voice_from_description"][0]

        mock_create = MagicMock()
        mock_create.task_id = "task-123"
        mock_client.text_to_voice.create_text_to_voice = AsyncMock(return_value=mock_create)

        mock_status = make_poll_status()
        mock_client.text_to_voice.get_text_to_voice_status = AsyncMock(return_value=mock_status)

        mock_voice_result = MagicMock()
        mock_voice_result.previews = []
        mock_client.text_to_voice.get_text_to_voice_result = AsyncMock(return_value=mock_voice_result)

        with patch.object(provider._helpers, "_get_client", return_value=mock_client):
            result = await vfd(text="Hello world", voice_description="A warm voice")

        body = json.loads(result)
        assert body["status"] == "completed"
        assert body["previews"] == []
