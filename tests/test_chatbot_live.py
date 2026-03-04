"""Live chatbot integration test — real CAMB API calls with audio playback.

Exercises all 9 CAMB tools against the live API, saving audio to temp files
and playing them back via macOS ``afplay``.

Run with::

    source .env && python -m pytest tests/test_chatbot_live.py -v -s --timeout=300

The ``-s`` flag is required so ``afplay`` output and print statements are visible.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
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

from strands_camb._helpers import CambHelpers  # noqa: E402 (after env setup)

PLAY_SECONDS = 10  # max seconds of audio to play per turn


def _play_audio(path: str, duration: float = PLAY_SECONDS) -> None:
    """Play an audio file with macOS afplay for up to *duration* seconds."""
    try:
        subprocess.run(
            ["afplay", "-t", str(duration), path],
            timeout=duration + 2,
            check=False,
        )
    except FileNotFoundError:
        print(f"  [afplay not available — skipping playback of {path}]")
    except subprocess.TimeoutExpired:
        pass


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not os.getenv("CAMB_API_KEY"),
    reason="CAMB_API_KEY not set (load .env or export it)",
)


class TestChatbotLive:
    """Live 9-turn Podcast Production Assistant — real API calls + audio."""

    @pytest.mark.asyncio
    async def test_live_podcast_assistant(self):
        helpers = CambHelpers()
        client = helpers._get_client()
        temp_files: list[str] = []

        try:
            # ==============================================================
            # Turn 1 — List voices
            # ==============================================================
            print("\n--- Turn 1: List voices ---")
            voices = await client.voice_cloning.list_voices()
            print(f"  Found {len(voices)} voices")
            assert len(voices) > 0
            voice_id = getattr(voices[0], "id", 147320)
            print(f"  Using voice_id={voice_id} for subsequent turns")

            # ==============================================================
            # Turn 2 — Text-to-Speech
            # ==============================================================
            print("\n--- Turn 2: Text-to-Speech ---")
            from camb import StreamTtsOutputConfiguration

            tts_chunks: list[bytes] = []
            async for chunk in client.text_to_speech.tts(
                text="Welcome to our podcast, powered by Strands SDK and CAMB AI! Today we explore the world of AI audio.",
                language="en-us",
                voice_id=voice_id,
                speech_model="mars-flash",
                output_configuration=StreamTtsOutputConfiguration(format="wav"),
            ):
                tts_chunks.append(chunk)

            tts_data = b"".join(tts_chunks)
            tts_path = helpers._save_audio(tts_data, ".wav")
            temp_files.append(tts_path)
            print(f"  Audio saved: {tts_path} ({len(tts_data):,} bytes)")
            assert len(tts_data) > 0
            _play_audio(tts_path)

            # ==============================================================
            # Turn 3 — Translation
            # (SDK may raise ApiError with status 200 for plain-text responses)
            # ==============================================================
            print("\n--- Turn 3: Translation ---")
            from camb.core.api_error import ApiError

            try:
                tr_result = await client.translation.translation_stream(
                    text="Strands SDK and CAMB AI make building audio agents easy!",
                    source_language=1,
                    target_language=2,
                )
                translated = helpers._extract_translation(tr_result)
            except ApiError as e:
                if e.status_code == 200 and e.body:
                    translated = str(e.body)
                else:
                    raise
            print(f"  Translated: {translated}")
            assert len(translated) > 0

            # ==============================================================
            # Turn 4 — Translated TTS
            # ==============================================================
            print("\n--- Turn 4: Translated TTS ---")
            import httpx

            tt_result = await client.translated_tts.create_translated_tts(
                text="Strands SDK and CAMB AI make building audio agents easy!",
                source_language=1,
                target_language=2,
                voice_id=voice_id,
            )
            tt_status = await helpers._poll_async(
                client.translated_tts.get_translated_tts_task_status,
                tt_result.task_id,
            )
            run_id = tt_status.run_id
            url = f"https://client.camb.ai/apis/tts-result/{run_id}"
            async with httpx.AsyncClient() as http:
                resp = await http.get(url, headers={"x-api-key": helpers._api_key or ""})

            if resp.status_code == 200 and resp.content:
                fmt = helpers._detect_audio_format(resp.content, resp.headers.get("content-type", ""))
                tt_audio = resp.content
                if fmt == "pcm":
                    tt_audio = helpers._add_wav_header(tt_audio)
                    fmt = "wav"
                ext_map = {"wav": ".wav", "mp3": ".mp3", "flac": ".flac", "ogg": ".ogg"}
                tt_path = helpers._save_audio(tt_audio, ext_map.get(fmt, ".wav"))
                temp_files.append(tt_path)
                print(f"  Translated TTS audio saved: {tt_path} ({len(tt_audio):,} bytes)")
                _play_audio(tt_path)
            else:
                print(f"  Translated TTS download returned status {resp.status_code}")

            # ==============================================================
            # Turn 5 — Transcription
            # ==============================================================
            print("\n--- Turn 5: Transcription ---")
            with open(tts_path, "rb") as f:
                tc_result = await client.transcription.create_transcription(
                    language=1, media_file=f,
                )
            tc_status = await helpers._poll_async(
                client.transcription.get_transcription_task_status,
                tc_result.task_id,
            )
            transcription = await client.transcription.get_transcription_result(tc_status.run_id)
            tc_json = helpers._format_transcription(transcription)
            tc_data = json.loads(tc_json)
            tc_text = tc_data.get("text", "")
            tc_segments = tc_data.get("segments", [])
            print(f"  Transcribed: {tc_text or '(text empty, checking segments)'}")
            if tc_segments:
                print(f"  Segments: {tc_segments[:3]}")
            assert tc_text or tc_segments, "Transcription returned no text and no segments"

            # ==============================================================
            # Turn 6 — Audio Separation
            # ==============================================================
            print("\n--- Turn 6: Audio Separation ---")
            with open(tts_path, "rb") as f:
                sep_result = await client.audio_separation.create_audio_separation(media_file=f)
            sep_status = await helpers._poll_async(
                client.audio_separation.get_audio_separation_status,
                sep_result.task_id,
            )
            sep_info = await client.audio_separation.get_audio_separation_run_info(sep_status.run_id)
            fg_url = getattr(sep_info, "foreground_audio_url", None)
            bg_url = getattr(sep_info, "background_audio_url", None)
            print(f"  Foreground: {fg_url}")
            print(f"  Background: {bg_url}")
            assert fg_url or bg_url

            # ==============================================================
            # Turn 7 — Text to Sound
            # ==============================================================
            print("\n--- Turn 7: Text to Sound ---")
            snd_result = await client.text_to_audio.create_text_to_audio(
                prompt="upbeat tech podcast intro jingle, futuristic synth vibes",
            )
            snd_status = await helpers._poll_async(
                client.text_to_audio.get_text_to_audio_status,
                snd_result.task_id,
            )
            snd_chunks: list[bytes] = []
            async for chunk in client.text_to_audio.get_text_to_audio_result(snd_status.run_id):
                snd_chunks.append(chunk)

            snd_data = b"".join(snd_chunks)
            snd_path = helpers._save_audio(snd_data, ".wav")
            temp_files.append(snd_path)
            print(f"  Sound saved: {snd_path} ({len(snd_data):,} bytes)")
            assert len(snd_data) > 0
            _play_audio(snd_path)

            # ==============================================================
            # Turn 8 — Clone Voice
            # ==============================================================
            print("\n--- Turn 8: Clone Voice ---")
            clone_sample = Path(__file__).resolve().parent.parent.parent / "yt-dlp" / "voices" / "original" / "sydney-original-clip.mp3"
            assert clone_sample.exists(), f"Voice sample not found: {clone_sample}"
            with open(clone_sample, "rb") as f:
                clone_result = await client.voice_cloning.create_custom_voice(
                    voice_name="PodcastHostLiveTest",
                    gender=0,
                    file=f,
                )
            clone_id = getattr(clone_result, "voice_id", getattr(clone_result, "id", None))
            print(f"  Cloned voice ID: {clone_id}")
            assert clone_id is not None

            # ==============================================================
            # Turn 9 — Voice from Description
            # ==============================================================
            print("\n--- Turn 9: Voice from Description ---")
            vfd_result = await client.text_to_voice.create_text_to_voice(
                text=(
                    "Hey everyone, welcome to the Strands SDK podcast! "
                    "Today we are diving into how CAMB AI powers multilingual audio agents."
                ),
                voice_description="professional podcast host, warm and clear",
            )
            vfd_status = await helpers._poll_async(
                client.text_to_voice.get_text_to_voice_status,
                vfd_result.task_id,
            )
            vfd_info = await client.text_to_voice.get_text_to_voice_result(vfd_status.run_id)
            previews = getattr(vfd_info, "previews", [])
            print(f"  Voice previews: {previews}")

            print("\n=== All 9 turns completed successfully! ===")

        finally:
            for path in temp_files:
                try:
                    os.unlink(path)
                except OSError:
                    pass
