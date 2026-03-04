# strands-agents-camb

> **Community-maintained package** - This package is not owned or supported by the Strands team. It is maintained by [CAMB.AI](https://camb.ai).

CAMB.AI audio and speech tools for [Strands Agents](https://github.com/strands-agents/sdk-python) — 9 tools, 140+ languages.

## Prerequisites

- Python >= 3.10
- CAMB AI API key — get one at [studio.camb.ai](https://studio.camb.ai)
- `strands-agents` >= 1.0.0 (auto-installed as a dependency)

## Installation

```bash
pip install strands-agents-camb
```

The `[url]` extra installs `httpx`, which is required for translated TTS (always) and for URL-based audio operations (transcription from URL, audio separation from URL):

```bash
pip install 'strands-agent-camb[url]'
```

## Quick Start

```python
from strands import Agent
from strands_camb import CambAIToolProvider

provider = CambAIToolProvider(api_key="your-camb-api-key")
agent = Agent(tools=[provider])

response = agent("Convert 'Hello world' to speech")
```

You can also set the `CAMB_API_KEY` environment variable instead of passing it directly.

### Multi-Tool Agent

Enable only the tools you need:

```python
from strands import Agent
from strands_camb import CambAIToolProvider

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

agent = Agent(tools=[provider])
response = agent("Translate 'Good morning' to French")
```

## Available Tools

| Tool | Description |
|------|-------------|
| `camb_tts` | Text-to-speech with 140+ languages and multiple voice models |
| `camb_translate` | Text translation between 140+ languages |
| `camb_transcribe` | Audio transcription with speaker identification |
| `camb_translated_tts` | Translate text and convert to speech in one step |
| `camb_clone_voice` | Clone a voice from a 2+ second audio sample |
| `camb_list_voices` | List all available voices |
| `camb_text_to_sound` | Generate sounds, music, or soundscapes from text |
| `camb_audio_separation` | Separate vocals/speech from background audio |
| `camb_voice_from_description` | Generate a synthetic voice from a text description |

## Tool Examples

### Text-to-Speech

```python
from strands import Agent
from strands_camb import CambAIToolProvider

agent = Agent(tools=[CambAIToolProvider()])

agent("Convert 'Hello, how are you today?' to speech using voice 147320 in English")

# Supports multiple models: mars-flash (default), mars-pro, mars-instruct
agent("Say 'Welcome to our podcast' using the mars-pro model")

# Adjust speed
agent("Generate speech for 'Breaking news' at 1.5x speed")
```

Parameters: `text` (3–3,000 characters), `language` (BCP-47, default `"en-us"`), `voice_id` (default `147320`), `speech_model`, `speed`, `user_instructions` (mars-instruct only). Output is always a WAV temp file.

### Translation

```python
from strands import Agent
from strands_camb import CambAIToolProvider

agent = Agent(tools=[CambAIToolProvider()])

agent("Translate 'Hello, how are you?' from English to Spanish with formal tone")
```

Parameters: `text`, `source_language` (integer code), `target_language` (integer code), `formality` (`1`=formal, `2`=informal).

### Transcription

```python
from strands import Agent
from strands_camb import CambAIToolProvider

agent = Agent(tools=[CambAIToolProvider()])

# From a local file
agent("Transcribe the audio file at /path/to/recording.mp3 in English")

# From a URL (requires strands-camb[url])
agent("Transcribe the audio at https://example.com/audio.mp3 in English")
```

Returns full text, timed segments with start/end timestamps, and speaker identification.

Parameters: `language` (integer code), `audio_url` or `audio_file_path`.

### Translated TTS

```python
from strands import Agent
from strands_camb import CambAIToolProvider

agent = Agent(tools=[CambAIToolProvider()])

# Translate and speak in one step (requires strands-camb[url])
agent("Translate 'Good morning everyone' from English to French and generate speech")
```

Parameters: `text`, `source_language`, `target_language`, `voice_id` (default `147320`), `formality`.

### Voice Cloning

```python
from strands import Agent
from strands_camb import CambAIToolProvider

agent = Agent(tools=[CambAIToolProvider()])

agent("Clone a voice named 'My Voice' from the audio sample at /path/to/sample.wav")
```

The audio sample must be at least 2 seconds long. Returns a `voice_id` for use with TTS tools.

Parameters: `voice_name`, `audio_file_path`, `gender` (`0`=Not Specified, `1`=Male, `2`=Female, `9`=N/A), `description`, `age`, `language`.

### List Voices

```python
from strands import Agent
from strands_camb import CambAIToolProvider

agent = Agent(tools=[CambAIToolProvider()])

agent("List all available voices")
```

Returns an array of `{id, voice_name}` objects. Use the `id` as `voice_id` in TTS calls.

### Text-to-Sound

```python
from strands import Agent
from strands_camb import CambAIToolProvider

agent = Agent(tools=[CambAIToolProvider()])

agent("Generate 30 seconds of upbeat electronic music with a driving beat")

agent("Create a sound effect of rain on a tin roof")
```

Parameters: `prompt`, `duration` (seconds), `audio_type` (`"music"` or `"sound"`).

### Audio Separation

```python
from strands import Agent
from strands_camb import CambAIToolProvider

agent = Agent(tools=[CambAIToolProvider()])

# From a local file
agent("Separate the vocals from /path/to/song.mp3")

# From a URL (requires strands-camb[url])
agent("Separate vocals and background from https://example.com/song.mp3")
```

Returns URLs to separated foreground (vocals) and background audio files.

Parameters: `audio_url` or `audio_file_path`.

### Voice from Description

```python
from strands import Agent
from strands_camb import CambAIToolProvider

agent = Agent(tools=[CambAIToolProvider()])

agent("Generate a voice that sounds like a warm, friendly female narrator and say 'Welcome to the show'")
```

Returns JSON with a `previews` array containing preview audio URLs for the generated voice.

Parameters: `text` (sample text to speak), `voice_description` (detailed description of the desired voice).

## Selective Tools

### TTS-Focused Agent

```python
from strands_camb import CambAIToolProvider

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
```

### Translation-Focused Agent

```python
from strands_camb import CambAIToolProvider

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
```

## Language Codes

CAMB AI uses integer language codes for translation and transcription, and BCP-47 strings for TTS.

| Code | Language | BCP-47 |
|------|----------|--------|
| 1 | English (US) | en-us |
| 31 | German (Germany) | de-de |
| 54 | Spanish (Spain) | es-es |
| 76 | French (France) | fr-fr |
| 87 | Italian | it-it |
| 88 | Japanese | ja-jp |
| 94 | Korean | ko-kr |
| 108 | Dutch | nl-nl |
| 111 | Portuguese (Brazil) | pt-br |
| 114 | Russian | ru-ru |
| 139 | Chinese (Simplified) | zh-cn |

For the full list of 140+ supported languages, see the [CAMB AI documentation](https://docs.camb.ai) or call `client.languages.get_source_languages()` from the `camb` SDK.

## Configuration

### API Key

```python
from strands_camb import CambAIToolProvider

# Via environment variable
import os
os.environ["CAMB_API_KEY"] = "your-api-key"

provider = CambAIToolProvider()

# Or pass directly
provider = CambAIToolProvider(api_key="your-api-key")
```

### Timeouts and Polling

```python
from strands_camb import CambAIToolProvider

provider = CambAIToolProvider(
    api_key="your-key",
    timeout=60.0,            # HTTP request timeout in seconds
    max_poll_attempts=60,    # max polling attempts for async tasks
    poll_interval=2.0,       # seconds between poll attempts
)
```

## Agent Integration

Strands Agents maintain conversation context across turns, enabling multi-step audio workflows:

```python
from strands import Agent
from strands_camb import CambAIToolProvider

provider = CambAIToolProvider()
agent = Agent(tools=[provider])

# Multi-turn conversation
agent("List available voices")
agent("Generate speech saying 'Hello world' using the first voice")
agent("Now translate that to Spanish and generate speech")
agent("Separate the vocals from the last audio file")
```

## License

MIT
