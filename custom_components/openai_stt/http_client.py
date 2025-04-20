"""HTTP client for OpenAI STT."""

from __future__ import annotations

from collections.abc import AsyncIterable
import io
import logging
import wave

import httpx

from homeassistant.components.stt import SpeechMetadata, SpeechResult, SpeechResultState

_LOGGER = logging.getLogger(__name__)


async def async_process_audio_stream(
    client,
    api_key: str,
    api_url: str,
    model: str,
    prompt: str,
    temperature: int,
    metadata: SpeechMetadata,
    stream: AsyncIterable[bytes],
) -> SpeechResult:
    """Process audio stream using HTTP POST."""
    _LOGGER.debug(
        "Start processing audio stream via HTTP for language: %s", metadata.language
    )

    # Collect audio data from the stream
    audio_data = b""
    async for chunk in stream:
        audio_data += chunk

    _LOGGER.debug("Audio data size: %d bytes", len(audio_data))

    # Convert audio data to WAV format
    wav_stream = io.BytesIO()

    with wave.open(wav_stream, "wb") as wf:
        wf.setnchannels(metadata.channel)
        wf.setsampwidth(metadata.bit_rate // 8)
        wf.setframerate(metadata.sample_rate)
        wf.writeframes(audio_data)

    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    # Prepare multipart form data
    files = {
        "file": ("whisper_audio.wav", wav_stream.getvalue(), "audio/wav"),
        "model": (None, model),
        "language": (None, metadata.language),
        "prompt": (None, prompt),
        "temperature": (None, str(temperature)),
        "response_format": (None, "json"),
    }

    _LOGGER.debug(
        "Preparing request to API: %s",
        {k: v for k, v in files.items() if k != "file"},
    )

    url = f"{api_url}/audio/transcriptions"

    _LOGGER.debug("Sending request to API: %s", url)

    try:
        # Send the request to the API
        response = await client.post(
            url,
            headers=headers,
            files=files,
            timeout=httpx.Timeout(10.0),
        )
        response.raise_for_status()
        result = response.json()
        _LOGGER.debug("API response: %s", result)
    except httpx.HTTPError as err:
        if hasattr(err, "response") and err.response:
            _LOGGER.error(
                "HTTP error %s: %s",
                err.response.status_code,
                err.response.json()["error"]["message"],
            )
        else:
            _LOGGER.error("HTTP error: %s", err)
        return SpeechResult("", SpeechResultState.ERROR)
    except Exception:
        _LOGGER.exception("Error sending audio")
        return SpeechResult("", SpeechResultState.ERROR)

    final_text = result.get("text").strip() if result.get("text") else ""

    _LOGGER.debug("Final text processed: %s", final_text)

    if not final_text:
        _LOGGER.warning("HTTP transcription resulted in empty text")
        return SpeechResult("", SpeechResultState.SUCCESS)

    return SpeechResult(final_text, SpeechResultState.SUCCESS)
