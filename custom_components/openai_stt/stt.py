"""Setting up OpenAISTTProvider."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterable
import io
import json
import logging
import wave

import aiohttp
import httpx
import voluptuous as vol

from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    Provider,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
import homeassistant.helpers.config_validation as cv
from homeassistant.helpers.httpx_client import get_async_client

_LOGGER = logging.getLogger(__name__)

CONF_API_KEY = "api_key"
CONF_API_URL = "api_url"
CONF_MODEL = "model"
CONF_PROMPT = "prompt"
CONF_TEMP = "temperature"
CONF_REALTIME = "realtime"
CONF_NOISE_REDUCTION = "noise_reduction"

DEFAULT_API_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini-transcribe"
DEFAULT_PROMPT = ""
DEFAULT_TEMP = 0
DEFAULT_REALTIME = False
DEFAULT_NOISE_REDUCTION = None

SUPPORTED_MODELS = [
    "whisper-1",
    "gpt-4o-mini-transcribe",
    "gpt-4o-transcribe",
]

SUPPORTED_NOISE_REDUCTION = [
    None,
    "near_field",
    "far_field",
]

SUPPORTED_LANGUAGES = [
    "af",
    "ar",
    "hy",
    "az",
    "be",
    "bs",
    "bg",
    "ca",
    "zh",
    "hr",
    "cs",
    "da",
    "nl",
    "en",
    "et",
    "fi",
    "fr",
    "gl",
    "de",
    "el",
    "he",
    "hi",
    "hu",
    "is",
    "id",
    "it",
    "ja",
    "kn",
    "kk",
    "ko",
    "lv",
    "lt",
    "mk",
    "ms",
    "mr",
    "mi",
    "ne",
    "no",
    "fa",
    "pl",
    "pt",
    "ro",
    "ru",
    "sr",
    "sk",
    "sl",
    "es",
    "sw",
    "sv",
    "tl",
    "ta",
    "th",
    "tr",
    "uk",
    "ur",
    "vi",
    "cy",
]

MODEL_SCHEMA = vol.In(SUPPORTED_MODELS)
NOISE_REDUCTION_SCHEMA = vol.In(SUPPORTED_NOISE_REDUCTION)

PLATFORM_SCHEMA = cv.PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_API_KEY): cv.string,
        vol.Optional(CONF_API_URL, default=DEFAULT_API_URL): cv.string,
        vol.Optional(CONF_MODEL, default=DEFAULT_MODEL): MODEL_SCHEMA,
        vol.Optional(CONF_PROMPT, default=DEFAULT_PROMPT): cv.string,
        vol.Optional(CONF_TEMP, default=DEFAULT_TEMP): cv.positive_int,
        vol.Optional(CONF_REALTIME, default=DEFAULT_REALTIME): cv.boolean,
        vol.Optional(
            CONF_NOISE_REDUCTION, default=DEFAULT_NOISE_REDUCTION
        ): NOISE_REDUCTION_SCHEMA,
    }
)


async def async_get_engine(
    hass: HomeAssistant, config: dict, discovery_info: dict | None = None
) -> OpenAISTTProvider:
    """Return the OpenAI STT provider."""
    api_key = config[CONF_API_KEY]
    api_url = config.get(CONF_API_URL, DEFAULT_API_URL)
    model = config.get(CONF_MODEL, DEFAULT_MODEL)
    prompt = config.get(CONF_PROMPT, DEFAULT_PROMPT)
    temperature = config.get(CONF_TEMP, DEFAULT_TEMP)
    realtime = config.get(CONF_REALTIME, DEFAULT_REALTIME)
    noise_reduction = config.get(CONF_NOISE_REDUCTION, DEFAULT_NOISE_REDUCTION)

    return OpenAISTTProvider(
        hass, api_key, api_url, model, prompt, temperature, realtime, noise_reduction
    )


class OpenAISTTProvider(Provider):
    """The OpenAI STT provider."""

    def __init__(
        self,
        hass: HomeAssistant,
        api_key: str,
        api_url: str,
        model: str,
        prompt: str,
        temperature: int,
        realtime: bool,
        noise_reduction: str,
    ) -> None:
        """Init OpenAI STT service."""
        self.hass = hass
        self.name = "OpenAI STT"

        self._api_key = api_key
        self._api_url = api_url
        self._model = model
        self._prompt = prompt
        self._temperature = temperature
        self._realtime = realtime
        self._noise_reduction = noise_reduction
        self._client = (
            async_get_clientsession(self.hass) if realtime else get_async_client(hass)
        )

    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return SUPPORTED_LANGUAGES

    @property
    def supported_formats(self) -> list[AudioFormats]:
        """Return a list of supported formats."""
        return [AudioFormats.WAV, AudioFormats.OGG]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        """Return a list of supported codecs."""
        return [AudioCodecs.PCM, AudioCodecs.OPUS]

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        """Return a list of supported bitrates."""
        return [AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        """Return a list of supported samplerates."""
        return [AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self) -> list[AudioChannels]:
        """Return a list of supported channels."""
        return [AudioChannels.CHANNEL_MONO]

    async def _async_process_audio_stream_http(
        self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]
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
            "Authorization": f"Bearer {self._api_key}",
        }

        # Prepare multipart form data
        files = {
            "file": ("whisper_audio.wav", wav_stream.getvalue(), "audio/wav"),
            "model": (None, self._model),
            "language": (None, metadata.language),
            "prompt": (None, self._prompt),
            "temperature": (None, str(self._temperature)),
            "response_format": (None, "json"),
        }

        _LOGGER.debug(
            "Preparing request to API: %s",
            {k: v for k, v in files.items() if k != "file"},
        )

        url = f"{self._api_url}/audio/transcriptions"

        _LOGGER.debug("Sending request to API: %s", url)

        try:
            # Send the request to the API
            response = await self._client.post(
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

        return SpeechResult(result["text"], SpeechResultState.SUCCESS)

    async def _async_process_audio_stream_websocket(
        self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> SpeechResult:
        """Process audio stream using WebSocket."""
        _LOGGER.debug(
            "Start processing audio stream via WebSocket for language: %s",
            metadata.language,
        )
        session = async_get_clientsession(self.hass)

        uri = f"{self._api_url}/realtime?intent=transcription"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        _LOGGER.debug("Opening WebSocket connection to %s", uri)
        async with session.ws_connect(uri, headers=headers) as ws:
            _LOGGER.debug("Language: %s", metadata.language)

            # Send initial configuration for the transcription session
            config = {
                "type": "transcription_session.update",
                "session": {
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": {
                        "model": self._model,
                        "prompt": self._prompt,
                        "language": metadata.language,
                    },
                    "turn_detection": {
                        "type": "server_vad",
                    },
                },
            }

            # Add input_audio_noise_reduction to config, if specified
            if self._noise_reduction:
                config["session"]["input_audio_noise_reduction"] = {
                    "type": self._noise_reduction
                }
            else:
                config["session"]["input_audio_noise_reduction"] = None

            _LOGGER.debug("Sending configuration: %s", config)
            await ws.send_json(config)

            final_text = ""

            # Task to send audio chunks asynchronously
            async def send_audio():
                try:
                    async for chunk in stream:
                        if not chunk or ws.closed:
                            break
                        # Audio data must be base64 encoded
                        b64 = base64.b64encode(chunk).decode("utf-8")
                        await ws.send_json(
                            {
                                "type": "input_audio_buffer.append",
                                "audio": b64,
                            }
                        )
                        _LOGGER.debug("Audio sent (%d bytes)", len(chunk))

                    if not ws.closed:
                        # Signal the end of the audio stream to the server
                        _LOGGER.debug("Sending end-of-stream signal")
                        await ws.send_json({"type": "input_audio_buffer.commit"})

                except asyncio.CancelledError:
                    _LOGGER.debug("send_audio() was cancelled")
                except Exception:
                    _LOGGER.exception("Error sending audio")
                    if not ws.closed:
                        await ws.close(
                            code=aiohttp.WSCloseCode.INTERNAL_ERROR,
                            message=b"Error sending audio",
                        )

            # Task to receive transcription results asynchronously
            async def receive_transcription():
                nonlocal final_text
                try:
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            _LOGGER.debug("Received response: %s", data)

                            msg_type = data.get("type")
                            if (
                                msg_type
                                == "conversation.item.input_audio_transcription.delta"
                            ):
                                _LOGGER.debug('Partial: "%s"', data.get("delta"))
                            elif (
                                msg_type
                                == "conversation.item.input_audio_transcription.completed"
                            ):
                                # Get final transcription
                                text = data.get("transcript", "")
                                final_text = text
                                _LOGGER.debug('Final: "%s"', text)
                                break  # Exit loop on completion
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            _LOGGER.error("WebSocket error: %s", ws.exception())
                            break
                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            _LOGGER.debug("WebSocket closed by server")
                            break
                except asyncio.CancelledError:
                    _LOGGER.debug("receive_transcription() was cancelled")
                except Exception:
                    _LOGGER.exception("Error receiving transcription")
                finally:
                    # Ensure the sending task is cancelled if the receiving task finishes first
                    if not send_task.done():
                        send_task.cancel()

            # Create and manage the concurrent send/receive tasks
            send_task = asyncio.create_task(send_audio())
            recv_task = asyncio.create_task(receive_transcription())

            # Wait for the first task to complete
            done, pending = await asyncio.wait(
                [send_task, recv_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Handle task completion and cancellation logic
            if recv_task in done:
                _LOGGER.debug("Transcription finished - cancelling audio task")
                if not send_task.done():
                    send_task.cancel()
                await asyncio.gather(send_task, return_exceptions=True)
            elif send_task in done:
                _LOGGER.debug("Audio finished - waiting for final transcription")
                await recv_task
            else:
                _LOGGER.warning(
                    "Unexpected state in task completion, ensuring tasks are awaited/cancelled"
                )
                for task in pending:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

            for task in done:
                if task.exception():
                    _LOGGER.error("Task completed with exception: %s", task.exception())

            if not ws.closed:
                await ws.close()
                _LOGGER.debug("WebSocket closed cleanly")

            final_text = final_text.strip()

            _LOGGER.debug('Transcription completed: "%s"', final_text)

            if not final_text:
                _LOGGER.warning("WebSocket transcription resulted in empty text")
                return SpeechResult("", SpeechResultState.SUCCESS)

            return SpeechResult(final_text, SpeechResultState.SUCCESS)

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> SpeechResult:
        """Process audio stream using the configured method (HTTP or WebSocket)."""
        if self._realtime:
            return await self._async_process_audio_stream_websocket(metadata, stream)
        else:
            return await self._async_process_audio_stream_http(metadata, stream)
