"""WebSocket client for OpenAI STT."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterable
import json
import logging
import time
from typing import Final

from aiohttp import ClientError, WSCloseCode, WSMsgType

from homeassistant.components.stt import SpeechMetadata, SpeechResult, SpeechResultState

_LOGGER = logging.getLogger(__name__)

# Maximum time to wait for a response (in seconds)
WEBSOCKET_TIMEOUT: Final = 30


class OpenAIWebSocketClient:
    """WebSocket client for OpenAI STT API."""

    def __init__(
        self,
        client,
        api_key: str,
        api_url: str,
        model: str,
        prompt: str,
        noise_reduction: str,
    ) -> None:
        """Initialize the WebSocket client."""
        self.client = client
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.prompt = prompt
        self.noise_reduction = noise_reduction
        self.start_time = 0.0
        self.ws = None

    async def _send_audio_stream(self, stream: AsyncIterable[bytes]) -> None:
        """Send audio chunks to WebSocket server."""
        try:
            async for chunk in stream:
                if not chunk or self.ws.closed:
                    break
                # Audio data must be base64 encoded
                b64 = base64.b64encode(chunk).decode("utf-8")
                await self.ws.send_json(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": b64,
                    }
                )
                _LOGGER.debug("Audio sent (%d bytes)", len(chunk))

            if not self.ws.closed:
                # Signal the end of the audio stream to the server
                _LOGGER.debug("Sending end-of-stream signal")
                await self.ws.send_json({"type": "input_audio_buffer.commit"})

                # Set start time after sending all audio data
                self.start_time = time.perf_counter()

        except asyncio.CancelledError:
            _LOGGER.debug("send_audio() was cancelled")
        except Exception:
            _LOGGER.exception("Error sending audio")
            if not self.ws.closed:
                await self.ws.close(
                    code=WSCloseCode.INTERNAL_ERROR,
                    message=b"Error sending audio",
                )

    async def _receive_transcription(self, send_task: asyncio.Task) -> str:
        """Receive transcription results from WebSocket server."""
        final_text = ""
        try:
            async with asyncio.timeout(WEBSOCKET_TIMEOUT):
                async for msg in self.ws:
                    if msg.type == WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        msg_type = data.get("type")
                        _LOGGER.debug("Received response: %s", data)

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
                            final_text = data.get("transcript", "")
                            if (
                                self.start_time > 0
                            ):  # Only calculate if start_time is set
                                duration = time.perf_counter() - self.start_time
                                _LOGGER.debug(
                                    "Transcription processing duration: %.2f seconds",
                                    duration,
                                )
                            else:
                                _LOGGER.debug(
                                    "Could not calculate processing duration: start_time not set"
                                )
                            _LOGGER.debug('Final: "%s"', final_text)
                            return final_text
                    elif msg.type == WSMsgType.ERROR:
                        _LOGGER.error("WebSocket error: %s", self.ws.exception())
                        break
                    elif msg.type == WSMsgType.CLOSED:
                        _LOGGER.debug("WebSocket closed by server")
                        break
        except TimeoutError:
            _LOGGER.warning("Timeout waiting for transcription response")
        except asyncio.CancelledError:
            _LOGGER.debug("receive_transcription() was cancelled")
        except Exception:
            _LOGGER.exception("Error receiving transcription")
        finally:
            if not send_task.done():
                send_task.cancel()

        return final_text

    def _create_session_config(self, language: str) -> dict:
        """Create configuration for the transcription session."""
        config = {
            "type": "transcription_session.update",
            "session": {
                "input_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": self.model,
                    "prompt": self.prompt,
                    "language": language,
                },
                "turn_detection": {
                    "type": "server_vad",
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                    "threshold": 0.5,
                },
            },
        }

        if self.noise_reduction:
            config["session"]["input_audio_noise_reduction"] = {
                "type": self.noise_reduction
            }
        else:
            config["session"]["input_audio_noise_reduction"] = None

        return config

    async def _handle_tasks(
        self, send_task: asyncio.Task, recv_task: asyncio.Task
    ) -> None:
        """Handle task completion and cancellation logic."""
        try:
            done, pending = await asyncio.wait(
                [send_task, recv_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Handle task completion and cancellation
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
        finally:
            if not self.ws.closed:
                try:
                    await self.ws.close()
                    _LOGGER.debug("WebSocket closed cleanly")
                except Exception:
                    _LOGGER.exception("Error closing WebSocket connection")

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> SpeechResult:
        """Process audio stream via WebSocket to OpenAI Realtime API."""

        uri = f"{self.api_url}/realtime?intent=transcription"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        try:
            _LOGGER.debug("Opening WebSocket connection to %s", uri)
            async with self.client.ws_connect(uri, headers=headers, heartbeat=30) as ws:
                self.ws = ws
                self.start_time = 0  # Reset start_time

                # Send initial configuration
                config = self._create_session_config(metadata.language)
                _LOGGER.debug("Sending configuration: %s", config)
                await ws.send_json(config)

                # Create and manage concurrent tasks
                send_task = asyncio.create_task(self._send_audio_stream(stream))
                recv_task = asyncio.create_task(self._receive_transcription(send_task))

                # Handle tasks completion
                await self._handle_tasks(send_task, recv_task)

                # Process final result
                if not recv_task.done():
                    _LOGGER.warning("Transcription task was not completed")
                    return SpeechResult("", SpeechResultState.SUCCESS)

                final_text = recv_task.result().strip()

                _LOGGER.debug('Transcription completed: "%s"', final_text)

                if not final_text:
                    _LOGGER.warning("WebSocket transcription resulted in empty text")
                    return SpeechResult("", SpeechResultState.SUCCESS)

                return SpeechResult(final_text, SpeechResultState.SUCCESS)

        except ClientError as err:
            _LOGGER.error("WebSocket connection error: %s", err)
            return SpeechResult("", SpeechResultState.ERROR)
        except Exception:
            _LOGGER.exception("Unexpected error in WebSocket communication")
            return SpeechResult("", SpeechResultState.ERROR)
