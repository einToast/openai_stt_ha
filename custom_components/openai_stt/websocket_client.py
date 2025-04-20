"""WebSocket client for OpenAI STT."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterable
import json
import logging
from typing import Final

import aiohttp

from homeassistant.components.stt import SpeechMetadata, SpeechResult, SpeechResultState

_LOGGER = logging.getLogger(__name__)

# Maximum time to wait for a response (in seconds)
WEBSOCKET_TIMEOUT: Final = 30


async def _send_audio_stream(
    ws: aiohttp.ClientWebSocketResponse, stream: AsyncIterable[bytes]
) -> None:
    """Send audio chunks to WebSocket server."""
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


async def _receive_transcription(
    ws: aiohttp.ClientWebSocketResponse, send_task: asyncio.Task
) -> str:
    """Receive transcription results from WebSocket server."""
    final_text = ""
    try:
        async with asyncio.timeout(WEBSOCKET_TIMEOUT):
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    _LOGGER.debug("Received response: %s", data)

                    msg_type = data.get("type")
                    if msg_type == "conversation.item.input_audio_transcription.delta":
                        _LOGGER.debug('Partial: "%s"', data.get("delta"))
                    elif (
                        msg_type
                        == "conversation.item.input_audio_transcription.completed"
                    ):
                        # Get final transcription
                        text = data.get("transcript", "")
                        final_text = text
                        _LOGGER.debug('Final: "%s"', text)
                        return final_text
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    _LOGGER.error("WebSocket error: %s", ws.exception())
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    _LOGGER.debug("WebSocket closed by server")
                    break
    except TimeoutError:
        _LOGGER.warning("Timeout waiting for transcription response")
    except asyncio.CancelledError:
        _LOGGER.debug("receive_transcription() was cancelled")
    except Exception:
        _LOGGER.exception("Error receiving transcription")
    finally:
        # Ensure the sending task is cancelled if the receiving task finishes first
        if not send_task.done():
            send_task.cancel()

    return final_text


def _create_session_config(
    model: str, prompt: str, language: str, noise_reduction: str
) -> dict:
    """Create configuration for the transcription session."""
    config = {
        "type": "transcription_session.update",
        "session": {
            "input_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": model,
                "prompt": prompt,
                "language": language,
            },
            "turn_detection": {
                "type": "server_vad",
            },
        },
    }

    # Add input_audio_noise_reduction to config, if specified
    if noise_reduction:
        config["session"]["input_audio_noise_reduction"] = {"type": noise_reduction}
    else:
        config["session"]["input_audio_noise_reduction"] = None

    return config


async def _handle_tasks(
    send_task: asyncio.Task,
    recv_task: asyncio.Task,
    ws: aiohttp.ClientWebSocketResponse,
) -> None:
    """Handle task completion and cancellation logic."""
    # Wait for the first task to complete
    try:
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
    finally:
        # Always ensure connection is properly closed
        if not ws.closed:
            try:
                await ws.close()
                _LOGGER.debug("WebSocket closed cleanly")
            except Exception:
                _LOGGER.exception("Error closing WebSocket connection")


async def async_process_audio_stream(
    client,
    api_key: str,
    api_url: str,
    model: str,
    prompt: str,
    noise_reduction: str,
    metadata: SpeechMetadata,
    stream: AsyncIterable[bytes],
) -> SpeechResult:
    """Process audio stream using WebSocket."""
    _LOGGER.debug(
        "Start processing audio stream via WebSocket for language: %s",
        metadata.language,
    )

    uri = f"{api_url}/realtime?intent=transcription"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "realtime=v1",
    }

    try:
        _LOGGER.debug("Opening WebSocket connection to %s", uri)
        async with client.ws_connect(uri, headers=headers, heartbeat=30) as ws:
            _LOGGER.debug("Language: %s", metadata.language)

            # Send initial configuration
            config = _create_session_config(
                model, prompt, metadata.language, noise_reduction
            )
            _LOGGER.debug("Sending configuration: %s", config)
            await ws.send_json(config)

            # Create and manage concurrent tasks
            send_task = asyncio.create_task(_send_audio_stream(ws, stream))
            recv_task = asyncio.create_task(_receive_transcription(ws, send_task))

            # Handle tasks completion
            await _handle_tasks(send_task, recv_task, ws)

            # Process final result
            if recv_task.done():
                final_text = recv_task.result().strip()
                _LOGGER.debug('Transcription completed: "%s"', final_text)

                if not final_text:
                    _LOGGER.warning("WebSocket transcription resulted in empty text")
                    return SpeechResult("", SpeechResultState.SUCCESS)

                return SpeechResult(final_text, SpeechResultState.SUCCESS)
            else:
                _LOGGER.warning("Transcription task was not completed")
                return SpeechResult("", SpeechResultState.SUCCESS)
    except aiohttp.ClientError as err:
        _LOGGER.error("WebSocket connection error: %s", err)
        return SpeechResult("", SpeechResultState.ERROR)
    except Exception:
        _LOGGER.exception("Unexpected error in WebSocket communication")
        return SpeechResult("", SpeechResultState.ERROR)
