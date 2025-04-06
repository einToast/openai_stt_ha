# OpenAI Speech-To-Text for Home Assistant

This custom component integrates [OpenAI Speech-to-Text](https://platform.openai.com/docs/guides/speech-to-text), also known as Whisper, into Home Assistant via the OpenAI API for use in the Assist pipeline.

## Installation

### HACS (Recommended)


This integration is part of the standard HACS repository. Just search for "OpenAI Whisper API" to install it, or use this link to go directly there:

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=einToast&repository=openai_stt_ha&category=integration)


### Manual

1. Inside your `config` directory, create a new directory named `custom_components`
2. Create a new directory named `openai_stt` inside the `custom_components` directory
3. Place all the files from this repository in the `openai_stt` directory
4. Restart Home Assistant

## Configuration

You need to create an account on the [OpenAI website](https://platform.openai.com/signup) and get an [API key](https://platform.openai.com/api-keys).
Then add the following to your `configuration.yaml` and restart Home Assistant:

```yaml
stt:
  - platform: openai_stt
    api_key: YOUR_API_KEY
    #  Optional parameters
    api_url: https://api.openai.com/v1
    model: gpt-4o-mini-transcribe
    prompt: ""
    temperature: 0
```

Parameters:

- `api_key` (Required): Your OpenAI API key
- `api_url` (Optional): The API URL to use. The default is `https://api.openai.com/v1`. Specify this to use any compatible OpenAI API
- `model` (Optional): The model to use. The default is `gpt-4o-mini-transcribe`. Currently, the [supported models](#models) are `gpt-4o-mini-transcribe`, `gpt-4o-transcribe` and `whisper-1`. All available models are listed in the [OpenAI model list](https://platform.openai.com/docs/models) under the Transcription section
- `prompt` (Optional): The prompt to use. The default is an empty string. See the [OpenAI documentation](https://platform.openai.com/docs/guides/speech-to-text#prompting) for more information
- `temperature` (Optional): The temperature to use between 0 and 1. The default is 0. A higher temperature will make the model more creative, but less accurate. See the [OpenAI API reference](https://platform.openai.com/docs/api-reference/audio/createTranscription#audio-createtranscription-temperature) for more information

## Models
See the accuracy comparison of the models [here](https://openai.com/index/introducing-our-next-generation-audio-models/).

- `gpt-4o-mini-transcribe`: model optimized for speed and cost. Cost: estimated $0.003 per minute of audio
- `gpt-4o-transcribe`: model optimized for accuracy. Cost: estimated $0.006 per minute of audio
- `whisper-1`: original `whisper-large-v2` model. Superseded by `gpt-4o-mini-transcribe` and `gpt-4o-transcribe`. Cost: $0.006 per minute of audio

## Error

If you get the following error in the Home Assistant system log:

```
The stt integration does not support any configuration parameters, got [{'platform': 'openai_stt', 'api_key': 'YOUR_API_KEY'}]. Please remove the configuration parameters from your configuration.
```

This issues is a known [bug](https://github.com/home-assistant/core/issues/97161) in Home Assistant >= 2023.7. The reported message does not affect the functionality of this integration, it should still work.