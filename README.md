# OpenAI Speech-To-Text for Home Assistant

This custom component integrates [OpenAI Speech-to-Text](https://www.openai.com/speech-to-text), also known as "Whisper", into Home Assistant via the OpenAI API.

## Installation

### HACS

You can install this integration via [HACS](https://hacs.xyz/).

1. Go to HACS / Integrations / Three-dots menu / Custom repositories
2. Add:
   - Repository: `https://github.com/einToast/openai_stt_ha`
   - Category: Integration
3. Install the "OpenAI Speech-To-Text" integration.
4. Restart Home Assistant.

### Manual

1. Inside your `config` directory, create a new directory named `custom_components`.
2. Create a new directory named `openai_stt` inside the `custom_components` directory.
3. Place all the files from this repository in the `openai_stt` directory.
4. Restart Home Assistant.

## Configuration

You need to create an account on the [OpenAI website](https://platform.openai.com/signup) and get an [API key](https://platform.openai.com/api-keys) .
Then add the following to your `configuration.yaml`:

```yaml
stt:
  - platform: openai_stt
    api_key: YOUR_API_KEY
    #  Optional parameters
    model: whisper-1
    prompt: ""
    temperature: 0
```

Parameters:

- `api_key` (Required): Your OpenAI API key.
- `model` (Optional): The model to use. The default is `whisper-1`. Currently, the only available model is `whisper-1`. The available models are listed [here](https://platform.openai.com/docs/models/whisper).
- `prompt` (Optional): The prompt to use. The default is an empty string. See the [OpenAI documentation](https://platform.openai.com/docs/guides/speech-to-text/prompting) for more information.
- `temperature` (Optional): The temperature to use between 0 and 1. The default is 0. A higher temperature will make the model more creative, but less accurate.

## Error

If you get the following error in the Home Assistant system log:

```
The stt integration does not support any configuration parameters, got [{'platform': 'openai_stt', 'api_key': 'YOUR_API_KEY'}]. Please remove the configuration parameters from your configuration.
```

This issues is a known (bug)[https://github.com/home-assistant/core/issues/97161] in Home Assistant >= 2023.7. However, the reported message does not affect the functionality of this integration, it should still work as expected.