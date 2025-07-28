# Twilio AI Phone Agent - Python

AI-powered phone agent using Twilio webhooks, local speech recognition, and conversation models.

## Features

- **Twilio Integration**: Handles incoming calls via webhooks
- **Real-time Audio**: WebSocket streaming with Twilio Media Streams  
- **Local Speech Recognition**: OpenAI Whisper (tiny.en model)
- **AI Conversation**: Microsoft DialoGPT for natural responses
- **Python-based**: Flask + SocketIO for high performance

## Setup

### Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Fill in your Twilio credentials.

### Local Development

```bash
pip install -r requirements.txt
python main.py
```

### Brev Deployment

1. Push to Git repository
2. Connect repo to Brev.dev
3. Set environment variables in Brev dashboard
4. Deploy

### Twilio Configuration

Set webhook URLs in Twilio Console:
- Voice webhook: `https://your-brev-url.com/webhook/voice`
- Status callback: `https://your-brev-url.com/webhook/status`

## Endpoints

- `POST /webhook/voice` - Incoming call handler
- `POST /webhook/status` - Call status updates  
- `GET /health` - Health check
- `WebSocket /websocket` - Real-time audio streaming

## Models

- **Speech-to-Text**: OpenAI Whisper tiny.en
- **Conversation**: Microsoft DialoGPT-small
- **Audio Processing**: ¼-law to PCM conversion