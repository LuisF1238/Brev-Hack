import os
import json
import base64
import asyncio
from typing import Dict, List, Optional
from flask import Flask, request, Response
from flask_socketio import SocketIO, emit
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from twilio.twiml.voice_response import VoiceResponse
import threading
from collections import defaultdict
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables for models and call tracking
whisper_processor = None
whisper_model = None
conversation_tokenizer = None
conversation_model = None
active_calls = {}

def initialize_models():
    """Initialize AI models on startup"""
    global whisper_processor, whisper_model, conversation_tokenizer, conversation_model
    
    print("Initializing AI models...")
    
    # Initialize Whisper for speech recognition
    model_name = "openai/whisper-tiny.en"
    whisper_processor = WhisperProcessor.from_pretrained(model_name)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    # Initialize conversation model
    conv_model_name = "microsoft/DialoGPT-small"
    conversation_tokenizer = AutoTokenizer.from_pretrained(conv_model_name)
    conversation_model = AutoModelForCausalLM.from_pretrained(conv_model_name)
    
    # Add pad token if it doesn't exist
    if conversation_tokenizer.pad_token is None:
        conversation_tokenizer.pad_token = conversation_tokenizer.eos_token
    
    print("AI models initialized successfully")

class CallData:
    def __init__(self, call_sid: str):
        self.call_sid = call_sid
        self.audio_buffer = []
        self.conversation_history = []
        self.is_processing = False
        self.chat_history_ids = None

@app.route('/webhook/voice', methods=['POST'])
def handle_incoming_call():
    """Handle incoming Twilio voice calls"""
    caller_number = request.form.get('From', 'Unknown')
    call_sid = request.form.get('CallSid')
    
    print(f"Incoming call from: {caller_number}, CallSid: {call_sid}")
    
    # Create TwiML response
    response = VoiceResponse()
    
    # Greet the caller
    response.say("Hello! I am your AI assistant. Please speak after the tone.", 
                voice='alice', language='en-US')
    
    # Start media stream
    response.start().stream(
        url=f"wss://{request.host}/websocket",
        track='inbound_track'
    )
    
    # Keep the call active
    response.say("I am listening. Please go ahead and speak.",
                voice='alice', language='en-US')
    
    return Response(str(response), mimetype='text/xml')

@app.route('/webhook/status', methods=['POST'])
def handle_call_status():
    """Handle Twilio call status updates"""
    call_sid = request.form.get('CallSid')
    call_status = request.form.get('CallStatus')
    
    print(f"Call status update - CallSid: {call_sid}, Status: {call_status}")
    
    # Clean up call data when call ends
    if call_status == 'completed' and call_sid in active_calls:
        del active_calls[call_sid]
        print(f"Cleaned up call data for {call_sid}")
    
    return '', 200

@socketio.on('connect')
def handle_websocket_connect():
    """Handle WebSocket connection for media stream"""
    print("WebSocket connection established")

@socketio.on('disconnect')
def handle_websocket_disconnect():
    """Handle WebSocket disconnection"""
    print("WebSocket connection closed")

@socketio.on('media')
def handle_media_message(data):
    """Handle incoming media data from Twilio"""
    try:
        # Parse the incoming message
        if isinstance(data, str):
            data = json.loads(data)
        
        event = data.get('event')
        
        if event == 'connected':
            print("Media stream connected")
        elif event == 'start':
            call_sid = data.get('start', {}).get('callSid', str(uuid.uuid4()))
            print(f"Media stream started for call: {call_sid}")
            active_calls[call_sid] = CallData(call_sid)
        elif event == 'media':
            call_sid = data.get('streamSid', 'unknown')
            if call_sid in active_calls:
                audio_payload = data.get('media', {}).get('payload', '')
                process_audio_data(active_calls[call_sid], audio_payload)
        elif event == 'stop':
            call_sid = data.get('streamSid', 'unknown')
            print(f"Media stream stopped for call: {call_sid}")
            if call_sid in active_calls:
                del active_calls[call_sid]
                
    except Exception as e:
        print(f"Error processing media message: {e}")

def process_audio_data(call_data: CallData, audio_payload: str):
    """Process incoming audio data"""
    if call_data.is_processing:
        return
    
    # Decode base64 audio data
    try:
        audio_data = base64.b64decode(audio_payload)
        call_data.audio_buffer.append(audio_data)
        
        # Process when we have enough audio data
        if len(call_data.audio_buffer) >= 10:  # Adjust threshold as needed
            call_data.is_processing = True
            
            # Process in a separate thread to avoid blocking
            threading.Thread(
                target=process_audio_chunk,
                args=(call_data,),
                daemon=True
            ).start()
            
    except Exception as e:
        print(f"Error processing audio data: {e}")

def process_audio_chunk(call_data: CallData):
    """Process accumulated audio chunk"""
    try:
        # Combine audio buffers
        combined_audio = b''.join(call_data.audio_buffer)
        call_data.audio_buffer = []
        
        # Convert μ-law to PCM
        audio_array = mulaw_to_pcm(combined_audio)
        
        # Transcribe audio
        transcription = transcribe_audio(audio_array)
        
        if transcription and len(transcription.strip()) > 3:
            print(f"User said: {transcription}")
            
            # Generate AI response
            ai_response = generate_ai_response(call_data, transcription)
            print(f"AI response: {ai_response}")
            
            # Send response back to caller
            send_tts_response(call_data, ai_response)
            
    except Exception as e:
        print(f"Error in process_audio_chunk: {e}")
    finally:
        call_data.is_processing = False

def mulaw_to_pcm(mulaw_data: bytes) -> np.ndarray:
    """Convert μ-law encoded audio to PCM float32"""
    # Convert bytes to numpy array
    mulaw_array = np.frombuffer(mulaw_data, dtype=np.uint8)
    
    # μ-law to linear conversion
    mulaw_array = mulaw_array.astype(np.int16)
    sign = (mulaw_array & 0x80) != 0
    exponent = (mulaw_array & 0x70) >> 4
    mantissa = mulaw_array & 0x0F
    
    # Decode μ-law
    sample = mantissa * 2 + 33
    sample = sample << (exponent + 2)
    sample = (sample - 33)
    sample = np.where(sign, -sample, sample)
    
    # Convert to float32 and normalize
    pcm_data = sample.astype(np.float32) / 32768.0
    
    return pcm_data

def transcribe_audio(audio_data: np.ndarray) -> str:
    """Transcribe audio using Whisper"""
    try:
        # Ensure audio is in the right format for Whisper
        if len(audio_data) < 1600:  # Less than 0.1 seconds at 16kHz
            return ""
        
        # Resample to 16kHz if needed (Whisper expects 16kHz)
        sample_rate = 8000  # Twilio uses 8kHz
        target_rate = 16000
        
        if sample_rate != target_rate:
            # Simple upsampling by repeating samples
            audio_data = np.repeat(audio_data, target_rate // sample_rate)
        
        # Process with Whisper
        inputs = whisper_processor(
            audio_data, 
            sampling_rate=target_rate, 
            return_tensors="pt"
        )
        
        with torch.no_grad():
            predicted_ids = whisper_model.generate(inputs["input_features"])
            transcription = whisper_processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
        
        return transcription.strip()
        
    except Exception as e:
        print(f"Error in transcription: {e}")
        return ""

def generate_ai_response(call_data: CallData, user_text: str) -> str:
    """Generate AI response using conversation model"""
    try:
        # Add user input to conversation history
        call_data.conversation_history.append(f"Human: {user_text}")
        
        # Create conversation context
        conversation_context = "\n".join(call_data.conversation_history[-6:])  # Last 6 exchanges
        conversation_context += "\nAssistant:"
        
        # Tokenize input
        inputs = conversation_tokenizer.encode(
            conversation_context, 
            return_tensors='pt'
        )
        
        # Generate response
        with torch.no_grad():
            outputs = conversation_model.generate(
                inputs,
                max_length=inputs.shape[1] + 50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=conversation_tokenizer.pad_token_id,
                eos_token_id=conversation_tokenizer.eos_token_id
            )
        
        # Decode response
        generated_text = conversation_tokenizer.decode(
            outputs[0][inputs.shape[1]:], 
            skip_special_tokens=True
        )
        
        # Clean up response
        ai_response = generated_text.split('\n')[0].strip()
        
        if not ai_response:
            ai_response = "I understand. Could you tell me more about that?"
        
        # Add to conversation history
        call_data.conversation_history.append(f"Assistant: {ai_response}")
        
        # Keep conversation history manageable
        if len(call_data.conversation_history) > 10:
            call_data.conversation_history = call_data.conversation_history[-8:]
        
        return ai_response
        
    except Exception as e:
        print(f"Error generating AI response: {e}")
        return "I'm sorry, I didn't catch that. Could you please repeat?"

def send_tts_response(call_data: CallData, text: str):
    """Send TTS response back to caller via Twilio"""
    try:
        # For now, we'll use Twilio's built-in TTS
        # In a production system, you might want to use a local TTS model
        
        # Create TwiML response
        response = VoiceResponse()
        response.say(text, voice='alice', language='en-US')
        
        # Send the response (this would typically be handled differently in a real implementation)
        # For WebSocket communication, we'd need to send specific Twilio Media Stream commands
        print(f"Would send TTS response: {text}")
        
    except Exception as e:
        print(f"Error sending TTS response: {e}")

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'active_calls': len(active_calls),
        'models_loaded': all([
            whisper_processor is not None,
            whisper_model is not None,
            conversation_tokenizer is not None,
            conversation_model is not None
        ])
    }

if __name__ == '__main__':
    # Initialize models before starting the server
    initialize_models()
    
    # Start the Flask-SocketIO server
    port = int(os.environ.get('PORT', 3000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)