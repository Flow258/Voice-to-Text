Jumpstart Voice-to-Text Assistant
A real-time voice transcription and intelligent NLP response system built for customer support in the fashion retail industry. This system captures audio from customer and agent microphones, transcribes speech using Whisper, and applies advanced NLP for sentiment analysis, intent classification, and auto-responses using Hugging Face Transformers.

Features
Dual-channel transcription (customer and agent)

Real-time transcription via WebSocket

NLP processing (intent, sentiment, entity extraction)

Whisper integration (GPU/CPU supported)

Duplicate filtering, urgency detection, escalation logic

Installation
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/jumpstart-voice-assistant.git
cd jumpstart-voice-assistant
2. Set Up Python Environment
You can use venv or conda:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
3. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
If requirements.txt is not available, install manually:

bash
Copy
Edit
pip install fastapi uvicorn pyaudio numpy torch \
  transformers sentence-transformers openai-whisper \
  SpeechRecognition
Note: You may need to install system dependencies for pyaudio.
For Ubuntu:

bash
Copy
Edit
sudo apt install portaudio19-dev python3-pyaudio
For Windows:

bash
Copy
Edit
pip install pipwin
pipwin install pyaudio
Configuration
Adjust the transcription settings in your app code or via API parameters (e.g., microphone index, timeout thresholds, model type: tiny, base, small, medium, or large).

Make sure to check the index of your audio devices:

python
Copy
Edit
import speech_recognition as sr
print(sr.Microphone.list_microphone_names())
NLP Models Used
Intent Detection: facebook/bart-large-mnli

Sentiment Analysis: cardiffnlp/twitter-roberta-base-sentiment-latest

Embeddings: sentence-transformers for entity similarity

Running the Server
bash
Copy
Edit
python transcriber_api.py
This starts the FastAPI server at:

cpp
Copy
Edit
http://0.0.0.0:8000
WebSocket Usage
Connect to WebSocket for real-time transcription updates:

bash
Copy
Edit
ws://localhost:8000/ws/{session_id}
Each session will stream live transcription updates with speaker labels (customer or agent) and associated NLP metadata.

API Endpoints (Pluggable)
POST /start_session: Start a new transcription session

GET /sessions: List active sessions

GET /session/{id}: Fetch transcription logs

GET /audio-devices: (Optional) Get available input devices

Sample Use Case
Start the server.

Plug in two mics or configure audio sources.

Connect to WebSocket and stream audio.

View structured transcriptions and smart replies from the assistant.

Debugging Tips
Check torch.cuda.is_available() to see if GPU is enabled.

Set non_english=True if your inputs are not in English.

Use .en models for faster English-only transcription.

If microphone input fails, check system permissions or device indexes.

Tested On
Python 3.9+

Windows 10 / Ubuntu 22.04

Whisper v20231106

Transformers v4.41+

License
MIT License
