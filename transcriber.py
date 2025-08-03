#!/usr/bin/env python3

import os
import uuid
from datetime import datetime, timedelta, timezone
from threading import Thread, Lock
from queue import Queue
import time
from collections import deque
import asyncio
from typing import Dict, Optional, List, Any, Tuple
import json
import re
import hashlib

import numpy as np
import speech_recognition as sr
import whisper
import torch
import pyaudio
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import atexit

# Hugging Face imports
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering, pipeline,
    BertTokenizer, BertForSequenceClassification
)
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

# Pydantic models
class TranscriptionConfig(BaseModel):
    model: str = "medium"
    non_english: bool = False
    energy_threshold: int = 300
    record_timeout: float = 3.0
    phrase_timeout: float = 2.0
    min_phrase_length: float = 0.5  # Minimum phrase length in seconds
    customer_device_index: Optional[int] = None
    agent_device_index: Optional[int] = None
    enable_nlp: bool = True
    auto_response: bool = False
    duplicate_threshold: float = 0.8  # Similarity threshold for duplicate detection

class SessionInfo(BaseModel):
    session_id: str
    status: str
    start_time: str
    config: TranscriptionConfig
    transcription_count: int

class TranscriptionLine(BaseModel):
    timestamp: str
    speaker: str
    text: str
    session_id: str
    is_final: Optional[bool] = True
    intent: Optional[str] = None
    confidence: Optional[float] = None
    entities: Optional[Dict[str, Any]] = None
    suggested_response: Optional[str] = None
    priority: Optional[str] = None
    escalation_needed: Optional[bool] = False
    audio_hash: Optional[str] = None  # For duplicate detection

class NLPAnalysis(BaseModel):
    intent: str
    confidence: float
    entities: Dict[str, Any]
    sentiment: str
    sentiment_score: float
    suggested_response: str
    priority: str
    escalation_needed: bool

class JumpstartNLPProcessor:
    """Optimized NLP processor for Jumpstart Fashion Retailer customer support"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Jumpstart NLP Processor using device: {self.device}")
        
        # Initialize models with error handling
        self._load_models()
        
        # Jumpstart-specific response templates
        self.response_templates = {
            "exchange_request": [
                "I understand you need to exchange your item. Let me help you with that process.",
                "I can assist you with the exchange. Could you please provide your order number?",
                "I'm sorry for the inconvenience. I'll help you process the exchange right away.",
                "I've initiated an exchange request for you. You'll receive a return label via email."
            ],
            "return_request": [
                "I can help you process a return. Let me check your order details.",
                "I understand you'd like to return an item. I'll assist you with that.",
                "Let me help you with the return process. May I have your order information?"
            ],
            "refund_request": [
                "I can help you with a refund. Let me check your order details.",
                "I understand you'd like to process a refund. I'll assist you with that.",
                "Let me help you with the refund process. May I have your order information?"
            ],
            "size_inquiry": [
                "I can help you with sizing information. What item are you looking for?",
                "Let me check our size guide for you. Which product are you interested in?",
                "I'd be happy to help you find the right size."
            ],
            "order_status": [
                "Let me check your order status for you right away.",
                "I can track your order. Please provide your order number.",
                "I'll look up your order information immediately."
            ],
            "shipping_inquiry": [
                "I can provide you with shipping information. What would you like to know?",
                "Let me check shipping options and timelines for you.",
                "I'll help you with shipping details right away."
            ],
            "product_availability": [
                "Let me check product availability for you.",
                "I can help you find that item. What are you looking for?",
                "I'll check our inventory for that product."
            ],
            "complaint": [
                "I sincerely apologize for this experience. Let me help resolve this issue immediately.",
                "I'm sorry to hear about this problem. I'll escalate this to ensure it's resolved properly.",
                "Thank you for bringing this to our attention. I'll make sure we address this right away."
            ],
            "compliment": [
                "Thank you so much for your kind words! We really appreciate your feedback.",
                "I'm delighted to hear you're happy with our service. Thank you for choosing Jumpstart!",
                "That's wonderful to hear! I'll make sure to share your positive feedback with the team."
            ],
            "payment_issue": [
                "I can help you resolve this payment issue right away.",
                "Let me assist you with your payment concern. I'll check your account details.",
                "I understand this is frustrating. I'll help resolve your payment issue immediately."
            ],
            "greeting": [
                "Hello! Welcome to Jumpstart Fashion. How can I assist you today?",
                "Good day! Thank you for contacting Jumpstart. I'm here to help you.",
                "Thank you for choosing Jumpstart. How may I assist you with your fashion needs today?"
            ],
            "general_inquiry": [
                "I'm here to help you with that. Could you provide more details?",
                "Let me assist you with your inquiry. What specific information do you need?",
                "I'd be happy to help. Can you tell me more about what you're looking for?"
            ]
        }
        
        # Improved entity patterns
        self.entity_patterns = {
            "order_number": r"(?:order|order number|#|number)\s*([A-Z0-9]{6,12})",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b",
            "product_size": r"(?:size|sz)\s*([XSMLXL]+|[0-9]+|small|medium|large|extra small|extra large)",
            "clothing_item": r"(?:shirt|dress|pants|jeans|jacket|shoes|sneakers|boots|top|blouse|skirt|shorts|sweater|hoodie|cardigan)",
            "color": r"(?:black|white|red|blue|green|yellow|pink|purple|orange|brown|gray|grey|navy|beige|tan)",
            "amount": r"\$?(\d+(?:\.\d{2})?)",
        }

    def _load_models(self):
        """Load NLP models with better error handling"""
        try:
            print("Loading NLP models for fashion retail...")
            
            # Use lightweight models for better performance
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if self.device == "cuda" else -1,
                return_all_scores=False
            )
            
            # Zero-shot classification for intent detection
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.device == "cuda" else -1
            )
            
            print("NLP models loaded successfully!")
            
        except Exception as e:
            print(f"Warning: Error loading advanced NLP models: {e}")
            print("Falling back to basic models...")
            try:
                self.sentiment_pipeline = pipeline("sentiment-analysis")
                self.classifier = None
            except Exception as e2:
                print(f"Error loading fallback models: {e2}")
                self.sentiment_pipeline = None
                self.classifier = None

    def classify_intent(self, text: str) -> Tuple[str, float]:
        """Improved intent classification with confidence scoring"""
        try:
            if not self.classifier:
                # Simple keyword-based fallback
                text_lower = text.lower()
                if any(word in text_lower for word in ["exchange", "swap", "change"]):
                    return "exchange_request", 0.8
                elif any(word in text_lower for word in ["return", "send back"]):
                    return "return_request", 0.8
                elif any(word in text_lower for word in ["refund", "money back"]):
                    return "refund_request", 0.8
                elif any(word in text_lower for word in ["size", "sizing", "fit"]):
                    return "size_inquiry", 0.7
                elif any(word in text_lower for word in ["order", "tracking", "status"]):
                    return "order_status", 0.7
                elif any(word in text_lower for word in ["hello", "hi", "hey"]):
                    return "greeting", 0.9
                elif any(word in text_lower for word in ["complaint", "problem", "issue", "wrong", "bad"]):
                    return "complaint", 0.7
                else:
                    return "general_inquiry", 0.5
            
            # Advanced classification
            candidate_labels = [
                "exchange_request", "return_request", "refund_request", "size_inquiry",
                "order_status", "shipping_inquiry", "product_availability", "complaint",
                "compliment", "payment_issue", "greeting", "general_inquiry"
            ]
            
            result = self.classifier(text, candidate_labels)
            intent = result['labels'][0]
            confidence = result['scores'][0]
            
            # Apply confidence threshold
            if confidence < 0.3:
                return "general_inquiry", confidence
            
            return intent, confidence
            
        except Exception as e:
            print(f"Intent classification error: {e}")
            return "general_inquiry", 0.5

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities with improved accuracy"""
        entities = {}
        
        try:
            for entity_type, pattern in self.entity_patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    entities[entity_type] = matches
            
            # Context-based entity extraction
            text_lower = text.lower()
            
            # Urgency detection
            urgency_indicators = ["urgent", "asap", "immediately", "right away", "emergency", "now"]
            if any(word in text_lower for word in urgency_indicators):
                entities["urgency"] = ["high"]
            
            # Dissatisfaction detection
            dissatisfaction_words = ["terrible", "awful", "horrible", "worst", "disgusted", "furious", "angry"]
            if any(word in text_lower for word in dissatisfaction_words):
                entities["dissatisfaction_level"] = ["high"]
            
        except Exception as e:
            print(f"Entity extraction error: {e}")
            
        return entities

    def analyze_sentiment(self, text: str) -> Tuple[str, float]:
        """Analyze sentiment with error handling"""
        try:
            if not self.sentiment_pipeline:
                return "neutral", 0.5
                
            result = self.sentiment_pipeline(text)[0]
            sentiment = result['label'].lower()
            score = result['score']
            
            # Normalize sentiment labels
            if sentiment in ['positive', 'pos', 'label_2']:
                sentiment = 'positive'
            elif sentiment in ['negative', 'neg', 'label_0']:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
                
            return sentiment, score
            
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return "neutral", 0.5

    def generate_response(self, intent: str, entities: Dict[str, Any], sentiment: str) -> str:
        """Generate contextual response"""
        try:
            templates = self.response_templates.get(intent, self.response_templates["general_inquiry"])
            base_response = np.random.choice(templates)
            
            # Customize based on context
            if sentiment == "negative" and "apologize" not in base_response.lower():
                base_response = "I apologize for any inconvenience. " + base_response
            
            # Add entity-specific information
            if "order_number" in entities:
                order_num = entities["order_number"][0]
                base_response += f" I can see this is regarding order #{order_num}."
            
            if entities.get("urgency") == ["high"]:
                base_response = "I understand this is urgent. " + base_response
            
            return base_response
            
        except Exception as e:
            print(f"Response generation error: {e}")
            return "Thank you for contacting Jumpstart Fashion. How can I assist you today?"

    def determine_priority(self, intent: str, sentiment: str, confidence: float, entities: Dict[str, Any]) -> str:
        """Determine priority with improved logic"""
        # High priority conditions
        if sentiment == "negative" and confidence > 0.6:
            return "high"
        if intent in ["complaint", "refund_request", "payment_issue"]:
            return "high"
        if entities.get("urgency") == ["high"]:
            return "high"
        if entities.get("dissatisfaction_level") == ["high"]:
            return "high"
        
        # Medium priority
        if intent in ["exchange_request", "return_request", "order_status"]:
            return "medium"
        
        return "low"

    def needs_escalation(self, intent: str, sentiment: str, entities: Dict[str, Any]) -> bool:
        """Determine escalation needs"""
        return (
            entities.get("dissatisfaction_level") == ["high"] or
            (intent == "complaint" and sentiment == "negative") or
            intent == "payment_issue"
        )

    def process_text(self, text: str) -> NLPAnalysis:
        """Main processing method with error handling"""
        try:
            intent, confidence = self.classify_intent(text)
            entities = self.extract_entities(text)
            sentiment, sentiment_score = self.analyze_sentiment(text)
            suggested_response = self.generate_response(intent, entities, sentiment)
            priority = self.determine_priority(intent, sentiment, confidence, entities)
            escalation_needed = self.needs_escalation(intent, sentiment, entities)
            
            return NLPAnalysis(
                intent=intent,
                confidence=confidence,
                entities=entities,
                sentiment=sentiment,
                sentiment_score=sentiment_score,
                suggested_response=suggested_response,
                priority=priority,
                escalation_needed=escalation_needed
            )
            
        except Exception as e:
            print(f"NLP processing error: {e}")
            return NLPAnalysis(
                intent="general_inquiry",
                confidence=0.5,
                entities={},
                sentiment="neutral",
                sentiment_score=0.5,
                suggested_response="Thank you for contacting Jumpstart Fashion. How can I assist you today?",
                priority="low",
                escalation_needed=False
            )

class TranscriptionSession:
    def __init__(self, session_id: str, config: TranscriptionConfig, event_loop: asyncio.AbstractEventLoop):
        self.session_id = session_id
        self.config = config
        self.event_loop = event_loop
        self.status = "initialized"
        self.start_time = datetime.now(timezone.utc)
        self.transcription = deque(maxlen=1000)
        self.websocket_clients = set()
        self.customer_queue = Queue()
        self.agent_queue = Queue()
        
        # NLP processor
        self.nlp_processor = JumpstartNLPProcessor() if config.enable_nlp else None
        
        # Audio management
        self.customer_audio_buffer = np.array([], dtype=np.float32)
        self.agent_audio_buffer = np.array([], dtype=np.float32)
        self.customer_last_transcription_time = None
        self.agent_last_transcription_time = None
        
        # Duplicate detection
        self.recent_transcriptions = deque(maxlen=10)
        
        # Current speaking state
        self.current_customer_line = None
        self.current_agent_line = None
        
        # Threading
        self._stop_event = False
        self._transcription_thread = None
        self._agent_audio_thread = None
        self.recorder = None
        self.pa = None
        self.customer_source = None
        self.agent_stream = None
        self.audio_model = None
        self.lock = Lock()

    def _calculate_audio_hash(self, audio_data: np.ndarray) -> str:
        """Calculate hash for audio data to detect duplicates"""
        try:
            # Normalize and quantize audio for consistent hashing
            normalized = (audio_data * 32767).astype(np.int16)
            return hashlib.md5(normalized.tobytes()).hexdigest()[:16]
        except:
            return str(time.time())

    def _is_duplicate_transcription(self, text: str, audio_hash: str) -> bool:
        """Check if transcription is a duplicate"""
        # Check text similarity
        for recent_text, recent_hash in self.recent_transcriptions:
            if recent_hash == audio_hash:
                return True
            # Simple text similarity check
            if len(text) > 10 and text in recent_text:
                return True
            if len(recent_text) > 10 and recent_text in text:
                return True
        return False

    def _is_valid_transcription(self, text: str) -> bool:
        """Filter out invalid transcriptions"""
        if not text or len(text.strip()) < 3:
            return False
        
        # Filter out common noise/artifacts
        noise_patterns = [
            "this is a conversation between",
            "thank you for watching",
            "subscribe",
            "like and subscribe",
            "please subscribe",
            "thanks for watching"
        ]
        
        text_lower = text.lower().strip()
        
        # Check for noise patterns
        if any(pattern in text_lower for pattern in noise_patterns):
            return False
        
        # Check for repetitive characters
        if len(set(text_lower.replace(' ', ''))) < 3:
            return False
        
        # Check for overly repetitive words
        words = text_lower.split()
        if len(words) > 1:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.5:  # Less than 50% unique words
                return False
        
        return True

    def initialize_audio(self):
        """Initialize audio with better error handling"""
        try:
            self.recorder = sr.Recognizer()
            self.recorder.energy_threshold = self.config.energy_threshold
            self.recorder.dynamic_energy_threshold = False
            self.pa = pyaudio.PyAudio()
            
            # Setup microphones
            customer_success = self._setup_customer_microphone()
            agent_success = self._setup_agent_microphone()
            
            if not customer_success:
                print("Warning: Customer audio source not properly configured")
            
            # Load Whisper model
            model_name = self.config.model
            if not self.config.non_english and self.config.model != "large":
                model_name = f"{self.config.model}.en"
            
            print(f"Loading Whisper model: {model_name}")
            self.audio_model = whisper.load_model(model_name)
            
            return True
            
        except Exception as e:
            print(f"Error initializing audio: {e}")
            self.cleanup_audio()
            return False

    def _setup_customer_microphone(self):
        """Setup customer microphone with fallback"""
        try:
            mic_names = sr.Microphone.list_microphone_names()
            
            if self.config.customer_device_index is not None:
                if 0 <= self.config.customer_device_index < len(mic_names):
                    self.customer_source = sr.Microphone(
                        sample_rate=16000, 
                        device_index=self.config.customer_device_index
                    )
                    print(f"Customer device: {mic_names[self.config.customer_device_index]}")
                else:
                    print(f"Invalid customer device index: {self.config.customer_device_index}")
                    return False
            else:
                # Auto-detect or use default
                self.customer_source = sr.Microphone(sample_rate=16000)
                print("Using default customer microphone")
            
            # Adjust for ambient noise
            if self.customer_source:
                with self.customer_source:
                    self.recorder.adjust_for_ambient_noise(self.customer_source, duration=0.5)
                return True
            
        except Exception as e:
            print(f"Customer microphone setup error: {e}")
            return False

    def _setup_agent_microphone(self):
        """Setup agent microphone with better detection"""
        try:
            agent_device = self.config.agent_device_index
            
            if agent_device is None:
                # Try to find a suitable agent microphone
                for i in range(self.pa.get_device_count()):
                    try:
                        device_info = self.pa.get_device_info_by_index(i)
                        if (device_info['maxInputChannels'] > 0 and 
                            'microphone' in device_info['name'].lower()):
                            agent_device = i
                            break
                    except:
                        continue
            
            if agent_device is not None:
                try:
                    self.agent_stream = self.pa.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        input_device_index=agent_device,
                        frames_per_buffer=1024
                    )
                    device_info = self.pa.get_device_info_by_index(agent_device)
                    print(f"Agent device: {device_info['name']}")
                    return True
                except Exception as e:
                    print(f"Failed to open agent device {agent_device}: {e}")
            
            print("No suitable agent microphone found")
            return False
            
        except Exception as e:
            print(f"Agent microphone setup error: {e}")
            return False

    def cleanup_audio(self):
        """Clean up audio resources"""
        try:
            if self.agent_stream:
                self.agent_stream.stop_stream()
                self.agent_stream.close()
                self.agent_stream = None
            if self.pa:
                self.pa.terminate()
                self.pa = None
        except Exception as e:
            print(f"Error cleaning up audio: {e}")

    def _customer_callback(self, _, audio: sr.AudioData):
        """Customer audio callback with validation"""
        if not self._stop_event:
            try:
                audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0
                if len(audio_data) > 0 and np.max(np.abs(audio_data)) > 0.001:  # Minimum volume threshold
                    self.customer_queue.put(audio_data)
            except Exception as e:
                print(f"Customer audio callback error: {e}")

    def _agent_audio_thread_func(self):
        """Agent audio thread with error handling"""
        if not self.agent_stream:
            return
            
        while not self._stop_event:
            try:
                data = self.agent_stream.read(1024, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                if len(audio_data) > 0 and np.max(np.abs(audio_data)) > 0.001:
                    self.agent_queue.put(audio_data)
            except Exception as e:
                print(f"Agent audio error: {e}")
                time.sleep(0.1)

    def _transcribe_audio_chunk(self, audio_chunk: np.ndarray) -> Optional[str]:
        """Transcribe audio with improved filtering"""
        try:
            # Validate audio length
            if len(audio_chunk) < int(16000 * self.config.min_phrase_length):
                return None
            
            # Check volume
            max_amplitude = np.max(np.abs(audio_chunk))
            if max_amplitude < 0.005:  # Increased minimum volume
                return None
            
            # Normalize audio
            if max_amplitude > 0.9:
                audio_chunk = audio_chunk * (0.9 / max_amplitude)
            
            # Transcribe with Whisper
            result = self.audio_model.transcribe(
                audio_chunk,
                fp16=torch.cuda.is_available(),
                language='en' if not self.config.non_english else None,
                initial_prompt="Customer service conversation about fashion, clothing, returns, and exchanges.",
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0
            )
            
            text = result['text'].strip()
            
            # Validate transcription
            if not self._is_valid_transcription(text):
                return None
                
            return text
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return None

    def _process_nlp(self, text: str, speaker: str) -> Dict[str, Any]:
        """Process NLP only for customer speech"""
        if not self.nlp_processor or speaker != "customer":
            return {}
        
        try:
            analysis = self.nlp_processor.process_text(text)
            return {
                "intent": analysis.intent,
                "confidence": analysis.confidence,
                "entities": analysis.entities,
                "suggested_response": analysis.suggested_response,
                "priority": analysis.priority,
                "escalation_needed": analysis.escalation_needed
            }
        except Exception as e:
            print(f"NLP processing error: {e}")
            return {}

    def _process_speaker_audio(self, speaker: str):
        """Process audio for speaker with duplicate detection"""
        queue = self.customer_queue if speaker == "customer" else self.agent_queue
        
        if queue.empty():
            return
        
        # Collect audio chunks
        new_audio = []
        while not queue.empty():
            new_audio.append(queue.get())
        
        if not new_audio:
            return
        
        # Update buffer
        buffer = getattr(self, f"{speaker}_audio_buffer")
        buffer = np.concatenate([buffer] + new_audio)
        setattr(self, f"{speaker}_audio_buffer", buffer)
        
        # Check timing
        now = datetime.now(timezone.utc)
        last_time = getattr(self, f"{speaker}_last_transcription_time")
        current_line = getattr(self, f"current_{speaker}_line")
        
        should_finalize = (
            last_time and 
            (now - last_time).total_seconds() > self.config.phrase_timeout
        )
        
        # Transcribe if buffer is long enough
        min_buffer_length = int(16000 * self.config.min_phrase_length)
        if len(buffer) >= min_buffer_length:
            transcribed_text = self._transcribe_audio_chunk(buffer)
            
            if transcribed_text:
                # Check for duplicates
                audio_hash = self._calculate_audio_hash(buffer)
                if self._is_duplicate_transcription(transcribed_text, audio_hash):
                    print(f"Filtered duplicate {speaker} transcription")
                    return
                
                # Add to recent transcriptions
                self.recent_transcriptions.append((transcribed_text, audio_hash))
                
                timestamp = now.strftime('%H:%M:%S')
                
                # Process NLP
                nlp_results = self._process_nlp(transcribed_text, speaker)
                
                if should_finalize or not current_line:
                    # Create new transcription line
                    transcription_line = TranscriptionLine(
                        timestamp=timestamp,
                        speaker=speaker,
                        text=transcribed_text,
                        session_id=self.session_id,
                        is_final=should_finalize,
                        audio_hash=audio_hash,
                        **nlp_results
                    )
                    
                    with self.lock:
                        self.transcription.append(transcription_line)
                    
                    setattr(self, f"current_{speaker}_line", transcription_line)
                    
                    # Log with NLP info
                    log_msg = f"[{speaker.upper()}] {transcribed_text}"
                    if nlp_results.get("intent"):
                        log_msg += f" | Intent: {nlp_results['intent']} ({nlp_results.get('confidence', 0):.2f})"
                        log_msg += f" | Priority: {nlp_results.get('priority', 'low').upper()}"
                        if nlp_results.get("escalation_needed"):
                            log_msg += " | 🚨 ESCALATION"
                    print(log_msg)
                    
                else:
                    # Update existing line
                    current_line.text = transcribed_text
                    current_line.timestamp = timestamp
                    current_line.is_final = should_finalize
                    current_line.audio_hash = audio_hash
                    
                    # Update NLP results
                    for key, value in nlp_results.items():
                        setattr(current_line, key, value)
                
                # Broadcast update
                self._broadcast_transcription(getattr(self, f"current_{speaker}_line"))
                
                # Update timing
                setattr(self, f"{speaker}_last_transcription_time", now)
                
                # Clear buffer if finalizing
                if should_finalize:
                    setattr(self, f"{speaker}_audio_buffer", np.array([], dtype=np.float32))
                    setattr(self, f"current_{speaker}_line", None)
                else:
                    # Keep recent audio for context
                    max_buffer_length = 48000  # 3 seconds
                    if len(buffer) > max_buffer_length:
                        setattr(self, f"{speaker}_audio_buffer", buffer[-max_buffer_length:])

    def _transcription_loop(self):
        """Optimized transcription loop with better error handling"""
        while not self._stop_event:
            try:
                # Process customer audio (higher priority)
                self._process_speaker_audio("customer")
                
                # Process agent audio
                if self.agent_stream:
                    self._process_speaker_audio("agent")
                    
                time.sleep(0.1)  # Prevent excessive CPU usage
                
            except Exception as e:
                print(f"Transcription loop error: {e}")
                time.sleep(1)

    def _broadcast_transcription(self, transcription_line: TranscriptionLine):
        """Broadcast transcription to WebSocket clients"""
        if not self.websocket_clients:
            return
            
        try:
            message = transcription_line.model_dump()
            message["company"] = "Jumpstart Fashion"
            message["version"] = "3.1.0-optimized"
            
            disconnected_clients = set()
            
            for websocket in self.websocket_clients:
                try:
                    asyncio.run_coroutine_threadsafe(
                        websocket.send_json(message), 
                        self.event_loop
                    )
                except Exception as e:
                    print(f"WebSocket send error: {e}")
                    disconnected_clients.add(websocket)
            
            # Clean up disconnected clients
            self.websocket_clients -= disconnected_clients
            
        except Exception as e:
            print(f"Broadcast error: {e}")

    def start(self):
        """Start the transcription session"""
        if not self.initialize_audio():
            self.status = "failed"
            raise HTTPException(status_code=500, detail="Failed to initialize audio")
        
        self.status = "running"
        self._stop_event = False
        
        # Start customer audio recording
        if self.customer_source:
            self.recorder.listen_in_background(
                self.customer_source, 
                self._customer_callback, 
                phrase_time_limit=self.config.record_timeout
            )
            print("✓ Customer audio recording started")
        
        # Start agent audio thread
        if self.agent_stream:
            self._agent_audio_thread = Thread(target=self._agent_audio_thread_func, daemon=True)
            self._agent_audio_thread.start()
            print("✓ Agent audio recording started")
        
        # Start transcription thread
        self._transcription_thread = Thread(target=self._transcription_loop, daemon=True)
        self._transcription_thread.start()
        print("✓ Transcription engine started")
        
        return True

    def stop(self):
        """Stop the transcription session"""
        self.status = "stopped"
        self._stop_event = True
        
        # Wait for threads to finish gracefully
        threads = [self._transcription_thread, self._agent_audio_thread]
        for thread in threads:
            if thread and thread.is_alive():
                thread.join(timeout=3)
        
        self.cleanup_audio()
        print("✓ Session stopped successfully")

    def get_transcription(self, limit: int = 100) -> List[TranscriptionLine]:
        """Get recent transcription lines"""
        with self.lock:
            return list(self.transcription)[-limit:]

    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        with self.lock:
            transcription_list = list(self.transcription)
        
        stats = {
            "total_lines": len(transcription_list),
            "customer_statements": sum(1 for line in transcription_list if line.speaker == "customer"),
            "agent_responses": sum(1 for line in transcription_list if line.speaker == "agent"),
            "high_priority_items": sum(1 for line in transcription_list if line.priority == "high"),
            "escalations_needed": sum(1 for line in transcription_list if line.escalation_needed),
            "session_duration": str(datetime.now(timezone.utc) - self.start_time),
            "avg_confidence": 0.0
        }
        
        # Calculate average confidence for customer statements
        customer_confidences = [
            line.confidence for line in transcription_list 
            if line.speaker == "customer" and line.confidence is not None
        ]
        if customer_confidences:
            stats["avg_confidence"] = sum(customer_confidences) / len(customer_confidences)
        
        return stats

    def save_transcription(self) -> str:
        """Save transcription to file with enhanced formatting"""
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        filename = f"jumpstart_conversation_{self.session_id}_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                # Header
                f.write("=" * 80 + "\n")
                f.write("JUMPSTART FASHION RETAILER - Customer Service Transcription\n")
                f.write("AI-Powered Voice Assistant with Advanced NLP Analysis\n")
                f.write("=" * 80 + "\n\n")
                
                # Session info
                f.write(f"Session ID: {self.session_id}\n")
                f.write(f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
                f.write(f"Duration: {datetime.now(timezone.utc) - self.start_time}\n")
                f.write(f"NLP Enabled: {self.config.enable_nlp}\n\n")
                
                # Statistics
                stats = self.get_session_stats()
                f.write("SESSION STATISTICS:\n")
                f.write("-" * 40 + "\n")
                for key, value in stats.items():
                    formatted_key = key.replace('_', ' ').title()
                    f.write(f"{formatted_key}: {value}\n")
                f.write("\n" + "=" * 80 + "\n\n")
                
                # Transcription
                f.write("CONVERSATION TRANSCRIPT:\n")
                f.write("-" * 40 + "\n\n")
                
                with self.lock:
                    for i, line in enumerate(self.transcription, 1):
                        # Speaker header
                        speaker_icon = "🛍️" if line.speaker == "customer" else "👩‍💼"
                        speaker_name = line.speaker.upper()
                        
                        f.write(f"{i:03d}. {speaker_icon} {speaker_name} [{line.timestamp}]")
                        
                        # Add metadata for customer lines
                        if line.speaker == "customer" and line.intent:
                            f.write(f" | {line.priority.upper()}")
                            if line.escalation_needed:
                                f.write(" | 🚨 ESCALATION")
                        
                        f.write("\n")
                        f.write(f"     {line.text}")
                        
                        if not line.is_final:
                            f.write(" [LIVE]")
                        
                        f.write("\n")
                        
                        # NLP analysis for customer statements
                        if line.speaker == "customer" and line.intent:
                            f.write(f"     🎯 Intent: {line.intent} (confidence: {line.confidence:.2f})\n")
                            
                            if line.entities:
                                entities_str = ", ".join([f"{k}: {v}" for k, v in line.entities.items()])
                                f.write(f"     📋 Entities: {entities_str}\n")
                            
                            if line.suggested_response:
                                f.write(f"     💡 Suggested Response: {line.suggested_response}\n")
                        
                        f.write("\n" + "-" * 60 + "\n\n")
                
                # Footer
                f.write("=" * 80 + "\n")
                f.write("End of Transcription\n")
                f.write("Generated by Jumpstart Fashion AI Assistant\n")
                f.write("=" * 80 + "\n")
            
            print(f"✓ Transcription saved to {filename}")
            return filename
            
        except Exception as e:
            print(f"Error saving transcription: {e}")
            return None

# FastAPI Application Setup
app = FastAPI(
    title="Jumpstart Fashion Voice-to-Text Assistant API", 
    version="3.1.0-optimized",
    description="Optimized AI-powered customer service voice assistant for Jumpstart Fashion"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global session management
sessions: Dict[str, TranscriptionSession] = {}
sessions_lock = Lock()

@app.get("/")
async def root():
    return {
        "message": "Jumpstart Fashion Voice-to-Text Assistant API", 
        "version": "3.1.0-optimized",
        "company": "Jumpstart Fashion",
        "status": "operational",
        "features": [
            "Real-time transcription with duplicate filtering",
            "Advanced intent classification", 
            "Smart entity extraction",
            "Automated response suggestions",
            "Priority-based routing",
            "Escalation detection",
            "Session analytics"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_sessions": len(sessions),
        "version": "3.1.0-optimized"
    }

@app.get("/devices")
async def list_devices():
    """List available audio devices with recommendations"""
    try:
        # Speech recognition devices
        sr_mics = []
        for i, name in enumerate(sr.Microphone.list_microphone_names()):
            device_info = {
                "index": i,
                "name": name,
                "type": "input",
                "recommended_for": "customer" if any(keyword in name.lower() 
                    for keyword in ["stereo", "mix", "speakers", "what u hear"]) else "agent"
            }
            sr_mics.append(device_info)
        
        # PyAudio devices
        pa = pyaudio.PyAudio()
        pa_devices = []
        for i in range(pa.get_device_count()):
            try:
                device_info = pa.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    pa_devices.append({
                        "index": i,
                        "name": device_info['name'],
                        "type": "input",
                        "channels": device_info['maxInputChannels'],
                        "sample_rate": device_info['defaultSampleRate'],
                        "recommended_for": "agent" if "microphone" in device_info['name'].lower() else "system"
                    })
            except:
                continue
        pa.terminate()
        
        return {
            "company": "Jumpstart Fashion",
            "speech_recognition_devices": sr_mics,
            "pyaudio_devices": pa_devices,
            "recommendations": {
                "customer_audio": "Use 'Stereo Mix', 'What U Hear', or similar to capture customer audio from calls/VoIP",
                "agent_audio": "Use built-in microphone, headset mic, or USB microphone for agent voice",
                "best_practice": "Test different devices to find the optimal setup for your environment"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing devices: {str(e)}")

@app.post("/sessions", response_model=SessionInfo)
async def create_session(config: TranscriptionConfig):
    """Create a new customer service session"""
    session_id = str(uuid.uuid4())
    loop = asyncio.get_running_loop()
    
    with sessions_lock:
        session = TranscriptionSession(session_id, config, loop)
        sessions[session_id] = session
    
    print(f"✓ Created session {session_id} (NLP: {config.enable_nlp})")
    return SessionInfo(
        session_id=session_id,
        status=session.status,
        start_time=session.start_time.isoformat(),
        config=config,
        transcription_count=0
    )

@app.post("/sessions/{session_id}/start")
async def start_session(session_id: str):
    """Start a transcription session"""
    with sessions_lock:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        session = sessions[session_id]
    
    if session.status == "running":
        return {"message": "Session already running", "session_id": session_id}
    
    try:
        success = session.start()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to start session")
        
        return {
            "message": "Session started successfully",
            "session_id": session_id,
            "company": "Jumpstart Fashion",
            "nlp_enabled": session.config.enable_nlp,
            "status": "running"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start session: {str(e)}")

@app.post("/sessions/{session_id}/stop")
async def stop_session(session_id: str):
    """Stop a transcription session"""
    with sessions_lock:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        session = sessions[session_id]
    
    try:
        session.stop()
        return {
            "message": "Session stopped successfully",
            "session_id": session_id,
            "status": "stopped"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error stopping session: {str(e)}")

@app.get("/sessions/{session_id}/transcription")
async def get_transcription(session_id: str, limit: int = 100):
    """Get transcription with analytics"""
    with sessions_lock:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        session = sessions[session_id]
    
    transcription = session.get_transcription(limit)
    stats = session.get_session_stats()
    
    return {
        "session_id": session_id,
        "company": "Jumpstart Fashion",
        "transcription": [line.model_dump() for line in transcription],
        "statistics": stats,
        "total_retrieved": len(transcription)
    }

@app.get("/sessions/{session_id}/stats")
async def get_session_stats(session_id: str):
    """Get detailed session statistics"""
    with sessions_lock:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        session = sessions[session_id]
    
    return {
        "session_id": session_id,
        "company": "Jumpstart Fashion",
        "statistics": session.get_session_stats(),
        "status": session.status,
        "config": session.config.model_dump()
    }

@app.post("/sessions/{session_id}/save")
async def save_transcription(session_id: str):
    """Save transcription to file"""
    with sessions_lock:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        session = sessions[session_id]
    
    try:
        filename = session.save_transcription()
        if filename:
            return {
                "message": "Transcription saved successfully",
                "filename": filename,
                "session_id": session_id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to save transcription")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving transcription: {str(e)}")

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    with sessions_lock:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        session.stop()
        del sessions[session_id]
    
    return {
        "message": "Session deleted successfully",
        "session_id": session_id
    }

@app.get("/sessions")
async def list_sessions():
    """List all sessions with stats"""
    with sessions_lock:
        session_list = []
        for sid, session in sessions.items():
            stats = session.get_session_stats()
            session_info = {
                "session_id": sid,
                "status": session.status,
                "start_time": session.start_time.isoformat(),
                "nlp_enabled": session.config.enable_nlp,
                "total_lines": stats["total_lines"],
                "high_priority": stats["high_priority_items"],
                "escalations": stats["escalations_needed"]
            }
            session_list.append(session_info)
    
    return {
        "company": "Jumpstart Fashion",
        "total_sessions": len(session_list),
        "sessions": session_list
    }

@app.websocket("/sessions/{session_id}/ws")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    
    with sessions_lock:
        if session_id not in sessions:
            await websocket.close(code=4004, reason="Session not found")
            return
        session = sessions[session_id]
    
    session.websocket_clients.add(websocket)
    print(f"✓ WebSocket client connected to session {session_id}")
    
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "welcome",
            "message": "Connected to Jumpstart Fashion Voice Assistant",
            "session_id": session_id,
            "version": "3.1.0-optimized",
            "features": ["Real-time transcription", "Intent detection", "Smart filtering"]
        })
        
        # Keep connection alive
        while True:
            try:
                # Wait for ping or close
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send keepalive ping
                await websocket.send_json({"type": "ping"})
            except Exception:
                break
                
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        session.websocket_clients.discard(websocket)
        print(f"✓ WebSocket client disconnected from session {session_id}")

@app.get("/demo")
async def demo_endpoint():
    """Demo endpoint showing capabilities"""
    return {
        "company": "Jumpstart Fashion",
        "version": "3.1.0-optimized",
        "improvements": [
            "✅ Fixed agent audio processing issues",
            "✅ Added duplicate transcription filtering", 
            "✅ Improved audio validation and noise filtering",
            "✅ Enhanced NLP confidence scoring",
            "✅ Better error handling and recovery",
            "✅ Added session analytics and statistics",
            "✅ Optimized memory usage and performance"
        ],
        "example_customer_interaction": {
            "input": "Hi, I need to exchange my jacket for a larger size",
            "processing": {
                "transcribed": "Hi, I need to exchange my jacket for a larger size",
                "intent": "exchange_request",
                "confidence": 0.92,
                "entities": {"clothing_item": ["jacket"], "product_size": ["larger size"]},
                "priority": "medium",
                "escalation_needed": False
            },
            "output": "I understand you need to exchange your item. Let me help you with that process. Could you please provide your order number?"
        }
    }

# Cleanup function
def cleanup_sessions():
    """Clean up all sessions on exit"""
    print("\n🧹 Cleaning up sessions...")
    with sessions_lock:
        for session_id, session in sessions.items():
            try:
                session.stop()
                print(f"✓ Stopped session {session_id}")
            except Exception as e:
                print(f"❌ Error stopping session {session_id}: {e}")
    print("✅ Cleanup completed")

if __name__ == "__main__":
    print("Starting Jumpstart Fashion Voice-to-Text Assistant API v3.1.0")
    print("Features: Optimized transcription, Smart filtering, Advanced NLP")
    print("Supporting 750 stores nationwide")
    
    # Register cleanup function
    atexit.register(cleanup_sessions)
    
    # Start the server
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
