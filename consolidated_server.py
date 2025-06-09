#!/usr/bin/env python3
"""
Consolidated AI Study Buddy API Server
Version: 3.2.1 (Deployment Ready)

A comprehensive FastAPI-based REST API that merges all server components
into a single, production-ready application.

Features:
- Multi-user chat sessions with persistent storage (Supabase).
- AI-powered responses using Groq API.
- Calm, patient, and professional English tutor personality.
- Context-aware conversations with injected page content.
- Session management (start/reuse) and history.
- Health monitoring and modern lifespan events for startup/shutdown.
"""

import os
import sys
import logging
import time
import uuid
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from contextlib import asynccontextmanager

# Third-party imports
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from supabase import create_client, Client
from groq import Groq
from dotenv import load_dotenv
import uvicorn

# Load environment variables from a .env file
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration class for the AI Chat Assistant"""
    
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_ANON_KEY = os.getenv('SUPABASE_ANON_KEY')
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    AI_MODEL = "llama3-8b-8192"
    
    @classmethod
    def validate(cls):
        """Validate that all required environment variables are set"""
        missing_vars = [var for var in ['SUPABASE_URL', 'SUPABASE_ANON_KEY', 'GROQ_API_KEY'] if not getattr(cls, var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        return True

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class ChatSession:
    """Data model for chat sessions, matching the Supabase table"""
    id: str
    student_name: str
    course_code: str
    session_name: str
    created_at: datetime
    context_content: Optional[str] = None
    
    @classmethod
    def create_new(cls, student_name: str, course_code: str, context_content: str, session_name: Optional[str] = None) -> 'ChatSession':
        session_id = str(uuid.uuid4())
        if not session_name:
            session_name = f"Session for {student_name} - {course_code}"
        return cls(id=session_id, student_name=student_name, course_code=course_code, session_name=session_name, created_at=datetime.now(), context_content=context_content)
    
    def to_dict(self) -> dict:
        return {'id': self.id, 'student_name': self.student_name, 'course_code': self.course_code, 'session_name': self.session_name, 'created_at': self.created_at.isoformat(), 'context_content': self.context_content}

@dataclass
class Message:
    """Data model for chat messages"""
    id: Optional[str]
    session_id: str
    role: str
    content: str
    timestamp: datetime
    
    @classmethod
    def create_user_message(cls, session_id: str, content: str) -> 'Message':
        return cls(id=str(uuid.uuid4()), session_id=session_id, role='user', content=content, timestamp=datetime.now())
    
    @classmethod
    def create_assistant_message(cls, session_id: str, content: str) -> 'Message':
        return cls(id=str(uuid.uuid4()), session_id=session_id, role='assistant', content=content, timestamp=datetime.now())
    
    def to_dict(self) -> dict:
        return {'id': self.id, 'session_id': self.session_id, 'role': self.role, 'content': self.content, 'timestamp': self.timestamp.isoformat()}

# ============================================================================
# API MODELS (Pydantic)
# ============================================================================

class StartSessionRequest(BaseModel):
    student_name: str = Field(..., description="The name or identifier of the student.")
    course_code: str = Field(..., description="The code for the course (e.g., 'CS101').")
    page_content: str = Field(..., description="The full text content from the current study page.")

class ChatSessionResponse(BaseModel):
    id: str
    student_name: str
    course_code: str
    session_name: str
    created_at: datetime
    context_updated_at: Optional[datetime] = Field(None, description="Timestamp of the last context update")

class MessageCreate(BaseModel):
    session_id: str = Field(..., description="Chat session ID")
    content: str = Field(..., description="Message content")

class MessageResponse(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    timestamp: datetime

class ChatResponse(BaseModel):
    user_message: MessageResponse
    assistant_message: MessageResponse
    session_id: str

class SessionListResponse(BaseModel):
    sessions: List[ChatSessionResponse]
    total: int

class MessageHistoryResponse(BaseModel):
    messages: List[MessageResponse]
    session_info: ChatSessionResponse
    total: int

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    services: Dict[str, str]
    version: str

# ============================================================================
# CHAT SERVICE (Business Logic)
# ============================================================================

class ChatService:
    def __init__(self):
        Config.validate()
        self.supabase: Client = create_client(Config.SUPABASE_URL, Config.SUPABASE_ANON_KEY)
        self.groq_client = Groq(api_key=Config.GROQ_API_KEY)
        logger.info("Chat service initialized successfully")

    async def start_or_reuse_session(self, student_name: str, course_code: str, page_content: str) -> ChatSessionResponse:
        try:
            result = self.supabase.table('chat_sessions').select('*').eq('student_name', student_name).eq('course_code', course_code).limit(1).execute()
            context_update_time = datetime.now()
            if result.data:
                session_data = result.data[0]
                session_id = session_data['id']
                logger.info(f"Reusing existing session {session_id} for {student_name} in {course_code}.")
                update_payload = {'context_content': page_content, 'context_updated_at': context_update_time.isoformat()}
                update_result = self.supabase.table('chat_sessions').update(update_payload).eq('id', session_id).execute()
                session_data = update_result.data[0]
            else:
                logger.info(f"Creating new session for {student_name} in {course_code}.")
                new_session = ChatSession.create_new(student_name, course_code, page_content)
                insert_result = self.supabase.table('chat_sessions').insert(new_session.to_dict()).execute()
                session_data = insert_result.data[0]
            return ChatSessionResponse(id=session_data['id'], student_name=session_data['student_name'], course_code=session_data['course_code'], session_name=session_data['session_name'], created_at=datetime.fromisoformat(session_data['created_at'].replace('Z', '+00:00')), context_updated_at=context_update_time)
        except Exception as e:
            logger.error(f"Error starting/reusing session: {e}", exc_info=True)
            raise

    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        try:
            result = self.supabase.table('chat_sessions').select('*').eq('id', session_id).execute()
            if not result.data: return None
            d = result.data[0]
            return ChatSession(id=d['id'], student_name=d['student_name'], course_code=d['course_code'], session_name=d['session_name'], created_at=datetime.fromisoformat(d['created_at'].replace('Z', '+00:00')), context_content=d.get('context_content'))
        except Exception as e:
            logger.error(f"Error getting session {session_id}: {e}", exc_info=True)
            raise

    async def save_message(self, message: Message) -> MessageResponse:
        try:
            self.supabase.table('messages').insert(message.to_dict()).execute()
            return MessageResponse(**message.to_dict())
        except Exception as e:
            logger.error(f"Error saving message: {e}", exc_info=True)
            raise

    async def get_session_messages(self, session_id: str, limit: int = 50) -> List[Message]:
        try:
            result = self.supabase.table('messages').select('*').eq('session_id', session_id).order('timestamp', desc=False).limit(limit).execute()
            messages = []
            for msg_data in result.data:
                msg_data['timestamp'] = datetime.fromisoformat(msg_data['timestamp'].replace('Z', '+00:00'))
                messages.append(Message(**msg_data))
            return messages
        except Exception as e:
            logger.error(f"Error getting messages for session {session_id}: {e}", exc_info=True)
            raise

    async def get_ai_response(self, user_message: str, session: ChatSession) -> str:
        try:
            conversation_history = await self.get_session_messages(session.id, limit=10)
            system_prompt_template = """You are a calm, patient, and encouraging AI Tutor. Your goal is to help students understand their study material clearly and without stress.

Your Personality & Style:
- **Patient and Supportive**: Always be encouraging. Use phrases like "That's a great question," "Let's break that down," or "You're on the right track."
- **Clear and Simple**: Explain complex topics in a simple, step-by-step manner. Use analogies if they help clarify a concept.
- **Guiding, Not Just Answering**: Help the student think for themselves. After explaining, ask follow-up questions like, "Does that make sense?" or "Can you try explaining that back in your own words?"

**Your Core Rules - VERY IMPORTANT:**
1.  **STRICTLY CONTEXT-BOUND**: You **MUST** base your answers **ONLY** on the "STUDY MATERIAL CONTEXT" provided below. Do not use any information from outside this text.
2.  **HANDLE MISSING INFORMATION**: If the answer is not in the context, you must politely state that the information isn't available in the provided material. For example: "Based on the text you've provided, that information isn't covered. Is there another section I can help you with?" Do not make up answers.
3.  **ENGLISH ONLY**: You must respond **only in English**.
4.  **USE FIRST NAME**: Address the student by their first name only. The student's full name is `{student_name}`. For example, if the name is 'Sara Johnson', you should simply call them 'Sara'.

---
STUDY MATERIAL CONTEXT FOR THIS QUESTION:
{page_content}
---

Now, calmly and patiently help `{student_name}` with their question.
"""
            student_first_name = session.student_name.split(' ')[0]
            system_prompt = system_prompt_template.format(student_name=student_first_name, page_content=session.context_content or "No context provided. Please ask the user to provide the study material from their page.")
            messages = [{"role": "system", "content": system_prompt}] + [{"role": msg.role, "content": msg.content} for msg in conversation_history] + [{"role": "user", "content": user_message}]
            completion = self.groq_client.chat.completions.create(model=Config.AI_MODEL, messages=messages, temperature=0.5, max_tokens=2048, top_p=1, stream=False)
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error getting AI response: {e}", exc_info=True)
            raise

    async def chat(self, session: ChatSession, user_message_content: str) -> Tuple[MessageResponse, MessageResponse]:
        try:
            user_message = Message.create_user_message(session.id, user_message_content)
            user_response = await self.save_message(user_message)
            ai_content = await self.get_ai_response(user_message_content, session)
            assistant_message = Message.create_assistant_message(session.id, ai_content)
            assistant_response = await self.save_message(assistant_message)
            return user_response, assistant_response
        except Exception as e:
            logger.error(f"Error in chat interaction: {e}", exc_info=True)
            raise

    async def health_check(self) -> dict:
        health = {"database": "unknown", "groq_api": "unknown"}
        try:
            self.supabase.table('chat_sessions').select('id').limit(1).execute()
            health["database"] = "healthy"
        except Exception: health["database"] = "unhealthy"
        try:
            self.groq_client.chat.completions.create(model=Config.AI_MODEL, messages=[{"role": "user", "content": "test"}], max_tokens=1)
            health["groq_api"] = "healthy"
        except Exception: health["groq_api"] = "unhealthy"
        return health

# ============================================================================
# FASTAPI APP SETUP
# ============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Global service instance for use in lifespan and endpoints
_chat_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    logger.info("Starting AI Study Buddy API...")
    global _chat_service
    _chat_service = ChatService()
    logger.info("Startup complete. Services are ready.")
    
    yield # The application runs while the 'yield' is active
    
    # Code to run on shutdown
    logger.info("Shutting down AI Study Buddy API...")

app = FastAPI(title="AI Study Buddy API", version="3.2.1", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_model=dict, tags=["General"])
async def root():
    return {"message": "AI Study Buddy API is running!", "version": "3.2.1"}

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check_endpoint():
    health_status = await _chat_service.health_check()
    overall_status = "healthy" if all("healthy" in s for s in health_status.values()) else "degraded"
    return HealthResponse(status=overall_status, timestamp=datetime.now(), services=health_status, version="3.2.1")

@app.post("/session/start", response_model=ChatSessionResponse, status_code=status.HTTP_200_OK, tags=["Session Management"])
async def start_session(request_data: StartSessionRequest):
    """Starts a new session or reuses an existing one for a student and course."""
    try:
        return await _chat_service.start_or_reuse_session(student_name=request_data.student_name, course_code=request_data.course_code, page_content=request_data.page_content)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not start or reuse the session.")

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(message_data: MessageCreate):
    """Send a message and get AI response within a session."""
    try:
        session = await _chat_service.get_session(message_data.session_id)
        if not session:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found. Please start a session first.")
        user_message, assistant_message = await _chat_service.chat(session=session, user_message_content=message_data.content)
        return ChatResponse(user_message=user_message, assistant_message=assistant_message, session_id=message_data.session_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to process chat message.")

@app.get("/sessions/{session_id}/messages", response_model=MessageHistoryResponse, tags=["Session Management"])
async def get_session_messages(session_id: str, limit: int = 50):
    """Get message history for a session."""
    try:
        session_obj = await _chat_service.get_session(session_id)
        if not session_obj:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
        messages = await _chat_service.get_session_messages(session_id, limit)
        message_responses = [MessageResponse(**msg.to_dict()) for msg in messages]
        session_info = ChatSessionResponse(id=session_obj.id, student_name=session_obj.student_name, course_code=session_obj.course_code, session_name=session_obj.session_name, created_at=session_obj.created_at)
        return MessageHistoryResponse(messages=message_responses, session_info=session_info, total=len(message_responses))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve message history.")

# ============================================================================
# MAIN ENTRY (For Local Development Only)
# ============================================================================

if __name__ == "__main__":
    # This block is for local development. 
    # Production platforms like Railway use a "Start Command" in their dashboard,
    # for example: uvicorn consolidated_server:app --host 0.0.0.0 --port $PORT
    
    # Assuming this file is named "consolidated_server.py"
    filename_without_extension = os.path.splitext(os.path.basename(__file__))[0]
    
    print(f"--- Starting in LOCAL DEVELOPMENT mode ---")
    print(f"--- To run in production, use a command like: uvicorn {filename_without_extension}:app ---")
    uvicorn.run(
        f"{filename_without_extension}:app",
        host="127.0.0.1", 
        port=8000, 
        reload=True
    )
