#!/usr/bin/env python3
"""
Consolidated AI Study Buddy API Server
A comprehensive FastAPI-based REST API that merges all server components
into a single, production-ready application serving multiple platforms.

Features:
- Multi-user chat sessions with persistent storage
- AI-powered responses using Groq API
- Egyptian tutoring personality
- Context-aware conversations with injected page content
- Session management (start/reuse) and history
- Health monitoring
- Production optimizations
- Multiple deployment options
"""

import os
import sys
import logging
import time
import uuid
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field

# Third-party imports
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from supabase import create_client, Client
from groq import Groq
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration class for the AI Chat Assistant"""
    
    # Supabase Configuration
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_ANON_KEY = os.getenv('SUPABASE_ANON_KEY')
    
    # Groq API Configuration
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    
    # Default AI Model
    AI_MODEL = "llama3-8b-8192" # A reliable and fast model from Groq
    
    @classmethod
    def validate(cls):
        """Validate that all required environment variables are set"""
        required_vars = [
            ('SUPABASE_URL', cls.SUPABASE_URL),
            ('SUPABASE_ANON_KEY', cls.SUPABASE_ANON_KEY),
            ('GROQ_API_KEY', cls.GROQ_API_KEY)
        ]
        
        missing_vars = []
        for var_name, var_value in required_vars:
            if not var_value:
                missing_vars.append(var_name)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        return True

# ============================================================================
# DATA MODELS
# ============================================================================

### EDITED ### - Updated dataclass to match new database schema
@dataclass
class ChatSession:
    """Data model for chat sessions"""
    id: str
    student_name: str
    course_code: str
    session_name: str
    created_at: datetime
    context_content: Optional[str] = None
    
    @classmethod
    def create_new(cls, student_name: str, course_code: str, context_content: str, session_name: Optional[str] = None) -> 'ChatSession':
        """Create a new chat session with generated ID"""
        session_id = str(uuid.uuid4())
        if not session_name:
            session_name = f"Session for {student_name} - {course_code}"
        
        return cls(
            id=session_id,
            student_name=student_name,
            course_code=course_code,
            session_name=session_name,
            created_at=datetime.now(),
            context_content=context_content
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for Supabase insertion"""
        return {
            'id': self.id,
            'student_name': self.student_name,
            'course_code': self.course_code,
            'session_name': self.session_name,
            'created_at': self.created_at.isoformat(),
            'context_content': self.context_content
        }

@dataclass
class Message:
    """Data model for chat messages"""
    id: Optional[str]
    session_id: str
    role: str  # 'user' or 'assistant'
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

### EDITED ### - New request model for starting/reusing a session
class StartSessionRequest(BaseModel):
    """Request model for starting or reusing a session."""
    student_name: str = Field(..., description="The name or identifier of the student.")
    course_code: str = Field(..., description="The code for the course (e.g., 'CS101').")
    page_content: str = Field(..., description="The full text content from the current study page.")

### EDITED ### - Updated response model to match new schema
class ChatSessionResponse(BaseModel):
    """Response model for chat session"""
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

    ### EDITED ### - New method to handle starting or reusing a session
    async def start_or_reuse_session(self, student_name: str, course_code: str, page_content: str) -> ChatSessionResponse:
        """Finds a session by student and course, or creates a new one. Updates the context."""
        try:
            # Check for an existing session
            result = self.supabase.table('chat_sessions').select('*').eq('student_name', student_name).eq('course_code', course_code).limit(1).execute()
            
            context_update_time = datetime.now()

            if result.data:
                # Session exists, reuse it and update the context
                existing_session = result.data[0]
                session_id = existing_session['id']
                logger.info(f"Reusing existing session {session_id} for {student_name} in {course_code}.")
                
                update_result = self.supabase.table('chat_sessions').update({
                    'context_content': page_content,
                    'context_updated_at': context_update_time.isoformat() # Add this column to Supabase if you want to track updates
                }).eq('id', session_id).execute()
                
                if not update_result.data:
                    raise Exception("Failed to update session context.")
                
                session_data = update_result.data[0]
            else:
                # Session does not exist, create a new one
                logger.info(f"Creating new session for {student_name} in {course_code}.")
                new_session = ChatSession.create_new(student_name, course_code, page_content)
                insert_result = self.supabase.table('chat_sessions').insert(new_session.to_dict()).execute()

                if not insert_result.data:
                    raise Exception("Failed to create new session in database.")
                
                session_data = insert_result.data[0]

            return ChatSessionResponse(
                id=session_data['id'],
                student_name=session_data['student_name'],
                course_code=session_data['course_code'],
                session_name=session_data['session_name'],
                created_at=datetime.fromisoformat(session_data['created_at'].replace('Z', '+00:00')),
                context_updated_at=context_update_time
            )

        except Exception as e:
            logger.error(f"Error starting/reusing session: {e}")
            raise

    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a full chat session object by ID, including context."""
        try:
            result = self.supabase.table('chat_sessions').select('*').eq('id', session_id).execute()
            if not result.data:
                return None
            d = result.data[0]
            return ChatSession(
                id=d['id'],
                student_name=d['student_name'],
                course_code=d['course_code'],
                session_name=d['session_name'],
                created_at=datetime.fromisoformat(d['created_at'].replace('Z', '+00:00')),
                context_content=d.get('context_content')
            )
        except Exception as e:
            logger.error(f"Error getting session {session_id}: {e}")
            raise

    ### EDITED ### - Updated to get sessions by student_name instead of user_id
    async def get_student_sessions(self, student_name: str) -> List[ChatSessionResponse]:
        """Get all sessions for a student."""
        try:
            result = self.supabase.table('chat_sessions').select('*').eq('student_name', student_name).order('created_at', desc=True).execute()
            sessions = []
            for d in result.data:
                sessions.append(ChatSessionResponse(
                    id=d['id'],
                    student_name=d['student_name'],
                    course_code=d['course_code'],
                    session_name=d['session_name'],
                    created_at=datetime.fromisoformat(d['created_at'].replace('Z', '+00:00'))
                ))
            return sessions
        except Exception as e:
            logger.error(f"Error getting sessions for student {student_name}: {e}")
            raise

    async def save_message(self, message: Message) -> MessageResponse:
        """Save a message to the database"""
        try:
            result = self.supabase.table('messages').insert(message.to_dict()).execute()
            if not result.data:
                raise Exception("Failed to save message to database")
            return MessageResponse(**message.to_dict())
        except Exception as e:
            logger.error(f"Error saving message: {e}")
            raise

    async def get_session_messages(self, session_id: str, limit: int = 50) -> List[Message]:
        """Get messages for a session"""
        try:
            result = self.supabase.table('messages').select('*').eq('session_id', session_id).order('timestamp', desc=False).limit(limit).execute()
            return [Message(**msg) for msg in result.data]
        except Exception as e:
            logger.error(f"Error getting messages for session {session_id}: {e}")
            raise

    ### EDITED ### - Major update to inject page content into the AI's system prompt
    async def get_ai_response(self, user_message: str, session: ChatSession) -> str:
        """Get AI response from Groq, using the session's context."""
        try:
            # Get conversation history
            conversation_history = await self.get_session_messages(session.id, limit=10)

            # Prepare the dynamic system prompt with injected knowledge
            system_prompt_template = """You are a professional Egyptian study buddy AI assistant named 'AI-Ostaz' (الأستاذ الذكي).

Your Personality:
- 🎓 **Professional & Academic**: You help students understand their materials.
- 🇪🇬 **Egyptian Tutor Style**: Friendly, encouraging ('ya basha', ' بالتوفيق'), and supportive. You can mix Arabic and English naturally.
- 💡 **Clarity is Key**: Break down complex topics simply. Use analogies if helpful.

**Your Core Rules - VERY IMPORTANT:**
1.  **STRICTLY CONTEXT-BOUND**: You **MUST** base your answers **ONLY** on the "STUDY MATERIAL CONTEXT" provided below. Do not use any external knowledge.
2.  **If the answer is not in the context**: Politely state that the information is not in the provided material. For example: "حسب المادة اللي قدامي، المعلومة دي مش موجودة. ممكن تتأكد من الصفحة أو تديني جزء تاني أساعدك فيه؟" (According to the material I have, this information isn't present. Could you check the page or give me another part to help with?).
3.  **Acknowledge the Student**: Address the student by their name, `{student_name}`.

---
STUDY MATERIAL CONTEXT FOR THIS QUESTION:
{page_content}
---

Now, help `{student_name}` with their question.
"""
            
            # Format the prompt with the specific session data
            system_prompt = system_prompt_template.format(
                student_name=session.student_name,
                page_content=session.context_content or "No context provided. Please ask the user to provide the study material from their page."
            )

            messages = [{"role": "system", "content": system_prompt}]
            for msg in conversation_history:
                messages.append({"role": msg.role, "content": msg.content})
            messages.append({"role": "user", "content": user_message})

            # Get response from Groq
            completion = self.groq_client.chat.completions.create(
                model=Config.AI_MODEL,
                messages=messages,
                temperature=0.7, # Slightly lower for more factual responses based on context
                max_tokens=2048,
                top_p=1,
                stream=False
            )

            return completion.choices[0].message.content

        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            raise Exception(f"Failed to get AI response: {str(e)}")

    ### EDITED ### - Updated to pass the full session object to get_ai_response
    async def chat(self, session: ChatSession, user_message_content: str) -> Tuple[MessageResponse, MessageResponse]:
        """Process a chat interaction"""
        try:
            # Create and save user message
            user_message = Message.create_user_message(session.id, user_message_content)
            user_response = await self.save_message(user_message)

            # Get AI response using the full session object (which includes context)
            ai_content = await self.get_ai_response(user_message_content, session)

            # Create and save assistant message
            assistant_message = Message.create_assistant_message(session.id, ai_content)
            assistant_response = await self.save_message(assistant_message)

            return user_response, assistant_response

        except Exception as e:
            logger.error(f"Error in chat interaction: {e}")
            raise

    # (The health_check, delete_session methods remain the same)
    async def delete_session(self, session_id: str) -> bool:
        """Delete a chat session and all its messages"""
        try:
            self.supabase.table('messages').delete().eq('session_id', session_id).execute()
            self.supabase.table('chat_sessions').delete().eq('id', session_id).execute()
            logger.info(f"Deleted session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            raise

    async def health_check(self) -> dict:
        health = {"database": "unknown", "groq_api": "unknown"}
        try:
            self.supabase.table('chat_sessions').select('id').limit(1).execute()
            health["database"] = "healthy"
        except Exception as e:
            health["database"] = f"error: {str(e)}"
        try:
            self.groq_client.chat.completions.create(model=Config.AI_MODEL, messages=[{"role": "user", "content": "test"}], max_tokens=1)
            health["groq_api"] = "healthy"
        except Exception as e:
            health["groq_api"] = f"error: {str(e)}"
        return health

# ============================================================================
# LOGGING, FASTAPI SETUP, MIDDLEWARE, ETC. (No changes needed here)
# ============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Study Buddy API", version="3.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Global service instance
_chat_service = None
def get_chat_service() -> ChatService:
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_model=dict)
async def root():
    return {"message": "AI Study Buddy API is running!"}

@app.get("/health", response_model=HealthResponse)
async def health_check_endpoint():
    chat_service = get_chat_service()
    services_health = await chat_service.health_check()
    overall_status = "healthy" if all("healthy" in s for s in services_health.values()) else "degraded"
    return HealthResponse(status=overall_status, timestamp=datetime.now(), services=services_health, version="3.1.0")

### EDITED ### - This is the NEW endpoint for starting/reusing a session.
@app.post("/session/start", response_model=ChatSessionResponse, status_code=status.HTTP_200_OK)
async def start_session(request_data: StartSessionRequest):
    """
    Starts a new session or reuses an existing one for a student and course.
    This endpoint is the primary entry point for a student. It sets the knowledge context
    for the session based on the current page content.
    """
    try:
        chat_service = get_chat_service()
        session = await chat_service.start_or_reuse_session(
            student_name=request_data.student_name,
            course_code=request_data.course_code,
            page_content=request_data.page_content
        )
        return session
    except Exception as e:
        logger.error(f"Failed to start/reuse session: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not start or reuse the session.")

### EDITED ### - The old /sessions endpoint is now deprecated. I've commented it out.
# @app.post("/sessions", response_model=ChatSessionResponse)
# async def create_session(session_data: ChatSessionCreate):
#    ...

### EDITED ### - This endpoint now gets sessions by student name.
@app.get("/sessions/student/{student_name}", response_model=SessionListResponse)
async def get_student_sessions_endpoint(student_name: str):
    """Get all sessions for a specific student."""
    try {
        chat_service = get_chat_service()
        sessions = await chat_service.get_student_sessions(student_name)
        return SessionListResponse(sessions=sessions, total=len(sessions))
    } except Exception as e:
        logger.error(f"Failed to get sessions for student {student_name}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve student sessions")


@app.post("/chat", response_model=ChatResponse)
async def chat(message_data: MessageCreate):
    """Send a message and get AI response within a session."""
    try:
        chat_service = get_chat_service()
        # Verify session exists and get its full details, including context
        session = await chat_service.get_session(message_data.session_id)
        if not session:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found. Please start a session first.")

        user_message, assistant_message = await chat_service.chat(
            session=session,
            user_message_content=message_data.content
        )

        return ChatResponse(
            user_message=user_message,
            assistant_message=assistant_message,
            session_id=message_data.session_id
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat interaction failed: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to process chat message.")


# (The /sessions/{session_id}/messages and /sessions/{session_id} endpoints can remain as they are, but let's ensure they return the correct response model)

@app.get("/sessions/{session_id}/messages", response_model=MessageHistoryResponse)
async def get_session_messages(session_id: str, limit: int = 50):
    try:
        chat_service = get_chat_service()
        session_obj = await chat_service.get_session(session_id)
        if not session_obj:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
        
        messages = await chat_service.get_session_messages(session_id, limit)
        message_responses = [MessageResponse(**msg.to_dict()) for msg in messages]

        # Create the session info response
        session_info = ChatSessionResponse(
            id=session_obj.id,
            student_name=session_obj.student_name,
            course_code=session_obj.course_code,
            session_name=session_obj.session_name,
            created_at=session_obj.created_at
        )

        return MessageHistoryResponse(messages=message_responses, session_info=session_info, total=len(message_responses))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get messages for session {session_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve message history")


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    try:
        chat_service = get_chat_service()
        if not await chat_service.get_session(session_id):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
        
        await chat_service.delete_session(session_id)
        return {"message": "Session deleted successfully", "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete session")


# ============================================================================
# STARTUP/SHUTDOWN & MAIN ENTRY
# (No changes needed here)
# ============================================================================

@app.on_event("startup")
async def startup_event():
    logger.info("Starting AI Study Buddy API...")
    get_chat_service()

if __name__ == "__main__":
    # Assumes development mode for direct execution
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
