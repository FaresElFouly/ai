#!/usr/bin/env python3
"""
Consolidated AI Study Buddy API Server
A comprehensive FastAPI-based REST API that merges all server components
into a single, production-ready application serving multiple platforms.

Features:

Multi-user chat sessions with persistent storage

AI-powered responses using Groq API

Egyptian tutoring personality

Context-aware conversations

Session management and history

Health monitoring

Production optimizations

Multiple deployment options
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
from pydantic import BaseModel, Field as PydanticField
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
    AI_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # Groq model

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

@dataclass
class ChatSession:
    """Data model for chat sessions"""
    id: str
    user_id: str
    session_name: str
    created_at: datetime
    course_id: Optional[str] = None
    page_content: Optional[str] = None

    @classmethod
    def create_new(cls, user_id: str, session_name: str = None) -> 'ChatSession':
        """Create a new chat session with generated ID"""
        session_id = str(uuid.uuid4())
        if not session_name:
            session_name = f"Chat Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        return cls(
            id=session_id,
            user_id=user_id,
            session_name=session_name,
            created_at=datetime.now()
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for Supabase insertion"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'session_name': self.session_name,
            'created_at': self.created_at.isoformat(),
            'course_id': self.course_id,
            'page_content': self.page_content
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
        """Create a new user message"""
        return cls(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role='user',
            content=content,
            timestamp=datetime.now()
        )

    @classmethod
    def create_assistant_message(cls, session_id: str, content: str) -> 'Message':
        """Create a new assistant message"""
        return cls(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role='assistant',
            content=content,
            timestamp=datetime.now()
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for Supabase insertion"""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat()
        }

# ============================================================================
# API MODELS (Pydantic)
# ============================================================================

class SessionStartOrGet(BaseModel):
    user_id: str
    course_code: str
    page_content: str

class ChatSessionCreate(BaseModel):
    """Request model for creating a new chat session"""
    user_id: str = PydanticField(..., description="User identifier")
    session_name: Optional[str] = PydanticField(None, description="Optional session name")

class ChatSessionResponse(BaseModel):
    """Response model for chat session"""
    id: str
    user_id: str
    session_name: str
    created_at: datetime
    course_id: Optional[str] = None
    page_content: Optional[str] = None

class MessageCreate(BaseModel):
    """Request model for sending a message"""
    session_id: str = PydanticField(..., description="Chat session ID")
    content: str = PydanticField(..., description="Message content")

class MessageResponse(BaseModel):
    """Response model for a message"""
    id: str
    session_id: str
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime

class ChatResponse(BaseModel):
    """Response model for chat interaction"""
    user_message: MessageResponse
    assistant_message: MessageResponse
    session_id: str

class SessionListResponse(BaseModel):
    """Response model for listing user sessions"""
    sessions: List[ChatSessionResponse]
    total: int

class MessageHistoryResponse(BaseModel):
    """Response model for message history"""
    messages: List[MessageResponse]
    session_info: ChatSessionResponse
    total: int

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: datetime
    services: Dict[str, str]
    version: str

# ============================================================================
# CHAT SERVICE (Business Logic)
# ============================================================================

class ChatService:
    """Service class for handling chat operations"""

    def __init__(self):
        """Initialize the chat service"""
        Config.validate()
        self.supabase: Client = create_client(Config.SUPABASE_URL, Config.SUPABASE_ANON_KEY)
        self.groq_client = Groq(api_key=Config.GROQ_API_KEY)
        logger.info("Chat service initialized successfully")

    async def start_or_get_session(self, user_id: str, course_code: str, page_content: str) -> ChatSessionResponse:
        """Starts a new session or retrieves an existing one for a user and course."""
        # Check if a course with the given code exists
        course_result = self.supabase.table('courses').select('id').eq('course_code', course_code).execute()
        if not course_result.data:
            # Create a new course if it doesn't exist
            new_course_result = self.supabase.table('courses').insert({'course_code': course_code}).execute()
            course_id = new_course_result.data[0]['id']
        else:
            course_id = course_result.data[0]['id']

        # Check if a session already exists for this user and course
        session_result = self.supabase.table('chat_sessions').select('*').eq('user_id', user_id).eq('course_id', course_id).execute()

        if session_result.data:
            # If a session exists, update the page_content and return the session
            session_id = session_result.data[0]['id']
            self.supabase.table('chat_sessions').update({'page_content': page_content}).eq('id', session_id).execute()
            return await self.get_session(session_id)
        else:
            # If no session exists, create a new one
            session_name = f"{course_code} Session"
            session = ChatSession.create_new(user_id=user_id, session_name=session_name)
            
            session_dict = session.to_dict()
            session_dict['course_id'] = course_id
            session_dict['page_content'] = page_content

            self.supabase.table('chat_sessions').insert(session_dict).execute()

            return ChatSessionResponse(
                id=session.id,
                user_id=session.user_id,
                session_name=session.session_name,
                created_at=session.created_at,
                course_id=course_id,
                page_content=page_content
            )

    async def create_session(self, user_id: str, session_name: str = None) -> ChatSessionResponse:
        """Create a new chat session"""
        try:
            session = ChatSession.create_new(user_id, session_name)

            # Insert session into Supabase
            result = self.supabase.table('chat_sessions').insert(session.to_dict()).execute()

            if not result.data:
                raise Exception("Failed to create session in database")

            logger.info(f"Created new session: {session.id} for user: {user_id}")

            return ChatSessionResponse(
                id=session.id,
                user_id=session.user_id,
                session_name=session.session_name,
                created_at=session.created_at
            )
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            raise

    async def get_session(self, session_id: str) -> Optional[ChatSessionResponse]:
        """Get a chat session by ID"""
        try:
            result = self.supabase.table('chat_sessions').select('*').eq('id', session_id).execute()

            if not result.data:
                return None

            session_data = result.data[0]
            return ChatSessionResponse(
                id=session_data['id'],
                user_id=session_data['user_id'],
                session_name=session_data['session_name'],
                created_at=datetime.fromisoformat(session_data['created_at'].replace('Z', '+00:00')),
                course_id=session_data.get('course_id'),
                page_content=session_data.get('page_content')
            )
        except Exception as e:
            logger.error(f"Error getting session {session_id}: {e}")
            raise

    async def get_user_sessions(self, user_id: str) -> List[ChatSessionResponse]:
        """Get all sessions for a user"""
        try:
            result = self.supabase.table('chat_sessions').select('*').eq('user_id', user_id).order('created_at', desc=True).execute()

            sessions = []
            for session_data in result.data:
                sessions.append(ChatSessionResponse(
                    id=session_data['id'],
                    user_id=session_data['user_id'],
                    session_name=session_data['session_name'],
                    created_at=datetime.fromisoformat(session_data['created_at'].replace('Z', '+00:00')),
                    course_id=session_data.get('course_id'),
                    page_content=session_data.get('page_content')
                ))

            return sessions
        except Exception as e:
            logger.error(f"Error getting sessions for user {user_id}: {e}")
            raise

    async def delete_session(self, session_id: str) -> bool:
        """Delete a chat session and all its messages"""
        try:
            # Delete messages first
            self.supabase.table('messages').delete().eq('session_id', session_id).execute()

            # Delete session
            result = self.supabase.table('chat_sessions').delete().eq('id', session_id).execute()

            logger.info(f"Deleted session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            raise

    async def save_message(self, message: Message) -> MessageResponse:
        """Save a message to the database"""
        try:
            result = self.supabase.table('messages').insert(message.to_dict()).execute()

            if not result.data:
                raise Exception("Failed to save message to database")

            return MessageResponse(
                id=message.id,
                session_id=message.session_id,
                role=message.role,
                content=message.content,
                timestamp=message.timestamp
            )
        except Exception as e:
            logger.error(f"Error saving message: {e}")
            raise

    async def get_session_messages(self, session_id: str, limit: int = 50) -> List[Message]:
        """Get messages for a session"""
        try:
            result = self.supabase.table('messages').select('*').eq('session_id', session_id).order('timestamp', desc=False).limit(limit).execute()

            messages = []
            for msg_data in result.data:
                messages.append(Message(
                    id=msg_data['id'],
                    session_id=msg_data['session_id'],
                    role=msg_data['role'],
                    content=msg_data['content'],
                    timestamp=datetime.fromisoformat(msg_data['timestamp'].replace('Z', '+00:00'))
                ))

            return messages
        except Exception as e:
            logger.error(f"Error getting messages for session {session_id}: {e}")
            raise

    async def get_ai_response(self, user_message: str, session_id: str) -> str:
        """Get AI response from Groq"""
        try:
            # Get conversation history and session info
            session = await self.get_session(session_id)
            conversation_history = await self.get_session_messages(session_id, limit=20)

            # Prepare conversation context with professional Egyptian study buddy personality
            system_prompt = f"""You are a professional Egyptian study buddy AI assistant. Your personality traits:

    🎓 Professional Academic Companion: You're here to help students learn and understand their study materials
    📚 Context-Only Responses: You ONLY answer questions based on the text/context provided by the user
    🇪🇬 Egyptian Tutoring Style: Friendly, encouraging, and supportive like a helpful Egyptian tutor
    🔍 Analytical Approach: Break down complex topics, explain step-by-step, provide examples
    💡 Mixed Language: Use Arabic for explanations but keep technical terms/equations in English
    👨‍🎓 Student-Focused: Address the student by name when provided, adapt to their learning level

    IMPORTANT RULES:

    Context Dependency: You can ONLY answer questions about the text/context the user provides. The context is: {session.page_content}

    No External Knowledge: Don't use information outside the provided context

    Language Adaptation: Match the language of the provided context (Arabic/English/Mixed)

    Professional Tone: Be nice, serious, and helpful - avoid overly casual terms

    Educational Focus: Always aim to help the student understand, not just provide answers

    Response Style:

    Start with a brief acknowledgment

    Provide clear, structured explanations

    Use examples from the provided context

    End with encouragement or follow-up questions

    Keep technical terms in English, explanations in Arabic when appropriate

    If no context is provided: Politely ask the student to share the text or material they want help with.

    Remember: You're a professional academic companion who ONLY works with the provided context! 📚"""

            messages = [
                {"role": "system", "content": system_prompt}
            ]

            # Add conversation history (last 10 messages for context)
            for msg in conversation_history[-10:]:
                messages.append({"role": msg.role, "content": msg.content})

            # Add current user message
            messages.append({"role": "user", "content": user_message})

            # Get response from Groq using updated API
            completion = self.groq_client.chat.completions.create(
                model=Config.AI_MODEL,
                messages=messages,
                temperature=1,
                max_completion_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )

            return completion.choices[0].message.content

        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            raise Exception(f"Failed to get AI response: {str(e)}")

    async def chat(self, session_id: str, user_message_content: str) -> Tuple[MessageResponse, MessageResponse]:
        """Process a chat interaction"""
        try:
            # Create and save user message
            user_message = Message.create_user_message(session_id, user_message_content)
            user_response = await self.save_message(user_message)

            # Get AI response
            ai_content = await self.get_ai_response(user_message_content, session_id)

            # Create and save assistant message
            assistant_message = Message.create_assistant_message(session_id, ai_content)
            assistant_response = await self.save_message(assistant_message)

            return user_response, assistant_response

        except Exception as e:
            logger.error(f"Error in chat interaction: {e}")
            raise

    async def health_check(self) -> dict:
        """Check the health of all services"""
        health = {
            "database": "unknown",
            "groq_api": "unknown"
        }

        # Check database connection
        try:
            result = self.supabase.table('chat_sessions').select('id').limit(1).execute()
            health["database"] = "healthy"
        except Exception as e:
            health["database"] = f"error: {str(e)}"

        # Check Groq API
        try:
            completion = self.groq_client.chat.completions.create(
                model=Config.AI_MODEL,
                messages=[{"role": "user", "content": "test"}],
                max_completion_tokens=1,
                temperature=0.1
            )
            health["groq_api"] = "healthy"
        except Exception as e:
            health["groq_api"] = f"error: {str(e)}"

        return health

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/aiasis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# FASTAPI APPLICATION SETUP
# ============================================================================
# Initialize FastAPI app with production settings
app = FastAPI(
    title="AI Study Buddy - Consolidated Global API",
    description="Professional AI-powered study companion with Egyptian tutoring personality - Consolidated Server",
    version="3.0.0",
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") != "production" else None
)

# Add production middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )

# Serve static files (if directory exists)
try:
    if os.path.exists("static"):
        app.mount("/static", StaticFiles(directory="static"), name="static")
        logger.info("Static files mounted at /static")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")

# ============================================================================
# SERVICE INITIALIZATION
# ============================================================================
# Global chat service instance
_chat_service = None

def get_chat_service() -> ChatService:
    """Get or create chat service instance"""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Study Buddy - Consolidated Global API",
        "version": "3.0.0",
        "description": "Professional AI-powered study companion with Egyptian tutoring personality",
        "endpoints": {
            "health": "/health",
            "docs": "/docs" if os.getenv("ENVIRONMENT") != "production" else "disabled",
            "web_interface": "/static/index.html" if os.path.exists("static") else "not_available"
        },
        "features": [
            "Context-only responses",
            "Egyptian tutoring personality",
            "Session management",
            "Multilingual support",
            "Production optimizations",
            "Multi-platform support"
        ]
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        chat_service = get_chat_service()
        services_health = await chat_service.health_check()

        overall_status = "healthy" if all(
            "healthy" in status for status in services_health.values()
        ) else "degraded"

        return HealthResponse(
            status=overall_status,
            timestamp=datetime.now(),
            services=services_health,
            version="3.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(),
            services={"error": str(e)},
            version="3.0.0"
        )

@app.post("/sessions/start_or_get", response_model=ChatSessionResponse)
async def start_or_get_session(session_data: SessionStartOrGet):
    """
    Starts a new session or retrieves an existing one for a user and course.
    """
    try:
        chat_service = get_chat_service()
        session = await chat_service.start_or_get_session(
            user_id=session_data.user_id,
            course_code=session_data.course_code,
            page_content=session_data.page_content
        )
        return session
    except Exception as e:
        logger.error(f"Failed to start or get session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start or get session"
        )

@app.post("/sessions", response_model=ChatSessionResponse)
async def create_session(session_data: ChatSessionCreate):
    """Create a new chat session"""
    try:
        chat_service = get_chat_service()
        session = await chat_service.create_session(
            user_id=session_data.user_id,
            session_name=session_data.session_name
        )
        return session
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create session"
        )

@app.get("/sessions/{session_id}", response_model=ChatSessionResponse)
async def get_session(session_id: str):
    """Get a specific session"""
    try:
        chat_service = get_chat_service()
        session = await chat_service.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        return session
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve session"
        )

@app.get("/users/{user_id}/sessions", response_model=SessionListResponse)
async def get_user_sessions(user_id: str):
    """Get all sessions for a user"""
    try:
        chat_service = get_chat_service()
        sessions = await chat_service.get_user_sessions(user_id)
        return SessionListResponse(sessions=sessions, total=len(sessions))
    except Exception as e:
        logger.error(f"Failed to get sessions for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user sessions"
        )

@app.post("/chat", response_model=ChatResponse)
async def chat(message_data: MessageCreate):
    """Send a message and get AI response"""
    try:
        chat_service = get_chat_service()
        # Verify session exists
        session = await chat_service.get_session(message_data.session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )

        # Process chat interaction
        user_message, assistant_message = await chat_service.chat(
            session_id=message_data.session_id,
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
        logger.error(f"Chat interaction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat message"
        )

@app.get("/sessions/{session_id}/messages", response_model=MessageHistoryResponse)
async def get_session_messages(session_id: str, limit: int = 50):
    """Get message history for a session"""
    try:
        chat_service = get_chat_service()

        # Verify session exists
        session = await chat_service.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )

        # Get messages
        messages = await chat_service.get_session_messages(session_id, limit)

        # Convert to response format
        message_responses = [
            MessageResponse(
                id=msg.id,
                session_id=msg.session_id,
                role=msg.role,
                content=msg.content,
                timestamp=msg.timestamp
            ) for msg in messages
        ]

        return MessageHistoryResponse(
            messages=message_responses,
            session_info=session,
            total=len(message_responses)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get messages for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve message history"
        )

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    try:
        chat_service = get_chat_service()

        # Verify session exists
        session = await chat_service.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )

        # Delete session
        await chat_service.delete_session(session_id)

        return {"message": "Session deleted successfully", "session_id": session_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete session"
        )

# ============================================================================
# STARTUP AND SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting AI Study Buddy Consolidated API...")
    try:
        # Initialize chat service
        chat_service = get_chat_service()
        logger.info("Chat service initialized successfully")

        # Test database connection
        health = await chat_service.health_check()
        logger.info(f"Service health check: {health}")

        logger.info("AI Study Buddy Consolidated API started successfully")
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down AI Study Buddy Consolidated API...")
    # Add any cleanup code here
    logger.info("AI Study Buddy Consolidated API shut down complete")

# ============================================================================
# DEPLOYMENT OPTIONS
# ============================================================================

def run_development():
    """Run in development mode with auto-reload"""
    logger.info("🚀 Starting AI Study Buddy Consolidated API - Development Mode")
    logger.info("=" * 60)

    try:
        # Validate configuration
        Config.validate()
        logger.info("✅ Configuration validated")

        # Server configuration
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8000"))

        logger.info(f"🌐 Server will start on: http://{host}:{port}")
        logger.info(f"📚 API Documentation: http://{host}:{port}/docs")
        logger.info(f"🔄 Auto-reload: enabled")
        logger.info("=" * 60)

        # Start the server
        uvicorn.run(
            "__main__:app",
            host=host,
            port=port,
            reload=True,
            log_level="info",
            access_log=True
        )

    except Exception as e:
        logger.error(f"❌ Failed to start development server: {e}")
        sys.exit(1)

def run_production():
    """Run in production mode"""
    logger.info("🚀 Starting AI Study Buddy Consolidated API - Production Mode")
    logger.info("=" * 60)

    try:
        # Validate configuration
        Config.validate()
        logger.info("✅ Configuration validated")

        # Server configuration
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8000"))

        logger.info(f"🌐 Server starting on: http://{host}:{port}")
        logger.info(f"🔒 Production mode: docs disabled")
        logger.info("=" * 60)

        # Start the server
        uvicorn.run(
            "__main__:app",
            host=host,
            port=port,
            reload=False,
            log_level="info",
            access_log=True
        )

    except Exception as e:
        logger.error(f"❌ Failed to start production server: {e}")
        sys.exit(1)

# WSGI application for deployment platforms like PythonAnywhere
application = app

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Determine run mode from environment or command line
    mode = os.getenv("RUN_MODE", "development").lower()

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

    if mode == "production":
        run_production()
    else:
        run_development()
