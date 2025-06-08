# AI Study Buddy - Consolidated API Server

A comprehensive, production-ready FastAPI server that consolidates all AI Study Buddy components into a single, powerful API that can serve multiple platforms and applications.

## 🚀 Features

- **🔧 All-in-One**: Consolidated server combining all previous components
- **🌍 Multi-Platform**: Serves web, mobile, desktop, and any HTTP client
- **🤖 AI-Powered**: Egyptian tutoring personality with Groq API integration
- **💾 Persistent Storage**: Supabase database integration
- **🔐 Session Management**: Multi-user support with session isolation
- **📚 Context-Aware**: AI responses based only on provided context
- **🏭 Production-Ready**: Optimized for deployment with proper logging
- **📊 Health Monitoring**: Built-in health checks and monitoring
- **🔄 Multiple Deployment Options**: Development, production, and WSGI modes

## 📁 What's Consolidated

This single file (`consolidated_server.py`) includes:

- **Configuration Management** (from `config.py`)
- **Data Models** (from `models.py` and `api_models.py`)
- **Business Logic** (from `chat_service.py`)
- **API Endpoints** (from `api_server.py` and `api_server_production.py`)
- **Middleware & Error Handling**
- **Startup/Shutdown Events**
- **Multiple Deployment Options**
- **WSGI Support** (from `wsgi.py`)
- **Server Startup Logic** (from `start_server.py`)

## 🛠️ Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file with:

```env
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key
GROQ_API_KEY=your_groq_api_key
```

### 3. Database Setup

Make sure your Supabase database has the required tables:

```sql
-- Chat sessions table
CREATE TABLE chat_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_name TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Messages table
CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_chat_sessions_user_id ON chat_sessions(user_id);
CREATE INDEX idx_messages_session_id ON messages(session_id);
CREATE INDEX idx_messages_timestamp ON messages(timestamp);
```

## 🚀 Running the Server

### Development Mode (with auto-reload)

```bash
python consolidated_server.py
# or
python consolidated_server.py development
```

### Production Mode

```bash
python consolidated_server.py production
# or
RUN_MODE=production python consolidated_server.py
```

### Using uvicorn directly

```bash
# Development
uvicorn consolidated_server:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn consolidated_server:app --host 0.0.0.0 --port 8000
```

### WSGI Deployment (PythonAnywhere, etc.)

The file includes a WSGI application object:

```python
# In your WSGI configuration
from consolidated_server import application
```

## 📡 API Endpoints

### Core Endpoints

- `GET /` - API information and status
- `GET /health` - Health check with service status
- `GET /docs` - Interactive API documentation (dev mode only)

### Session Management

- `POST /sessions` - Create new chat session
- `GET /sessions/{session_id}` - Get session details
- `GET /users/{user_id}/sessions` - List user sessions
- `DELETE /sessions/{session_id}` - Delete session

### Chat Operations

- `POST /chat` - Send message and get AI response
- `GET /sessions/{session_id}/messages` - Get message history

## 🌐 Multi-Platform Usage

### Web Applications

```javascript
// Create session
const session = await fetch('/sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: 'user123', session_name: 'Math Study' })
});

// Send message
const response = await fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
        session_id: sessionId, 
        content: 'Explain this equation: E=mc²' 
    })
});
```

### Mobile Applications

```python
import requests

# Create session
session_response = requests.post('http://your-server.com/sessions', json={
    'user_id': 'mobile_user_123',
    'session_name': 'Physics Study'
})

# Chat
chat_response = requests.post('http://your-server.com/chat', json={
    'session_id': session_id,
    'content': 'What is quantum mechanics?'
})
```

### Desktop Applications

```python
import httpx

async with httpx.AsyncClient() as client:
    # Create session
    session = await client.post('/sessions', json={
        'user_id': 'desktop_user',
        'session_name': 'Chemistry Study'
    })
    
    # Chat
    response = await client.post('/chat', json={
        'session_id': session_id,
        'content': 'Explain chemical bonding'
    })
```

## 🔧 Configuration Options

### Environment Variables

- `HOST` - Server host (default: 0.0.0.0)
- `PORT` - Server port (default: 8000)
- `ENVIRONMENT` - Environment mode (development/production)
- `RUN_MODE` - Run mode (development/production)

### AI Model Configuration

The server uses `meta-llama/llama-4-scout-17b-16e-instruct` by default. You can modify this in the `Config` class.

## 📊 Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

Returns service status for database and AI API connections.

### Logs

Logs are written to both console and `/tmp/aiasis.log` in production mode.

## 🚀 Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY consolidated_server.py .
COPY .env .

EXPOSE 8000
CMD ["python", "consolidated_server.py", "production"]
```

### PythonAnywhere

1. Upload `consolidated_server.py` and `requirements.txt`
2. Install dependencies: `pip install -r requirements.txt`
3. Configure WSGI file to import `application` from `consolidated_server`

### Cloud Platforms

The server works with any platform supporting Python ASGI/WSGI applications:
- Heroku
- Railway
- Render
- DigitalOcean App Platform
- AWS Lambda (with adapter)
- Google Cloud Run

## 🔒 Security Notes

- Configure CORS origins properly for production
- Use environment variables for sensitive data
- Consider implementing rate limiting
- Enable HTTPS in production
- Validate all user inputs

## 🎯 Benefits of Consolidation

1. **Simplified Deployment**: Single file to deploy
2. **Reduced Dependencies**: No inter-file imports
3. **Better Performance**: No module loading overhead
4. **Easier Maintenance**: All code in one place
5. **Platform Agnostic**: Works anywhere Python runs
6. **Self-Contained**: Includes all necessary components

This consolidated server provides the same functionality as the original distributed architecture but in a single, powerful, production-ready file that can serve any platform or application type.
