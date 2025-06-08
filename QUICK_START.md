# 🚀 AI Study Buddy - Quick Start Guide

Get your consolidated API server running in minutes!

## ⚡ Super Quick Start (3 steps)

### 1. Setup Environment
```bash
cd api_server
cp .env.example .env
# Edit .env with your actual API keys
```

### 2. Install & Run
```bash
./deploy.sh
# Choose option 1 for development or 2 for production
```

### 3. Test
Open http://localhost:8000/docs in your browser!

## 🔧 Manual Setup

### Prerequisites
- Python 3.8+
- Supabase account with database
- Groq API key

### Step-by-Step

1. **Clone/Navigate to the api_server folder**
   ```bash
   cd api_server
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   nano .env  # Edit with your credentials
   ```

4. **Run the server**
   ```bash
   # Development mode (with auto-reload)
   python consolidated_server.py
   
   # Production mode
   python consolidated_server.py production
   ```

## 🌐 Access Points

Once running, access your API at:

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **API Info**: http://localhost:8000/

## 🧪 Test the API

### Using the Test Client
```bash
python test_client.py
```

### Using curl
```bash
# Health check
curl http://localhost:8000/health

# Create session
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_user", "session_name": "Quick Test"}'

# Send message (replace SESSION_ID)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "SESSION_ID", "content": "Hello, can you help me study?"}'
```

## 🐳 Docker Quick Start

```bash
# Build and run
docker build -t ai-study-buddy .
docker run -p 8000:8000 --env-file .env ai-study-buddy

# Or use docker-compose
docker-compose up
```

## 🔑 Required Environment Variables

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_supabase_anon_key_here
GROQ_API_KEY=your_groq_api_key_here
```

## 📱 Connect Your Apps

### Web App (JavaScript)
```javascript
const API_BASE = 'http://localhost:8000';

// Create session
const session = await fetch(`${API_BASE}/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: 'web_user' })
});

// Chat
const chat = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
        session_id: sessionId, 
        content: 'Explain photosynthesis' 
    })
});
```

### Mobile App (React Native)
```javascript
const createSession = async () => {
    const response = await fetch('http://your-server.com/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: 'mobile_user' })
    });
    return response.json();
};
```

### Python Client
```python
import httpx

async def chat_example():
    async with httpx.AsyncClient() as client:
        # Create session
        session = await client.post('http://localhost:8000/sessions', 
            json={'user_id': 'python_user'})
        
        # Chat
        response = await client.post('http://localhost:8000/chat',
            json={'session_id': session_id, 'content': 'Help me study math'})
```

## 🚨 Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Change port in .env file
   PORT=8001
   ```

2. **Missing dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Database connection error**
   - Check your Supabase URL and key
   - Ensure database tables are created

4. **Groq API error**
   - Verify your Groq API key
   - Check your API quota

### Check Logs
```bash
# Development mode shows logs in console
# Production mode logs to /tmp/aiasis.log
tail -f /tmp/aiasis.log
```

## 🎯 Next Steps

1. **Deploy to production** - See README.md for deployment options
2. **Customize AI personality** - Edit the system prompt in consolidated_server.py
3. **Add authentication** - Implement user authentication as needed
4. **Scale up** - Use load balancers and multiple instances
5. **Monitor** - Set up monitoring and alerting

## 📚 Documentation

- **Full Documentation**: README.md
- **API Reference**: http://localhost:8000/docs (when running)
- **Consolidation Details**: CONSOLIDATION_SUMMARY.md

## 🆘 Need Help?

1. Check the health endpoint: http://localhost:8000/health
2. Review the logs for error messages
3. Ensure all environment variables are set correctly
4. Test with the provided test_client.py

---

**🎉 You're ready to serve multiple platforms with your AI Study Buddy!**
