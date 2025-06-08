#!/bin/bash

# AI Study Buddy - Consolidated Server Deployment Script

echo "🚀 AI Study Buddy - Consolidated Server Deployment"
echo "=================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    echo "❌ pip is not installed. Please install pip."
    exit 1
fi

# Use pip3 if available, otherwise pip
PIP_CMD="pip3"
if ! command -v pip3 &> /dev/null; then
    PIP_CMD="pip"
fi

echo "✅ Python and pip found"

# Install dependencies
echo "📦 Installing dependencies..."
$PIP_CMD install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✅ Dependencies installed successfully"

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating template..."
    cat > .env << EOF
# AI Study Buddy Configuration
SUPABASE_URL=your_supabase_url_here
SUPABASE_ANON_KEY=your_supabase_anon_key_here
GROQ_API_KEY=your_groq_api_key_here

# Server Configuration (optional)
HOST=0.0.0.0
PORT=8000
ENVIRONMENT=production
EOF
    echo "📝 Please edit .env file with your actual credentials"
    echo "   Then run this script again"
    exit 1
fi

echo "✅ Environment configuration found"

# Validate environment variables
echo "🔍 Validating configuration..."
python3 -c "
import os
from dotenv import load_dotenv
load_dotenv()

required_vars = ['SUPABASE_URL', 'SUPABASE_ANON_KEY', 'GROQ_API_KEY']
missing = [var for var in required_vars if not os.getenv(var) or os.getenv(var) == f'your_{var.lower()}_here']

if missing:
    print(f'❌ Missing or template values in .env: {missing}')
    exit(1)
else:
    print('✅ Configuration validated')
"

if [ $? -ne 0 ]; then
    echo "Please update your .env file with actual values"
    exit 1
fi

# Ask for deployment mode
echo ""
echo "🎯 Select deployment mode:"
echo "1) Development (with auto-reload)"
echo "2) Production (optimized)"
echo "3) Test configuration only"
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo "🔧 Starting in development mode..."
        python3 consolidated_server.py development
        ;;
    2)
        echo "🏭 Starting in production mode..."
        python3 consolidated_server.py production
        ;;
    3)
        echo "🧪 Testing configuration..."
        python3 -c "
from consolidated_server import Config, get_chat_service
import asyncio

async def test():
    try:
        Config.validate()
        print('✅ Configuration valid')
        
        service = get_chat_service()
        health = await service.health_check()
        print(f'✅ Health check: {health}')
        print('🎉 All systems ready!')
    except Exception as e:
        print(f'❌ Test failed: {e}')

asyncio.run(test())
"
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac
