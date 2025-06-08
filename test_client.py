#!/usr/bin/env python3
"""
AI Study Buddy - Test Client
Demonstrates how to use the consolidated API server from different platforms
"""

import asyncio
import httpx
import json
from datetime import datetime

class AIStudyBuddyClient:
    """Client for interacting with AI Study Buddy API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session_id = None
    
    async def health_check(self):
        """Check API health"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/health")
            return response.json()
    
    async def create_session(self, user_id: str, session_name: str = None):
        """Create a new chat session"""
        async with httpx.AsyncClient() as client:
            data = {"user_id": user_id}
            if session_name:
                data["session_name"] = session_name
            
            response = await client.post(f"{self.base_url}/sessions", json=data)
            if response.status_code == 200:
                session_data = response.json()
                self.session_id = session_data["id"]
                return session_data
            else:
                raise Exception(f"Failed to create session: {response.text}")
    
    async def send_message(self, content: str):
        """Send a message and get AI response"""
        if not self.session_id:
            raise Exception("No active session. Create a session first.")
        
        async with httpx.AsyncClient() as client:
            data = {
                "session_id": self.session_id,
                "content": content
            }
            
            response = await client.post(f"{self.base_url}/chat", json=data)
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Failed to send message: {response.text}")
    
    async def get_message_history(self):
        """Get message history for current session"""
        if not self.session_id:
            raise Exception("No active session. Create a session first.")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/sessions/{self.session_id}/messages")
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Failed to get message history: {response.text}")
    
    async def get_user_sessions(self, user_id: str):
        """Get all sessions for a user"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/users/{user_id}/sessions")
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Failed to get user sessions: {response.text}")

async def demo_web_client():
    """Demonstrate web client usage"""
    print("🌐 Web Client Demo")
    print("=" * 50)
    
    client = AIStudyBuddyClient()
    
    try:
        # Health check
        print("🔍 Checking API health...")
        health = await client.health_check()
        print(f"✅ API Status: {health['status']}")
        
        # Create session
        print("\n📝 Creating new session...")
        session = await client.create_session("web_user_123", "Physics Study Session")
        print(f"✅ Session created: {session['session_name']} (ID: {session['id'][:8]}...)")
        
        # Send messages with context
        context_message = """
        السياق: الفيزياء الكمية
        
        الفيزياء الكمية هي فرع من الفيزياء يدرس سلوك المادة والطاقة على المستوى الذري والجزيئي. 
        تتضمن المفاهيم الأساسية:
        
        1. التكميم (Quantization): الطاقة تأتي في حزم منفصلة تسمى الكوانتا
        2. مبدأ عدم اليقين (Uncertainty Principle): لا يمكن تحديد موقع وسرعة الجسيم بدقة في نفس الوقت
        3. الازدواجية الموجية-الجسيمية (Wave-Particle Duality): الضوء والمادة لها خصائص موجية وجسيمية
        
        المعادلة الأساسية: E = hf حيث E الطاقة، h ثابت بلانك، f التردد
        """
        
        print("\n💬 Sending context and question...")
        response = await client.send_message(context_message)
        print(f"🤖 AI: {response['assistant_message']['content'][:100]}...")
        
        # Ask a question about the context
        question = "اشرح لي مبدأ عدم اليقين بطريقة بسيطة"
        print(f"\n❓ Question: {question}")
        response = await client.send_message(question)
        print(f"🤖 AI Response:\n{response['assistant_message']['content']}")
        
        # Get message history
        print("\n📚 Getting message history...")
        history = await client.get_message_history()
        print(f"✅ Retrieved {len(history['messages'])} messages")
        
    except Exception as e:
        print(f"❌ Error: {e}")

async def demo_mobile_client():
    """Demonstrate mobile client usage"""
    print("\n📱 Mobile Client Demo")
    print("=" * 50)
    
    client = AIStudyBuddyClient()
    
    try:
        # Create session for mobile user
        session = await client.create_session("mobile_user_456", "Math Study")
        print(f"✅ Mobile session created: {session['id'][:8]}...")
        
        # Math context in English
        math_context = """
        Context: Linear Algebra - Matrices
        
        A matrix is a rectangular array of numbers arranged in rows and columns.
        
        Basic operations:
        1. Matrix Addition: Add corresponding elements
        2. Matrix Multiplication: Row by column multiplication
        3. Transpose: Flip rows and columns
        
        Example 2x2 matrix:
        A = [1 2]
            [3 4]
        
        Determinant of 2x2 matrix: det(A) = ad - bc
        """
        
        response = await client.send_message(math_context)
        print(f"🤖 Context processed")
        
        # Ask question
        question = "How do I calculate the determinant of the given matrix?"
        response = await client.send_message(question)
        print(f"📱 Mobile AI Response:\n{response['assistant_message']['content']}")
        
    except Exception as e:
        print(f"❌ Mobile demo error: {e}")

async def demo_desktop_client():
    """Demonstrate desktop client usage"""
    print("\n🖥️  Desktop Client Demo")
    print("=" * 50)
    
    client = AIStudyBuddyClient()
    
    try:
        # Create session for desktop user
        session = await client.create_session("desktop_user_789", "Chemistry Study")
        print(f"✅ Desktop session created: {session['id'][:8]}...")
        
        # Chemistry context
        chemistry_context = """
        السياق: الكيمياء العضوية - الهيدروكربونات
        
        الهيدروكربونات هي مركبات تتكون من الكربون والهيدروجين فقط.
        
        الأنواع الرئيسية:
        1. الألكانات (Alkanes): روابط أحادية، الصيغة العامة CnH2n+2
        2. الألكينات (Alkenes): رابطة مزدوجة واحدة، الصيغة العامة CnH2n  
        3. الألكاينات (Alkynes): رابطة ثلاثية واحدة، الصيغة العامة CnH2n-2
        
        أمثلة:
        - الميثان CH4 (أبسط ألكان)
        - الإيثين C2H4 (أبسط ألكين)
        - الإيثاين C2H2 (أبسط ألكاين)
        """
        
        response = await client.send_message(chemistry_context)
        print(f"🤖 Context processed")
        
        # Ask question in Arabic
        question = "ما الفرق بين الألكانات والألكينات من ناحية التركيب؟"
        response = await client.send_message(question)
        print(f"🖥️  Desktop AI Response:\n{response['assistant_message']['content']}")
        
    except Exception as e:
        print(f"❌ Desktop demo error: {e}")

async def main():
    """Run all client demos"""
    print("🚀 AI Study Buddy - Multi-Platform Client Demo")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        await demo_web_client()
        await demo_mobile_client()
        await demo_desktop_client()
        
        print("\n🎉 All demos completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
