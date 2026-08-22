"""
Interactive Groq API key verification script.
"""
import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv(override=True)

raw_key = os.getenv("GROQ_API_KEY", "")
key = str(raw_key).strip().strip('"').strip("'")
model_name = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-20b")

print("=" * 60)
print("🔑 GROQ API KEY SANITIZATION & CONNECTIVITY TEST")
print("=" * 60)
print(f"Raw Key Read: {'[EMPTY]' if not raw_key else raw_key[:10] + '...'}")
print(f"Sanitized Key: {'[EMPTY]' if not key else key[:10] + '...'}")
print(f"Key format valid (starts with 'gsk_'): {key.startswith('gsk_')}")
print(f"Target Model: {model_name}")

if not key or key in ["your_api_key_here", "your_groq_api_key_here"]:
    print("\n❌ STATUS: GROQ_API_KEY is not set or still contains a template placeholder.")
    print("👉 Update .env with your real key from https://console.groq.com/keys")
    sys.exit(1)

if not key.startswith("gsk_"):
    print(f"\n❌ STATUS: Invalid key prefix. Groq API keys must begin with 'gsk_'. (Received: '{key[:6]}...')")
    print("👉 Generate a new key at https://console.groq.com/keys")
    sys.exit(1)

try:
    print(f"\n⏳ Testing live ping with Groq Cloud ({model_name})...")
    llm = ChatGroq(model=model_name, api_key=key, temperature=0.0)
    res = llm.invoke("Hi, respond with 'PONG' to confirm connection.")
    print("✅ Live Ping Result:", res.content.strip())
    print("\n🎉 SUCCESS: Groq API key is 100% valid and operational!")
except Exception as e:
    print(f"\n❌ AUTH / MODEL ERROR: {str(e)}")
    print("👉 If you received a 401 Authentication Error, generate a fresh key at: https://console.groq.com/keys")
    sys.exit(1)
