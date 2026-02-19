"""
Helper script to check LM Studio connection and list available models.
Run this first to verify your setup and get exact model names.
"""

import requests
import json

LM_STUDIO_URL = "http://localhost:1234/v1"

def check():
    print("╔══════════════════════════════════════════════════╗")
    print("║       LM Studio Connection Checker              ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    try:
        resp = requests.get(f"{LM_STUDIO_URL}/models", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("data", [])
            print(f"✅ Connected to LM Studio!")
            print(f"📋 Available models ({len(models)}):")
            print()
            for m in models:
                mid = m.get("id", "unknown")
                print(f"   Model ID: {mid}")
                print(f"   Object:   {m.get('object', '?')}")
                print(f"   Owned by: {m.get('owned_by', '?')}")
                print()

            print("─" * 50)
            print("Copy the Model ID values above into config.py")
            print("in the MODELS dictionary.")
            return models
        else:
            print(f"❌ LM Studio returned status {resp.status_code}")
            print(f"   Response: {resp.text[:200]}")
    except requests.ConnectionError:
        print("❌ Cannot connect to LM Studio!")
        print()
        print("Make sure:")
        print("  1. LM Studio is running")
        print("  2. A model is loaded")
        print("  3. The server is started (Settings → Server → Start)")
        print("  4. It's running on port 1234 (default)")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    check()
