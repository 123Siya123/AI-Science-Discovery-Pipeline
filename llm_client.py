"""
LLM Client for the AI Science Discovery Team.
Handles all communication with LM Studio's OpenAI-compatible API.

This module is designed to be robust for long-running "First Principles" reasoning
chains. It includes:
1. Smart Retries: Handling connection drops (common with local LLMs).
2. Streaming: To allow the user to interrupt the process if the physics drift.
3. Model Agnostic: Can switch between different models if available.
"""

import json
import time
import requests
from config import LM_STUDIO_BASE_URL, LM_STUDIO_API_KEY


class LLMClient:
    """Client for communicating with LM Studio's OpenAI-compatible API."""

    def __init__(self, base_url="http://127.0.0.1:1234/v1", api_key="lm-studio", log_callback=None):
        self.base_url = base_url
        self.api_key = api_key
        self.log = log_callback or print
        self._current_model = None

    def _headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def check_connection(self):
        """Check if LM Studio is running and accessible."""
        try:
            resp = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                model_ids = [m.get("id", "unknown") for m in models]
                self.log(f"✅ Connected to LM Studio. Available models: {model_ids}")
                return True, model_ids
            else:
                self.log(f"❌ LM Studio returned status {resp.status_code}")
                return False, []
        except requests.ConnectionError:
            self.log("❌ Cannot connect to LM Studio. Is it running on port 1234?")
            return False, []
        except Exception as e:
            self.log(f"❌ Error connecting to LM Studio: {e}")
            return False, []

    def get_loaded_model(self):
        """Get the currently loaded model in LM Studio."""
        try:
            resp = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                if models:
                    return models[0].get("id", "unknown")
            return None
        except Exception:
            return None

    def chat(self, system_prompt, user_message, model_id, temperature=0.7,
             max_tokens=2000, top_p=0.9, retry_count=10, stop_callback=None):
        """
        Send a chat completion request to LM Studio.
        Uses streaming internally to allow for immediate interruption.
        
        Args:
            stop_callback: Optional function that returns True if we should abort.
        """
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": True,  # Always stream to allow interruption
        }

        for attempt in range(retry_count + 1):
            if stop_callback and stop_callback():
                self.log("🛑 Request executed but stop signal received before starting.")
                return None

            try:
                self.log(f"🤖 Calling model: {model_id} (attempt {attempt + 1})")
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                    timeout=7200,  # 2 hours — slow hardware needs patience
                    stream=True,
                )

                if resp.status_code != 200:
                    self.log(f"⚠️ API returned status {resp.status_code}: {resp.text}")
                    if attempt < retry_count:
                        time.sleep(2 ** attempt)
                    continue

                # Collect stream content
                full_content = ""
                for line in resp.iter_lines():
                    if stop_callback and stop_callback():
                        self.log("🛑 Request interrupted by user stop signal.")
                        return None
                        
                    if line:
                        line_str = line.decode("utf-8")
                        if line_str.startswith("data: "):
                            data_str = line_str[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                choices = data.get("choices", [])
                                if choices and len(choices) > 0:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        full_content += content
                            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                                continue
                
                return full_content

            except requests.Timeout:
                self.log(f"⏱️ Request timed out after 2 hours (attempt {attempt + 1}/{retry_count + 1})")
            except requests.ConnectionError:
                self.log(f"❌ Lost connection to LM Studio (attempt {attempt + 1}/{retry_count + 1}). "
                         f"LM Studio might be loading a model — will retry in 30s...")
                if attempt < retry_count:
                    time.sleep(30)  # LM Studio model loading can take a while
                    continue
            except Exception as e:
                self.log(f"❌ Error: {e} (attempt {attempt + 1}/{retry_count + 1})")
            
            if attempt < retry_count:
                wait_time = min(10 * (attempt + 1), 60)  # 10s, 20s, 30s... up to 60s
                self.log(f"⏳ Waiting {wait_time}s before retry...")
                time.sleep(wait_time)

        self.log("❌ All retry attempts failed")
        return None

    def chat_streaming(self, system_prompt, user_message, model_id,
                       temperature=0.7, max_tokens=2000, top_p=0.9,
                       chunk_callback=None):
        """
        Send a streaming chat completion request.
        Calls chunk_callback with each text chunk as it arrives.
        Returns the complete response text.
        """
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": True,
        }

        full_response = ""
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=7200,  # 2 hours — slow hardware needs patience
                stream=True,
            )

            if resp.status_code != 200:
                self.log(f"⚠️ Streaming API returned status {resp.status_code}")
                return None

            for line in resp.iter_lines():
                if line:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            choices = data.get("choices", [])
                            if choices and len(choices) > 0:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    full_response += content
                                    if chunk_callback:
                                        chunk_callback(content)
                        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                            continue

            return full_response

        except Exception as e:
            self.log(f"❌ Streaming error: {e}")
            return full_response if full_response else None
