"""
LLM Helper for SignFlow — sentence correction & next-word prediction.

Supports two backends:
  1. Local: Ollama + Mistral 7B (offline, requires Ollama installed)
  2. Online: Google Gemini API (free tier, requires API key)

Usage:
  helper = create_llm_helper("local")   # Ollama
  helper = create_llm_helper("gemini", api_key="...")  # Gemini
  result = helper.enhance("I want eat")
  # → {"corrected": "I want to eat", "suggestions": ["food", "pizza", "something"]}
"""

import json
import os
import re
import threading
import time
import traceback


# ── System prompt shared by both backends ──
SYSTEM_PROMPT = """You are a grammar assistant for an ASL (American Sign Language) recognition system.

The user signs words one by one. ASL grammar differs from English, so the raw words may be ungrammatical.

Your job:
1. **Correct** the sentence into natural English (fix grammar, add missing words like "to", "a", "the")
2. **Suggest** 3 likely next words the user might sign

Rules:
- Keep corrections minimal — don't change meaning, just fix grammar
- If the sentence is already correct, return it as-is
- Suggestions should be common everyday words
- Respond ONLY in this exact JSON format, nothing else:

{"corrected": "the corrected sentence", "suggestions": ["word1", "word2", "word3"]}"""


def _parse_response(text):
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    # Remove markdown code blocks
    if "```" in text:
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    # Try to find JSON object
    match = re.search(r'\{[^{}]*"corrected"[^{}]*\}', text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        data = json.loads(text)
        corrected = str(data.get("corrected", ""))
        suggestions = data.get("suggestions", [])
        if isinstance(suggestions, list):
            suggestions = [str(s).strip().lower() for s in suggestions[:5]]
        else:
            suggestions = []
        return {"corrected": corrected, "suggestions": suggestions}
    except (json.JSONDecodeError, KeyError):
        return None


class OllamaLLM:
    """Local LLM via Ollama (Mistral 7B)."""

    def __init__(self, model="mistral"):
        self.model = model
        self.base_url = "http://localhost:11434"
        self.name = f"Ollama ({model})"
        self._check_connection()

    def _check_connection(self):
        """Verify Ollama is running and model is available."""
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())
                models = [m["name"].split(":")[0] for m in data.get("models", [])]
                if self.model not in models:
                    print(f"  ⚠ Model '{self.model}' not found. Available: {models}")
                    print(f"  Run: ollama pull {self.model}")
                    raise RuntimeError(f"Model {self.model} not pulled")
                print(f"  ✓ Ollama connected — {self.model}")
        except Exception as e:
            if "Connection refused" in str(e) or "urlopen" in str(e):
                print(f"  ⚠ Ollama not running! Start it with: ollama serve")
                print(f"  Then pull model: ollama pull {self.model}")
            raise

    def enhance(self, raw_sentence):
        """Call Ollama API to correct sentence and suggest next words."""
        import urllib.request
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Signed words: {raw_sentence}"}
            ],
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 100}
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            text = data.get("message", {}).get("content", "")
            return _parse_response(text)


class GeminiLLM:
    """Online LLM via Google Gemini API (free tier)."""

    def __init__(self, api_key=None, model="gemini-2.0-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = model
        self.name = f"Gemini ({model})"
        if not self.api_key:
            raise ValueError(
                "Gemini API key required!\n"
                "  Get free key: https://aistudio.google.com/apikey\n"
                "  Then: set GEMINI_API_KEY=your_key\n"
                "  Or:   python sign_inference.py --llm gemini --api-key YOUR_KEY"
            )
        self._check_connection()

    def _check_connection(self):
        """Quick test that the API key works."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._genai = genai
            self._model = genai.GenerativeModel(
                self.model,
                system_instruction=SYSTEM_PROMPT
            )
            print(f"  ✓ Gemini API connected — {self.model}")
        except ImportError:
            raise ImportError(
                "google-generativeai not installed!\n"
                "  Run: pip install google-generativeai"
            )

    def enhance(self, raw_sentence):
        """Call Gemini API to correct sentence and suggest next words."""
        response = self._model.generate_content(
            f"Signed words: {raw_sentence}",
            generation_config={"temperature": 0.3, "max_output_tokens": 100}
        )
        return _parse_response(response.text)


class LLMWorker:
    """Background thread that handles LLM calls without blocking the webcam loop."""

    def __init__(self, llm):
        self.llm = llm
        self._lock = threading.Lock()
        self._request = None      # sentence to process
        self._result = None       # latest result
        self._last_sentence = ""  # avoid duplicate calls
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            sentence = None
            with self._lock:
                if self._request and self._request != self._last_sentence:
                    sentence = self._request
                    self._last_sentence = sentence
                    self._request = None

            if sentence:
                try:
                    t0 = time.time()
                    result = self.llm.enhance(sentence)
                    dt = time.time() - t0
                    if result:
                        result["latency"] = dt
                        with self._lock:
                            self._result = result
                except Exception:
                    traceback.print_exc()
            else:
                time.sleep(0.05)  # small sleep to avoid busy-waiting

    def request(self, sentence):
        """Submit a sentence for LLM processing (non-blocking)."""
        with self._lock:
            self._request = sentence

    def get_result(self):
        """Get the latest LLM result (or None if not ready)."""
        with self._lock:
            return self._result

    def clear(self):
        """Clear results (e.g., when sentence is cleared)."""
        with self._lock:
            self._result = None
            self._last_sentence = ""
            self._request = None

    def stop(self):
        self._running = False


def create_llm_helper(backend, api_key=None, model=None):
    """
    Create an LLM helper.
    
    Args:
        backend: "local" (Ollama), "gemini" (Google), or "off"
        api_key: API key for Gemini
        model: Override model name
    
    Returns:
        LLMWorker wrapping the chosen backend, or None if "off"
    """
    if backend == "off" or backend is None:
        return None

    print(f"\nInitializing LLM ({backend})...")
    try:
        if backend == "local":
            llm = OllamaLLM(model=model or "mistral")
        elif backend == "gemini":
            llm = GeminiLLM(api_key=api_key, model=model or "gemini-2.0-flash")
        else:
            print(f"  Unknown backend: {backend}. Use 'local', 'gemini', or 'off'")
            return None

        worker = LLMWorker(llm)
        print(f"  ✓ LLM ready: {llm.name}\n")
        return worker

    except Exception as e:
        print(f"  ⚠ LLM failed to initialize: {e}")
        print(f"  Continuing without LLM...\n")
        return None
