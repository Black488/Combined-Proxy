#!/usr/bin/env python3
import os, json, time, logging
from flask import Flask, request, Response, stream_with_context
import requests

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("PROXY")
app = Flask(__name__)

def get_keys(var):
    return [k.strip() for k in os.environ.get(var, "").split(",") if k.strip()]

NAVY_KEYS = get_keys("NAVY_KEYS")
NVIDIA_KEYS = get_keys("NVIDIA_KEYS")
KENARI_KEYS = get_keys("KENARI_KEYS")
TOKENREPLY_KEYS = get_keys("TOKENREPLY_KEYS")
ZEN_KEYS = get_keys("ZEN_KEYS")
KIOS_KEYS = get_keys("KIOS_KEYS")

BASE_URLS = {
    "kenari": "https://kenari.id/v1",
    "navy": os.environ.get("NAVY_API_BASE", ""),
    "nvidia": os.environ.get("NVIDIA_API_BASE", ""),
    "tokenreply": "https://api.tokenreply.com/v1",
    "zen": "https://opencode.ai/zen/v1",
    "kios": "https://router.kiosapi.com/v1",
}

class KeyRotator:
    def __init__(self, keys, cooldown=30):
        self.keys = keys
        self.cooldown = cooldown
        self.failed = {}
        self.idx = 0
    def get(self):
        if not self.keys: return None, None
        now = time.time()
        for _ in range(len(self.keys)):
            i = self.idx
            self.idx = (self.idx + 1) % len(self.keys)
            if i not in self.failed or (now - self.failed[i]) > self.cooldown:
                return i, self.keys[i]
        return None, None
    def mark_bad(self, i): self.failed[i] = time.time()
    def mark_good(self, i): self.failed.pop(i, None)

rotators = {
    "kenari": KeyRotator(KENARI_KEYS, 30),
    "navy": KeyRotator(NAVY_KEYS, 30),
    "nvidia": KeyRotator(NVIDIA_KEYS, 10),
    "tokenreply": KeyRotator(TOKENREPLY_KEYS, 30),
    "zen": KeyRotator(ZEN_KEYS, 30),
    "kios": KeyRotator(KIOS_KEYS, 30),
}

ROUTING = {
    "[Navy] GPT-5.1 (2.5x)": ("navy", "gpt-5.1"),
    "[Navy] GPT-5.4 (4.5x)": ("navy", "gpt-5.4"),
    "[Navy] Mimo-v2.5-pro (1.5x)": ("navy", "mimo-v2.5-pro"),
    "[Navy] Gemini 3 Flash (°REAS)": ("navy", "gemini-3-flash-preview-thinking"),
    "[Navy] Llama 4 Scout (10M)": ("navy", "llama-4-scout"),
    "[Navy] Hermes 4 405B (131k) (4x)": ("navy", "hermes-4-405b"),
    "[NVIDIA] MiniMax M3": ("nvidia", "minimaxai/minimax-m3"),
    "[NVIDIA] Mistral 3 (256k)": ("nvidia", "mistralai/mistral-large-3-675b-instruct-2512"),
    "[NVIDIA] Nemotron 3 Ultra": ("nvidia", "nvidia/nemotron-3-ultra-550b-a55b"),
    "[NVIDIA] DeepSeek V4 Flash (0731)": ("nvidia", "deepseek-v4-flash-0731"),
    "[TokenReply] DeepSeek V4 Pro": ("tokenreply", "deepseek-ai/deepseek-v4-pro"),
    "[TokenReply] GLM 5.2": ("tokenreply", "z-ai/glm-5.2"),
    "[TokenReply] Grok 4.20 Fast": ("tokenreply", "grok-4.20-fast"),
    "[Zen] Mimo-v2.5": ("zen", "mimo-v2.5-free"),
    "[Kenari] Hy3": ("kenari", "hy3:free"),
    "[Kenari] GLM 4.7 Flash": ("kenari", "glm-4-7-flash:free"),
    "[Kenari] Kimi K2.6": ("kenari", "kimi-k2-6:free"),
    "[Kios] Mimo-v2.5": ("kios", "oc/mimo-v2.5"),
    "[Kios] GLM 5.3 Flash": ("kios", "glm-5.3-flash"),
    "[Kios] Grok 4.6": ("kios", "grok-4.6"),
    "[Kios] Kimi K3": ("kios", "kimi-k3"),
    "[Kios] Qwen 3.8 Flash": ("kios", "qwen3.8-flash"),
    "[Kios] Muse Spark 1.3 Contributor": ("kios", "oc/muse-spark-1.3-contributor"),
}

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = request.headers.get('Origin', '*')
    response.headers['Access-Control-Allow-Headers'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response

@app.route('/v1/models', methods=['GET', 'OPTIONS'])
def list_models():
    if request.method == "OPTIONS": return Response(status=204)
    return Response(json.dumps({"object": "list", "data": [{"id": k, "object": "model"} for k in ROUTING]}), mimetype="application/json; charset=utf-8")

@app.route('/models', methods=['GET', 'OPTIONS'])
def root_models():
    if request.method == "OPTIONS": return Response(status=204)
    return list_models()

@app.route('/v1/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
def proxy(path):
    if request.method == "OPTIONS": return Response(status=204)
    body = request.get_data()
    display_name = None
    backend = "navy"
    if "chat/completions" in path and request.method == "POST" and body:
        try:
            data = json.loads(body)
            disp = data.get("model")
            if disp in ROUTING:
                backend, upstream_id = ROUTING[disp]
                data["model"] = upstream_id
                body = json.dumps(data).encode('utf-8')
                display_name = disp
        except Exception as e: log.error(f"JSON parse failed: {e}")

    rotator = rotators.get(backend, rotators["navy"])
    base_url = BASE_URLS.get(backend, "")
    if not rotator.keys: return Response(json.dumps({"error": f"No {backend} keys"}), status=500, mimetype="application/json")

    max_tries = min(len(rotator.keys), 3)
    last_err = b'{"error": "All keys failed"}'
    last_code = 503
    fwd_headers = {k: v for k, v in request.headers if k.lower() not in ['host', 'content-length', 'authorization', 'content-encoding']}
    fwd_headers['User-Agent'] = 'Mozilla/5.0'

    for _ in range(max_tries):
        idx, key = rotator.get()
        if idx is None: return Response(json.dumps({"error": "All keys on cooldown"}), status=429, mimetype="application/json")
        headers = {**fwd_headers, "Authorization": f"Bearer {key}"}
        try:
            resp = requests.request(request.method, f"{base_url}/{path}", headers=headers, data=body, stream=True, timeout=180)
            resp.encoding = 'utf-8'
            if resp.status_code in (400, 429, 500, 502, 503, 504):
                rotator.mark_bad(idx); last_err = resp.content; last_code = resp.status_code; continue
            rotator.mark_good(idx)
            safe_hdrs = {k: v for k, v in resp.headers.items() if k.lower() not in ['content-encoding', 'transfer-encoding', 'connection', 'content-length']}
            ct = resp.headers.get('content-type', '')
            if display_name and 'application/json' in ct:
                try:
                    obj = resp.json()
                    if 'model' in obj: obj['model'] = display_name
                    safe_hdrs['Content-Type'] = 'application/json; charset=utf-8'
                    return Response(json.dumps(obj), status=resp.status_code, headers=safe_hdrs)
                except Exception: return Response(resp.content, status=resp.status_code, headers=safe_hdrs)
            if display_name and 'text/event-stream' in ct:
                def generate():
                    try:
                        for line in resp.iter_lines(decode_unicode=True):
                            if line and line.startswith("data: ") and line != "data: [DONE]":
                                try:
                                    chunk = json.loads(line[6:])
                                    if 'model' in chunk: chunk['model'] = display_name
                                    yield "data: " + json.dumps(chunk) + "\n\n"
                                except Exception: yield line + "\n\n"
                            else:
                                if line: yield line + "\n\n"
                    except Exception as e: log.error(f"Stream dropped: {e}")
                safe_hdrs['Content-Type'] = 'text/event-stream; charset=utf-8'
                return Response(stream_with_context(generate()), status=resp.status_code, headers=safe_hdrs)
            if 'charset' not in ct.lower(): safe_hdrs['Content-Type'] = ct + '; charset=utf-8'
            return Response(stream_with_context(resp.iter_content(chunk_size=4096)), status=resp.status_code, headers=safe_hdrs)
        except Exception as e:
            rotator.mark_bad(idx); last_err = str(e).encode('utf-8'); last_code = 502
    return Response(last_err, status=last_code, mimetype="application/json; charset=utf-8")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    log.info(f"Starting Main Proxy on port {port}")
    app.run(host='0.0.0.0', port=port, threaded=True)