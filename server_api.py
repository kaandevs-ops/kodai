"""
Turkish-AI HTTP API Server
Ollama ile aynı şekilde çalışır, port 5001'de dinler.
Başlatmak için:
    cd ~/Desktop/turkish_ai
    source venv/bin/activate
    python3 server_api.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import threading

from core.ai_engine import TurkishAI
from autonomous_learner import AutonomousLearner
from scenario_engine import ScenarioEngine

# ── Model yükle ───────────────────────────────────────────
print("🧠 Turkish-AI yükleniyor...")
ai = TurkishAI(model_size="tiny")
ai.initialize()

# ── Otonom öğrenme motorunu başlat ────────────────────────
learner = AutonomousLearner(ai)
learner.start()
scenario = ScenarioEngine()
print("✅ Turkish-AI hazır, port 5001'de dinliyor...")

# ── HTTP Handler ──────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass  # Gereksiz logları kapat

    def do_GET(self):
        if self.path == "/health":
            self._json({"status": "ok", "model": "turkish-ai"})
        elif self.path == "/learner":
            self._json({"status": "success", "learner": learner.get_status()})
        elif self.path == "/learn-now":
            def _run():
                learner._fetch_and_learn_wikipedia()
                learner._fetch_from_node_search()
            threading.Thread(target=_run, daemon=True).start()
            self._json({"status": "success", "message": "Öğrenme tetiklendi"})
        elif self.path == "/stats":
            stats = ai.get_stats()
            stats["learner"]   = learner.get_status()
            stats["scenarios"] = scenario.get_stats()
            self._json({"status": "success", "stats": stats})
        elif self.path == "/scenario/stats":
            self._json({"status": "success", "stats": scenario.get_stats()})
        else:
            self._json({"status": "error", "message": "Bilinmeyen endpoint"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)

        try:
            data = json.loads(body)
        except:
            self._json({"status": "error", "message": "Geçersiz JSON"}, 400)
            return

        # /ask — sohbet endpoint'i
        if self.path == "/ask":
            message     = data.get("message", "")
            temperature = float(data.get("temperature", 0.8))
            max_length  = int(data.get("max_length", 150))
            use_memory  = bool(data.get("use_memory", True))
            use_beam    = bool(data.get("use_beam_search", False))

            if not message:
                self._json({"status": "error", "message": "message gerekli"}, 400)
                return

            try:
                response = ai.chat(
                    message,
                    max_length      = max_length,
                    temperature     = temperature,
                    use_memory      = use_memory,
                    use_beam_search = use_beam
                )
                # Konuşmadan otonom öğren
                try:
                    learner.learn_from_conversation(message, response)
                except Exception:
                    pass

                # Senaryo motoruna kaydet
                try:
                    scenario.record(message, response)
                except Exception:
                    pass

                self._json({
                    "status":   "success",
                    "response": response,
                    "model":    "turkish-ai"
                })
            except RuntimeError as e:
                self._json({"status": "error", "message": str(e)}, 500)

        # /reset — konuşmayı sıfırla
        elif self.path == "/reset":
            ai.reset_conversation()
            self._json({"status": "success", "message": "Konuşma sıfırlandı"})

        # /stats — model istatistikleri
        elif self.path == "/stats":
            stats = ai.get_stats()
            stats["learner"] = learner.get_status()
            self._json({"status": "success", "stats": stats})

        # /learner — otonom öğrenme durumu
        elif self.path == "/learner":
            self._json({"status": "success", "learner": learner.get_status()})

        # /learn-now — hemen öğrenmeyi tetikle
        elif self.path == "/learn-now":
            def _run():
                learner._fetch_and_learn_wikipedia()
                learner._fetch_from_node_search()
            import threading
            threading.Thread(target=_run, daemon=True).start()
            self._json({"status": "success", "message": "Öğrenme tetiklendi"})

        # /learn — geri bildirimle öğren
        elif self.path == "/learn":
            correct  = data.get("correct_response", "")
            positive = bool(data.get("positive", True))
            ai.learn_from_feedback(correct, positive)
            self._json({"status": "success"})

        # /scenario/analyze — senaryo analizi
        elif self.path == "/scenario/analyze":
            situation = data.get("situation", "")
            depth     = int(data.get("depth", 2))
            if not situation:
                self._json({"status": "error", "message": "situation gerekli"}, 400)
                return
            result = scenario.analyze(situation, depth=depth)
            self._json({"status": "success", "analysis": result})

        # /scenario/probability — olasılık hesapla
        elif self.path == "/scenario/probability":
            situation = data.get("situation", "")
            if not situation:
                self._json({"status": "error", "message": "situation gerekli"}, 400)
                return
            prob = scenario.probability(situation)
            self._json({"status": "success", "probability": prob})

        # /scenario/chain — zincirleme senaryo
        elif self.path == "/scenario/chain":
            start = data.get("start", "")
            steps = data.get("steps", [])
            if not start:
                self._json({"status": "error", "message": "start gerekli"}, 400)
                return
            result = scenario.chain(start, steps)
            self._json({"status": "success", "chain": result})

        # /scenario/record — manuel kayıt
        elif self.path == "/scenario/record":
            sit     = data.get("situation", "")
            res     = data.get("result", "")
            outcome = data.get("outcome", "neutral")
            scenario.record(sit, res, outcome)
            self._json({"status": "success"})

        else:
            self._json({"status": "error", "message": "Bilinmeyen endpoint"}, 404)

    def _json(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


# ── Sunucuyu başlat ───────────────────────────────────────
if __name__ == "__main__":
    PORT   = int(os.environ.get("TURKISH_AI_PORT", 5001))
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"🚀 Turkish-AI API çalışıyor: http://localhost:{PORT}")
    print("   Endpoint'ler:")
    print("   POST /ask     — sohbet")
    print("   POST /reset   — konuşmayı sıfırla")
    print("   POST /stats   — istatistikler")
    print("   POST /learn   — geri bildirim")
    print("   GET  /health  — sağlık kontrolü")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Turkish-AI API kapatıldı.")