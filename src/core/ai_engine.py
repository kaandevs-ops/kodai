"""
KodAI Ana Motor v5 — Qwen2.5-Coder Edition
Komuta özel sistem promptları + Gelişmiş bağlam yönetimi
"""
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import os
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
from typing import List, Dict, Optional, Callable
from datetime import datetime

# ─── SİSTEM PROMPTLARI ──────────────────────────────────────────────────────

PROMPT_GENEL = """Sen KodAI'sın — güçlü bir Türkçe kod asistanı. Mac M4 üzerinde yerel çalışıyorsun.
KURALLAR:
1. Her zaman Türkçe yanıt ver
2. Kod bloklarını ```dil ... ``` formatında yaz
3. Kısa ve öz ol
4. Türkçe teknik terimler için parantez içinde İngilizce yaz
UZMANLIK: Python, JavaScript, TypeScript, Bash, veri yapıları, algoritmalar, debugging, refactoring, test, git, API"""

PROMPT_KOD_YAZ = """Sen KodAI'sın — kod yazan bir asistan.
KURALLAR:
1. SADECE kod yaz, gereksiz açıklama ekleme
2. Çalışan, temiz, yorumlu kod üret
3. ```python ... ``` formatını kullan
4. Fonksiyonlara kısa Türkçe docstring ekle"""

PROMPT_HATA = """Sen KodAI'sın — hata analiz eden bir asistan.
FORMAT: Her zaman şu sırayla yanıtla:
1. **Hata Nedeni:** (1 cümle)
2. **Çözüm:** (adım adım)
3. **Düzeltilmiş Kod:** (varsa)
Türkçe yaz, kısa tut."""

PROMPT_REFACTOR = """Sen KodAI'sın — kod iyileştiren bir asistan.
YAPILACAKLAR:
1. Gereksiz tekrarları kaldır (DRY prensibi)
2. Fonksiyon isimlerini anlamlı yap
3. Magic number'ları sabit olarak tanımla
4. Type hint ekle
5. Türkçe olarak ne değiştirdiğini açıkla, sonra kodu yaz"""

PROMPT_ACIKLA = """Sen KodAI'sın — kod açıklayan bir asistan.
Kodu yeni başlayanlar anlayacak şekilde Türkçe açıkla.
Her önemli satır veya blok için ne yaptığını söyle.
Teknik terimlerin Türkçe karşılığını ver."""

PROMPT_TEST = """Sen KodAI'sın — test yazan bir asistan.
KURALLAR:
1. pytest formatında yaz
2. Edge case'leri test et (boş input, None, hata durumları)
3. Her test fonksiyonuna Türkçe yorum ekle
4. Test isimlerini `test_ne_yapar` formatında yaz"""

PROMPT_GIT = """Sen KodAI'sın — git konusunda yardımcı olan bir asistan.
Kısa, net, uygulanabilir git komutları ver.
Komutların ne yaptığını Türkçe açıkla."""

KOMUT_PROMPTLARI = {
    "yaz": PROMPT_KOD_YAZ,
    "hata": PROMPT_HATA,
    "refactor": PROMPT_REFACTOR,
    "acikla": PROMPT_ACIKLA,
    "test": PROMPT_TEST,
    "git": PROMPT_GIT,
    "genel": PROMPT_GENEL,
}


class TurkishAI:
    def __init__(self, model_size: str = "small", device: str = None, memory_dir: str = "./data/memory"):
        self.model_size = model_size
        self.device = device or (
            "mps" if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.memory_dir = memory_dir
        self.model = None
        self.tokenizer = None
        self.is_initialized = False
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conversation_history: List[Dict] = []
        self.model_name = ""
        self.toplam_token = 0
        print(f"\n{'='*60}")
        print(f"🧠 KodAI v6 Başlatılıyor")
        print(f"Cihaz: {self.device}")
        print(f"{'='*60}\n")

    def initialize(self, load_checkpoint: str = None, model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct"):
        self.model_name = model_name
        print(f"[1/2] Model yükleniyor: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16,
        ).to(self.device)
        self.model.eval()
        print("[2/2] Sistem hazırlanıyor...")
        os.makedirs(self.memory_dir, exist_ok=True)
        self.is_initialized = True
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"\n✅ KodAI v6 hazır! Parametre: {param_count:,}")

    def _prompt_sec(self, komut_tipi: str = "genel") -> str:
        return KOMUT_PROMPTLARI.get(komut_tipi, PROMPT_GENEL)

    def _mesajlari_hazirla(self, message: str, komut_tipi: str = "genel") -> str:
        sistem = self._prompt_sec(komut_tipi)
        messages = [{"role": "system", "content": sistem}]
        # Bağlam: genel sohbette geçmişi ekle, kod komutlarında ekleme (token tasarrufu)
        if komut_tipi == "genel" and self.conversation_history:
            messages.extend(self.conversation_history[-6:])
        messages.append({"role": "user", "content": message})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def chat(self, message: str, max_length: int = 512, temperature: float = 0.7,
             top_p: float = 0.95, use_memory: bool = True, komut_tipi: str = "genel") -> str:
        if not self.is_initialized:
            raise RuntimeError("Sistem başlatılmamış.")
        text = self._mesajlari_hazirla(message, komut_tipi)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_length, temperature=temperature,
                top_p=top_p, do_sample=True, pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        output_ids = outputs[0][len(inputs.input_ids[0]):]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        if not response:
            response = "Yanıt üretilemedi."
        self.toplam_token += len(output_ids)
        if use_memory and komut_tipi == "genel":
            self.conversation_history.append({"role": "user", "content": message})
            self.conversation_history.append({"role": "assistant", "content": response})
            if len(self.conversation_history) > 16:
                self.conversation_history = self.conversation_history[-16:]
        return response

    def stream_chat(self, message: str, max_length: int = 512, temperature: float = 0.7,
                    top_p: float = 0.95, use_memory: bool = True, komut_tipi: str = "genel",
                    callback: Callable[[str], None] = None) -> str:
        if not self.is_initialized:
            raise RuntimeError("Sistem başlatılmamış.")
        text = self._mesajlari_hazirla(message, komut_tipi)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            **inputs, streamer=streamer, max_new_tokens=max_length,
            temperature=temperature, top_p=top_p, do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id, repetition_penalty=1.1,
        )
        thread = Thread(target=self.model.generate, kwargs=generate_kwargs)
        thread.start()
        full_response = ""
        for chunk in streamer:
            full_response += chunk
            if callback:
                callback(chunk)
        thread.join()
        full_response = full_response.strip() or "Yanıt üretilemedi."
        self.toplam_token += len(full_response.split())
        if use_memory and komut_tipi == "genel":
            self.conversation_history.append({"role": "user", "content": message})
            self.conversation_history.append({"role": "assistant", "content": full_response})
            if len(self.conversation_history) > 16:
                self.conversation_history = self.conversation_history[-16:]
        return full_response

    def reset_conversation(self):
        self.conversation_history = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def get_stats(self) -> Dict:
        return {
            "model": self.model_name.split("/")[-1] if self.model_name else "-",
            "parametreler": f"{sum(p.numel() for p in self.model.parameters()):,}" if self.model else "0",
            "kelime_hazinesi": str(len(self.tokenizer)) if self.tokenizer else "0",
            "cihaz": self.device,
            "oturum": self.session_id,
            "bağlam": str(len(self.conversation_history) // 2),
            "toplam_token": str(self.toplam_token),
        }

def create_ai(model_size: str = "small", **kwargs) -> "TurkishAI":
    ai = TurkishAI(model_size=model_size, **kwargs)
    ai.initialize()
    return ai
