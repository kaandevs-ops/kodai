import sys
sys.path.append("./src")
from core.ai_engine import TurkishAI

print("Veri yükleniyor...")
with open("./datasets/wiki_clean.txt", encoding="utf-8") as f:
    raw = f.read()

texts = [p.strip() for p in raw.split("\n\n---\n\n") if len(p.strip()) > 50]
texts = texts[:20000]
print(f"Kullanılacak paragraf: {len(texts)}")

ai = TurkishAI(model_size="small", vocab_size=4000)
ai.initialize()
ai.train_tokenizer(texts)
print("Tokenizer hazır!")
