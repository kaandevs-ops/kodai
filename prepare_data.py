import os, json, sys
sys.path.append("./src")
from data.data_collector import DataCleaner, DatasetManager

print("Veri okunuyor...")
cleaner = DataCleaner(min_length=100, max_length=1500)
manager = DatasetManager("./datasets")

all_texts = []
wiki_dir = "./datasets/wiki_output"

for root, dirs, files in os.walk(wiki_dir):
    for fname in files:
        filepath = os.path.join(root, fname)
        try:
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if obj.get("text") and len(obj["text"]) > 100:
                        all_texts.append(obj["text"])
        except:
            pass

print(f"Ham paragraf sayısı: {len(all_texts)}")
cleaned = cleaner.clean_dataset(all_texts)
manager.save_raw_texts(cleaned, "wiki_clean.txt")
print(f"Temiz paragraf: {len(cleaned)}")
print("Kaydedildi: ./datasets/wiki_clean.txt")
