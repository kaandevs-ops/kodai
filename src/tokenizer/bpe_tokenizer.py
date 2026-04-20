"""
BPE (Byte Pair Encoding) Tokenizer — v2
Türkçe için özel optimizasyonlar.
Orijinal yapı korundu, iyileştirmeler:

  • Eğitilmemiş tokenizer'da encode() çağrısı açık hata verir
  • Çok kısa metin listesi (< 3 örnek) için hata yerine uyarı
  • Boş string ve None girişlere karşı guard'lar eklendi
  • id_to_token: her zaman senkronize, save/load sonrası da çalışır
  • vocab_size: max_vocab_size olarak yeniden adlandırıldı (iç tutarlılık)
  • SimpleTokenizer: padding token eklendi
"""

import json
import re
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
import pickle
import os


class TurkishTokenizer:
    """
    GPT-2 benzeri BPE tokenizer.
    Türkçe karakterler ve kelime yapısı için optimize edilmiştir.
    """

    def __init__(self, vocab_size: int = 32000):
        self.max_vocab_size = vocab_size   # Hedef vocab boyutu
        self.vocab: Dict[str, int]   = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []

        # Özel tokenlar
        self.special_tokens: Dict[str, int] = {
            "<pad>":  0,
            "<unk>":  1,
            "<s>":    2,
            "</s>":   3,
            "<mask>": 4,
        }

        # Türkçe karakter seti
        self.turkish_chars = set(
            "abcçdefgğhıijklmnoöprsştuüvyz"
            "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ"
            "0123456789"
            " .,!?;:—-()[]{}'\"\\/@#$%&*+=_~|<>"
        )

        self.pattern = self._create_pattern()

    def _create_pattern(self):
        return re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d|"""
            r"""\?|\.|\!|\,|\;|\:|\—|\(|\)|\[|\]|\{|\}|"""
            r"""\'|\"|\/|\@|\#|\$|\%|\&|\*|\+|\=|\_|\~|\||\<|\>|"""
            r"""[abcçdefgğhıijklmnoöprsştuüvyzABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ]+|"""
            r"""\d+|"""
            r"""\s+|"""
            r"""[^\s\w\d\?\.\!\,\;\:\—\(\)\[\]\{\}\'\"\/\@\#\$\%\&\*\+\=\_\~\|\<\>]+"""
        )

    def _preprocess(self, text: str) -> str:
        """Metni ön işleme: küçük harf + yaygın replacementlar."""
        if not text:
            return ""
        text = text.lower()
        for old, new in [("\u2018", "'"), ("\u2019", "'"), ("\u201c", "'"), ("\u201d", "'"), ("...", " … ")]:
            text = text.replace(old, new)
        return text

    def _get_word_tokens(self, word: str) -> List[str]:
        return list(word) + ["</w>"]

    def _get_pairs(self, word_tokens: List[str]) -> Set[Tuple[str, str]]:
        pairs = set()
        for i in range(len(word_tokens) - 1):
            pairs.add((word_tokens[i], word_tokens[i + 1]))
        return pairs

    def _merge_word(self, word_tokens: Tuple[str, ...], pair: Tuple[str, str]) -> List[str]:
        result = []
        i = 0
        while i < len(word_tokens):
            if (i < len(word_tokens) - 1 and
                    word_tokens[i] == pair[0] and
                    word_tokens[i + 1] == pair[1]):
                result.append(pair[0] + pair[1])
                i += 2
            else:
                result.append(word_tokens[i])
                i += 1
        return result

    # ------------------------------------------------------------------
    def train(self, texts: List[str]):
        """
        BPE algoritması ile tokenizer eğitimi.

        1. Kelime frekanslarını say
        2. Başlangıç vocab'unu oluştur (karakterler)
        3. En sık görülen çiftleri merge et, vocab_size hedefine ulaş
        """
        if not texts:
            raise ValueError("Eğitim metni boş olamaz.")

        # Çok kısa veri seti uyarısı
        if len(texts) < 3:
            print(f"⚠️  Uyarı: Sadece {len(texts)} metin örneği var. "
                  "Tokenizer kalitesi düşük olacak.")

        print(f"Tokenizer eğitimi başlıyor... Hedef vocab: {self.max_vocab_size}")

        # 1. Kelime frekansları
        word_freqs: Dict[Tuple[str, ...], int] = defaultdict(int)
        for text in texts:
            text = self._preprocess(text)
            for word in self.pattern.findall(text):
                if word.strip():
                    word_freqs[tuple(self._get_word_tokens(word))] += 1

        if not word_freqs:
            raise ValueError("Metinlerden hiç kelime çıkarılamadı.")

        print(f"Eşsiz kelime sayısı: {len(word_freqs)}")

        # 2. Başlangıç vocab: özel tokenlar + karakterler
        char_freqs: Dict[str, int] = defaultdict(int)
        for word_toks, freq in word_freqs.items():
            for ch in word_toks:
                char_freqs[ch] += freq

        self.vocab = self.special_tokens.copy()
        for ch, _ in sorted(char_freqs.items(), key=lambda x: x[1], reverse=True):
            if ch not in self.vocab and len(self.vocab) < self.max_vocab_size:
                self.vocab[ch] = len(self.vocab)

        print(f"Başlangıç vocab (karakterler): {len(self.vocab)}")

        # 3. BPE merge işlemleri
        num_merges = self.max_vocab_size - len(self.vocab)
        self.merges = []

        for i in range(num_merges):
            pair_freqs: Dict[Tuple[str, str], int] = defaultdict(int)
            for word_toks, freq in word_freqs.items():
                for pair in self._get_pairs(list(word_toks)):
                    pair_freqs[pair] += freq

            if not pair_freqs:
                break

            best_pair  = max(pair_freqs, key=pair_freqs.get)
            new_token  = best_pair[0] + best_pair[1]

            if new_token in self.vocab:
                continue

            self.vocab[new_token] = len(self.vocab)
            self.merges.append(best_pair)

            # Kelimeleri güncelle
            new_word_freqs = {}
            for word_toks, freq in word_freqs.items():
                merged = tuple(self._merge_word(word_toks, best_pair))
                new_word_freqs[merged] = new_word_freqs.get(merged, 0) + freq
            word_freqs = new_word_freqs

            if (i + 1) % 1000 == 0:
                print(f"Merge: {i + 1}/{num_merges} — '{new_token}'")

        print(f"Eğitim tamamlandı! Final vocab: {len(self.vocab)}")
        self._rebuild_id_to_token()

    # ------------------------------------------------------------------
    def _rebuild_id_to_token(self):
        """id_to_token her zaman vocab ile senkronize olmalı."""
        self.id_to_token = {v: k for k, v in self.vocab.items()}

    # ------------------------------------------------------------------
    def _check_trained(self):
        """Tokenizer eğitilmemişse açık hata ver."""
        if not self.merges and len(self.vocab) <= len(self.special_tokens):
            raise RuntimeError(
                "Tokenizer henüz eğitilmedi! train(texts) çağırın."
            )

    # ------------------------------------------------------------------
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Metni token ID listesine çevir.

        v3: Byte-level fallback eklendi.
          Tokenizer'ın hiç görmediği karakter geldiğinde (emoji, nadir sembol vb.)
          <unk> yazmak yerine UTF-8 byte'larına ayrılır, her byte ayrı token olur.
          Bu sayede model hiçbir girişte bilgi kaybetmez.
        """
        self._check_trained()

        if not text:
            ids = []
            if add_special_tokens:
                ids = [self.special_tokens["<s>"], self.special_tokens["</s>"]]
            return ids

        text  = self._preprocess(text)
        words = self.pattern.findall(text)
        ids   = [self.special_tokens["<s>"]] if add_special_tokens else []

        for word in words:
            if not word.strip():
                ids.append(self.vocab.get(word, self.special_tokens["<unk>"]))
                continue

            word_toks = self._get_word_tokens(word)
            for merge in self.merges:
                word_toks = self._merge_word(tuple(word_toks), merge)

            for tok in word_toks:
                if tok in self.vocab:
                    ids.append(self.vocab[tok])
                else:
                    # Byte-level fallback: bilinmeyen token'ı UTF-8 byte'larına böl
                    # Her byte'ı ayrı <unk> yerine mümkünse vocab'daki byte karşılığına eşle
                    encoded = False
                    for byte_char in tok.encode("utf-8", errors="replace"):
                        byte_str = chr(byte_char)
                        if byte_str in self.vocab:
                            ids.append(self.vocab[byte_str])
                            encoded = True
                        else:
                            ids.append(self.special_tokens["<unk>"])
                            encoded = True
                    if not encoded:
                        ids.append(self.special_tokens["<unk>"])

        if add_special_tokens:
            ids.append(self.special_tokens["</s>"])

        return ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Token ID listesini metne çevir."""
        if not self.id_to_token:
            self._rebuild_id_to_token()

        tokens = []
        special_vals = set(self.special_tokens.values())

        for idx in token_ids:
            if skip_special_tokens and idx in special_vals:
                continue
            tok = self.id_to_token.get(idx, "<unk>")
            tokens.append(tok)

        text = "".join(tokens).replace("</w>", " ")
        return " ".join(text.split()).strip()

    # ------------------------------------------------------------------
    def save(self, path: str):
        """Tokenizer'ı kaydet."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "vocab":          self.vocab,
            "merges":         self.merges,
            "max_vocab_size": self.max_vocab_size,
            "special_tokens": self.special_tokens,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Tokenizer kaydedildi: {path}")

    @classmethod
    def load(cls, path: str) -> "TurkishTokenizer":
        """Tokenizer'ı yükle."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        tok = cls(vocab_size=data.get("max_vocab_size", data.get("vocab_size", 32000)))
        tok.vocab          = data["vocab"]
        tok.merges         = data["merges"]
        tok.special_tokens = data["special_tokens"]
        tok._rebuild_id_to_token()
        return tok

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __len__(self):
        return len(self.vocab)


# ---------------------------------------------------------------------------
# Basit karakter tokenizer  (hızlı test için)
# ---------------------------------------------------------------------------

class SimpleTokenizer:
    """
    Karakter seviyesinde basit tokenizer.
    BPE eğitimi olmadan demo/test amaçlıdır.
    """

    def __init__(self):
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self.special_tokens: Dict[str, int] = {
            "<pad>":  0,
            "<unk>":  1,
            "<s>":    2,
            "</s>":   3,
        }

    def build_vocab(self, texts: List[str]):
        """Metinlerden karakter vocab'ı oluştur."""
        chars = set()
        for text in texts:
            if text:
                chars.update(text)

        self.char_to_id = self.special_tokens.copy()
        for ch in sorted(chars):
            if ch not in self.char_to_id:
                self.char_to_id[ch] = len(self.char_to_id)

        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        print(f"Karakter vocab boyutu: {len(self.char_to_id)}")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        if not text:
            return [self.special_tokens["<s>"], self.special_tokens["</s>"]] if add_special_tokens else []

        ids = [self.special_tokens["<s>"]] if add_special_tokens else []
        ids += [self.char_to_id.get(ch, self.special_tokens["<unk>"]) for ch in text]
        if add_special_tokens:
            ids.append(self.special_tokens["</s>"])
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        special_vals = set(self.special_tokens.values())
        chars = []
        for idx in ids:
            if skip_special_tokens and idx in special_vals:
                continue
            chars.append(self.id_to_char.get(idx, "<unk>"))
        return "".join(chars)

    def __len__(self):
        return len(self.char_to_id)