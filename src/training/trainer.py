"""
Model Eğitim Sistemi — v3
Tüm geliştirmeler eklendi:

  v2:
  • Mixed Precision (AMP), Cosine+Warmup, Gradient Clipping
  • Checkpoint yönetimi, eğitim geçmişi JSON

  v3 (yeni):
  • Validation split  → modelin ezberleyip ezberlemiyor kontrol
  • Eğitim sırasında örnek çıktı → her epoch'ta "nasıl gidiyor" görmek için
  • Gradient Checkpointing → büyük modellerde bellek tasarrufu
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import numpy as np
from typing import List, Dict, Optional, Tuple
import json
import os
from tqdm import tqdm
import logging
from datetime import datetime
import math
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Veri Setleri
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512, stride: int = 256):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.samples    = []

        pad_id = tokenizer.special_tokens.get("<pad>", 0)

        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=True)

            if len(tokens) <= max_length:
                padded = tokens + [pad_id] * (max_length - len(tokens) + 1)
                self.samples.append(padded[:max_length + 1])
            else:
                for i in range(0, len(tokens) - max_length, stride):
                    self.samples.append(tokens[i: i + max_length + 1])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:],  dtype=torch.long)
        return x, y


class ConversationDataset(Dataset):
    def __init__(self, conversations: List[Dict], tokenizer, max_length: int = 512):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.samples    = []

        bos = tokenizer.special_tokens.get("<s>", 2)
        eos = tokenizer.special_tokens.get("</s>", 3)
        pad = tokenizer.special_tokens.get("<pad>", 0)

        for conv in conversations:
            if "instruction" in conv:
                prompt   = f"Talimat: {conv['instruction']}\nGirdi: {conv.get('input', '')}\nYanıt: "
                response = conv.get("output", "")
            else:
                prompt   = conv.get("prompt", "")
                response = conv.get("response", "")

            prompt_tokens   = tokenizer.encode(prompt,   add_special_tokens=False)
            response_tokens = tokenizer.encode(response, add_special_tokens=False)

            full = [bos] + prompt_tokens + response_tokens + [eos]

            if len(full) > max_length:
                full = full[:max_length]
            else:
                full = full + [pad] * (max_length - len(full))

            labels    = [-100] * len(full)
            resp_start = 1 + len(prompt_tokens)
            resp_end   = resp_start + len(response_tokens)
            for i in range(resp_start, min(resp_end, max_length)):
                labels[i] = full[i]

            self.samples.append((full, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, labels = self.samples[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(labels[1:],  dtype=torch.long)
        return x, y


# ---------------------------------------------------------------------------
# Öğrenme Oranı Zamanlayıcısı
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Ana Eğitim Sınıfı
# ---------------------------------------------------------------------------

class TurkishAITrainer:
    def __init__(
        self,
        model,
        tokenizer,
        device: str             = None,
        learning_rate: float    = 5e-4,
        weight_decay: float     = 0.01,
        warmup_steps: int       = 100,
        max_grad_norm: float    = 1.0,
        checkpoint_dir: str     = "./checkpoints",
        use_amp: bool           = True,
        save_top_k: int         = 3,
        gradient_checkpointing: bool = False
    ):
        self.device        = device or (
            "mps"  if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model         = model.to(self.device)
        self.tokenizer     = tokenizer
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir
        self.warmup_steps  = warmup_steps
        self.save_top_k    = save_top_k

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Gradient Checkpointing — büyük modellerde bellek tasarrufu
        if gradient_checkpointing:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                logger.info("Gradient Checkpointing aktif")
            else:
                logger.info("Bu model gradient checkpointing desteklemiyor, atlandı")

        self.optimizer = AdamW(
            model.parameters(),
            lr           = learning_rate,
            weight_decay = weight_decay,
            betas        = (0.9, 0.95),
            eps          = 1e-8
        )

        # MPS için AMP şu an desteklenmiyor, sadece CUDA'da aktif
        self.use_amp = use_amp and self.device == "cuda"
        self.scaler  = torch.cuda.amp.GradScaler() if self.use_amp else None

        if self.use_amp:
            logger.info("Mixed Precision (AMP) aktif")

        self.global_step  = 0
        self.epoch        = 0
        self.loss_history: List[Dict] = []
        self.scheduler    = None
        self._saved_checkpoints: List[Tuple[float, str]] = []

        logger.info(f"Trainer hazır. Cihaz: {self.device}")
        logger.info(f"Model parametresi: {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------
    def create_dataloader(self, dataset: Dataset, batch_size: int = 4, shuffle: bool = True) -> DataLoader:
        total_steps   = (len(dataset) // batch_size) * 10
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            warmup_steps = min(self.warmup_steps, total_steps // 10),
            total_steps  = total_steps
        )
        return DataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = shuffle,
            num_workers = 0,
            pin_memory  = False
        )

    # ------------------------------------------------------------------
    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int               = 10,
        save_every: int               = 1,
        val_dataloader: DataLoader    = None,
        sample_prompt: str            = "Yapay zeka nedir?"
    ):
        """
        val_dataloader : validation seti (opsiyonel, overfitting tespiti için)
        sample_prompt  : her epoch sonunda bu prompt ile örnek üretilir
        """
        logger.info(f"Eğitim başlıyor: {num_epochs} epoch, toplam {num_epochs * len(train_dataloader)} adım")

        best_loss = float("inf")

        for epoch in range(num_epochs):
            self.model.train()
            self.epoch = epoch + 1

            total_loss  = 0.0
            num_batches = 0

            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

            for x, y in pbar:
                x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        logits, _ = self.model(x)
                        loss = nn.functional.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            y.view(-1),
                            ignore_index=-100
                        )
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    logits, _ = self.model(x)
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1),
                        ignore_index=-100
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                total_loss  += loss.item()
                num_batches += 1
                self.global_step += 1

                avg = total_loss / num_batches
                lr  = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg:.4f}", lr=f"{lr:.2e}")

            avg_loss = total_loss / max(1, num_batches)
            logger.info(f"Epoch {epoch + 1}/{num_epochs} tamamlandı. Train Loss: {avg_loss:.4f}")

            # --- Validation loss ---
            if val_dataloader is not None:
                val_loss = self.evaluate(val_dataloader)
                logger.info(f"  Validation Loss: {val_loss:.4f}")
                self.loss_history.append({
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "val_loss": val_loss
                })

                # Overfitting uyarısı
                if val_loss > avg_loss * 1.3:
                    logger.warning(f"  ⚠️  Overfitting belirtisi! Train: {avg_loss:.4f}, Val: {val_loss:.4f}")
            else:
                self.loss_history.append({"epoch": epoch + 1, "train_loss": avg_loss})

            # --- Örnek çıktı ---
            if sample_prompt:
                sample = self.generate_sample(sample_prompt, max_length=80)
                logger.info(f"  Örnek çıktı → {sample_prompt}")
                logger.info(f"  Model: {sample[:120]}")

            # --- Checkpoint kaydet ---
            if (epoch + 1) % save_every == 0:
                ckpt_name = f"checkpoint_epoch_{epoch + 1}.pt"
                self.save_checkpoint(ckpt_name)
                self._manage_top_k(avg_loss, ckpt_name)

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint("best_model.pt")
                logger.info(f"  → Yeni en iyi model kaydedildi! Loss: {best_loss:.4f}")

            self._save_history()

        logger.info("Eğitim tamamlandı!")

    # ------------------------------------------------------------------
    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss  = 0.0
        num_batches = 0

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        logits, _ = self.model(x)
                else:
                    logits, _ = self.model(x)

                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=-100
                )
                total_loss  += loss.item()
                num_batches += 1

        self.model.train()
        return total_loss / max(1, num_batches)

    # ------------------------------------------------------------------
    def save_checkpoint(self, filename: str):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            "epoch":                self.epoch,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict":    self.scaler.state_dict() if self.scaler else None,
            "global_step":          self.global_step,
            "loss_history":         self.loss_history,
        }, path)
        logger.info(f"Checkpoint kaydedildi: {path}")

    def load_checkpoint(self, filename: str):
        path = os.path.join(self.checkpoint_dir, filename)
        if not os.path.exists(path):
            logger.warning(f"Checkpoint bulunamadı: {path}")
            return

        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.epoch        = ckpt.get("epoch", 0)
        self.global_step  = ckpt.get("global_step", 0)
        self.loss_history = ckpt.get("loss_history", [])

        if self.scheduler and ckpt.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if self.scaler and ckpt.get("scaler_state_dict"):
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        logger.info(f"Checkpoint yüklendi: {path}")

    # ------------------------------------------------------------------
    def _manage_top_k(self, loss: float, filename: str):
        path = os.path.join(self.checkpoint_dir, filename)
        self._saved_checkpoints.append((loss, path))
        self._saved_checkpoints.sort(key=lambda x: x[0])

        while len(self._saved_checkpoints) > self.save_top_k:
            _, worst_path = self._saved_checkpoints.pop()
            if os.path.exists(worst_path) and "best_model" not in worst_path:
                os.remove(worst_path)
                logger.info(f"Eski checkpoint silindi: {worst_path}")

    def _save_history(self):
        path = os.path.join(self.checkpoint_dir, "training_history.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.loss_history, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    def generate_sample(self, prompt: str, max_length: int = 100) -> str:
        """Her epoch sonunda örnek üretir — modelin öğrenip öğrenmediğini gösterir."""
        self.model.eval()
        try:
            input_ids = torch.tensor(
                [self.tokenizer.encode(prompt, add_special_tokens=True)],
                device=self.device
            )
            with torch.no_grad():
                output = self.model.generate(input_ids, max_new_tokens=max_length, temperature=0.8)
            result = self.tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
        except Exception as e:
            result = f"(örnek üretilemedi: {e})"
        self.model.train()
        return result


# ---------------------------------------------------------------------------
# Online Learning
# ---------------------------------------------------------------------------

class OnlineLearning:
    def __init__(self, model, tokenizer, memory_manager, lr: float = 1e-5):
        self.model     = model
        self.tokenizer = tokenizer
        self.memory    = memory_manager
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.buffer: List[Dict] = []
        self.buffer_size = 10

    def learn_from_interaction(self, user_msg: str, assistant_msg: str, feedback: float = 1.0):
        self.buffer.append({
            "prompt":   user_msg,
            "response": assistant_msg,
            "feedback": feedback
        })
        if len(self.buffer) >= self.buffer_size:
            self._update_model()
            self.buffer = []

    def _update_model(self):
        self.model.train()
        device     = self.model.device
        total_loss = 0.0

        for sample in self.buffer:
            prompt_tok   = self.tokenizer.encode(sample["prompt"],   add_special_tokens=False)
            response_tok = self.tokenizer.encode(sample["response"], add_special_tokens=False)

            if not prompt_tok or not response_tok:
                continue

            full = prompt_tok + response_tok
            if len(full) < 2:
                continue

            x = torch.tensor([full[:-1]], device=device)
            y = torch.tensor([full[1:]],  device=device)

            logits, _ = self.model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

            if sample["feedback"] < 0:
                loss = -loss * 0.1

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += abs(loss.item())

        self.model.eval()
        logger.info(
            f"Online learning güncellendi. "
            f"Örnekler: {len(self.buffer)}, "
            f"Ort. loss: {total_loss / max(1, len(self.buffer)):.4f}"
        )


# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

def load_texts_from_directory(directory: str) -> List[str]:
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                texts.append(f.read())
    return texts


def create_datasets_from_conversations(
    conversations: List[Dict],
    tokenizer,
    train_ratio: float = 0.9
) -> Tuple[ConversationDataset, ConversationDataset]:
    """Konuşma verisini train/val olarak böl."""
    random.shuffle(conversations)
    split = int(len(conversations) * train_ratio)
    train = ConversationDataset(conversations[:split], tokenizer)
    val   = ConversationDataset(conversations[split:], tokenizer)
    logger.info(f"Train: {len(train)} örnek, Validation: {len(val)} örnek")
    return train, val


def create_text_datasets(
    texts: List[str],
    tokenizer,
    train_ratio: float = 0.9,
    max_length: int    = 512
) -> Tuple[TextDataset, TextDataset]:
    """Ham metni train/val olarak böl."""
    random.shuffle(texts)
    split      = int(len(texts) * train_ratio)
    train_data = TextDataset(texts[:split], tokenizer, max_length)
    val_data   = TextDataset(texts[split:], tokenizer, max_length)
    logger.info(f"Train: {len(train_data)} örnek, Validation: {len(val_data)} örnek")
    return train_data, val_data