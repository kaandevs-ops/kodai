"""
Gelişmiş Loglama Sistemi — v2
Orijinal API korundu, iyileştirmeler:

  • Rotasyonlu dosya logu: max 5 MB, 3 yedek
  • Milisaniye hassasiyetli timestamp
  • get_logger() kısayolu eklendi
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler


class ColoredFormatter(logging.Formatter):
    """Terminal için renkli log formatı."""

    COLORS = {
        "DEBUG":    "\033[36m",   # Cyan
        "INFO":     "\033[32m",   # Green
        "WARNING":  "\033[33m",   # Yellow
        "ERROR":    "\033[31m",   # Red
        "CRITICAL": "\033[35m",   # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str              = "turkish_ai",
    level: str             = "INFO",
    log_file: Optional[str] = None,
    console: bool          = True,
    max_bytes: int         = 5 * 1024 * 1024,   # 5 MB
    backup_count: int      = 3
) -> logging.Logger:
    """
    Logger oluştur ve yapılandır.

    Args:
        name         : Logger adı
        level        : Log seviyesi (DEBUG, INFO, WARNING, ERROR)
        log_file     : Dosyaya kaydet (opsiyonel)
        console      : Konsola yazdır
        max_bytes    : Dosya boyutu sınırı (rotasyon için)
        backup_count : Yedek dosya sayısı
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers = []   # Mevcut handler'ları temizle

    fmt       = "%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s"
    datefmt   = "%Y-%m-%d %H:%M:%S"
    plain_fmt = logging.Formatter(fmt, datefmt=datefmt)
    color_fmt = ColoredFormatter(fmt, datefmt=datefmt)

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(color_fmt)
        logger.addHandler(ch)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            log_file,
            encoding     = "utf-8",
            maxBytes     = max_bytes,
            backupCount  = backup_count
        )
        fh.setFormatter(plain_fmt)
        logger.addHandler(fh)

    return logger


def get_logger(name: str = "turkish_ai") -> logging.Logger:
    """Mevcut logger'ı getir (kısayol)."""
    return logging.getLogger(name)