#!/usr/bin/env python3
"""
KodAI - Terminal AI Asistanı v6
Bug fix + Yeni komutlar + Komuta özel AI + Gelişmiş arayüz
"""
import os, sys
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HUGGINGFACE_HUB_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import subprocess, time, json, ast, difflib
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.prompt import Prompt
from rich.text import Text
from rich.rule import Rule
from rich.columns import Columns
from rich import box
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML

from core.ai_engine import TurkishAI

console = Console()

BANNER = """[bold cyan]
██╗  ██╗ ██████╗ ██████╗      █████╗ ██╗
██║ ██╔╝██╔═══██╗██╔══██╗    ██╔══██╗██║
█████╔╝ ██║   ██║██║  ██║    ███████║██║
██╔═██╗ ██║   ██║██║  ██║    ██╔══██║██║
██║  ██╗╚██████╔╝██████╔╝    ██║  ██║██║
╚═╝  ╚═╝ ╚═════╝ ╚═════╝     ╚═╝  ╚═╝╚═╝
[/bold cyan][dim]Offline AI Kod Asistanı v6 • Mac M4 • Streaming[/dim]
"""

MODELLER = {
    "1": {"isim": "Qwen2.5-Coder-3B", "id": "Qwen/Qwen2.5-Coder-3B-Instruct", "aciklama": "Hızlı • 6GB • Günlük kullanım", "renk": "green"},
    "2": {"isim": "Qwen2.5-Coder-7B", "id": "Qwen/Qwen2.5-Coder-7B-Instruct", "aciklama": "Güçlü • 14GB • Karmaşık projeler", "renk": "yellow"},
}

# Tam stdlib listesi - /bagim bug fix
STDLIB = {
    "os", "sys", "re", "json", "time", "datetime", "pathlib", "subprocess",
    "threading", "collections", "itertools", "functools", "math", "random",
    "string", "io", "copy", "types", "typing", "abc", "warnings", "logging",
    "hashlib", "base64", "urllib", "http", "email", "html", "xml", "csv",
    "sqlite3", "ast", "inspect", "importlib", "pkgutil", "contextlib",
    "difflib", "dataclasses", "pickle", "struct", "socket", "ssl", "uuid",
    "enum", "queue", "heapq", "bisect", "array", "weakref", "gc", "platform",
    "shutil", "tempfile", "glob", "fnmatch", "stat", "errno", "signal",
    "traceback", "linecache", "pprint", "textwrap", "unicodedata", "codecs",
    "concurrent", "multiprocessing", "asyncio", "select", "selectors",
    "configparser", "argparse", "getopt", "getpass", "readline",
    "tokenize", "token", "keyword", "dis", "zipfile", "tarfile",
    "gzip", "bz2", "lzma", "zlib", "secrets", "hmac", "decimal",
    "fractions", "statistics", "cmath", "numbers", "operator", "builtins",
    # Proje-spesifik iç modüller ve eski kalıntılar
    "core", "data", "src", "autonomous_learner", "scenario_engine",
    "faiss", "bs4",
}

KOMUT_GRUPLARI = {
    "💬 Sohbet": {
        "/help":      "Yardım menüsü",
        "/reset":     "Konuşmayı sıfırla",
        "/gecmis":    "Geçmiş göster       → /gecmis [n]",
        "/ara":       "Geçmişte ara        → /ara <kelime>",
        "/kaydet":    "Konuşmayı aktar     → /kaydet [dosya.md]",
        "/kopyala":   "Son cevabı kopyala",
    },
    "📁 Dosya": {
        "/dosya":     "Dosya analizi       → /dosya <yol>",
        "/dosyalar":  "Çoklu analiz        → /dosyalar <yol1> <yol2>",
        "/ozet":      "Kısa özet           → /ozet <yol>",
        "/acikla":    "Satır satır açıkla  → /acikla <yol>",
        "/yaz":       "Dosyaya kod yaz     → /yaz <yol>",
        "/duzenle":   "AI ile düzenle      → /duzenle <yol>",
        "/diff":      "Dosyaları karşılaş  → /diff <yol1> <yol2>",
    },
    "🛠 Kod": {
        "/test":      "Test yaz            → /test <yol>",
        "/refactor":  "Refactor            → /refactor <yol>",
        "/optimize":  "Performans analizi  → /optimize <yol>",
        "/karmasiklik": "Karmaşıklık skoru → /karmasiklik <yol>",
        "/rename":    "Yeniden adlandır    → /rename <yol>",
        "/hata":      "Hata analizi        → /hata [metin]",
        "/calistir":  "Kodu çalıştır       → /calistir <yol>",
        "/dok":       "Docstring ekle      → /dok <yol>",
        "/snippet":   "Snippet yönetimi    → /snippet <isim|list|sil|yukle>",
        "/todo":      "TODO'ları listele   → /todo [klasör]",
        "/bagim":     "Bağımlılık analizi  → /bagim [klasör]",
    },
    "🔧 Proje": {
        "/proje":     "Proje analizi       → /proje <klasör>",
        "/git":       "Git yardımı         → /git <komut>",
        "/bul":       "Dosyalarda ara      → /bul <pattern> [klasör]",
    },
    "⚙️  Sistem": {
        "/stats":     "İstatistikler",
        "/sicaklik":  "Sıcaklık ayarla    → /sicaklik 0.1-2.0",
        "/uzunluk":   "Max token          → /uzunluk 100-2000",
        "/model":     "Model değiştir",
        "/stream":    "Streaming aç/kapat",
        "/temizle":   "Ekranı temizle",
        "/exit":      "Çıkış",
    },
}

AYARLAR = {
    "temperature": 0.7,
    "max_len": 600,
    "streaming": True,
    "model_isim": "Qwen2.5-Coder-3B",
}

SNIPPET_DOSYA = ".kodai_snippets.json"
prompt_style = Style.from_dict({"prompt": "bold cyan"})

# ─── YARDIMCI ───────────────────────────────────────────────────────────────

def durum_satiri():
    return (f"[dim]🤖 {AYARLAR['model_isim']} │ 🌡 {AYARLAR['temperature']} │ "
            f"📏 {AYARLAR['max_len']} token │ "
            f"{'⚡ Stream' if AYARLAR['streaming'] else '⏳ Normal'}[/dim]")

def print_banner():
    console.print(BANNER)
    console.print(Panel(
        "[bold white]Kod yazma • Hata ayıklama • Dosya analizi • Git • Test • Türkçe[/bold white]",
        border_style="cyan", box=box.ROUNDED
    ))

def model_sec():
    tablo = Table(box=box.ROUNDED, border_style="cyan", show_header=True)
    tablo.add_column("#", style="bold cyan", width=3)
    tablo.add_column("Model", style="bold white")
    tablo.add_column("Açıklama", style="dim")
    for no, bilgi in MODELLER.items():
        tablo.add_row(no, f"[{bilgi['renk']}]{bilgi['isim']}[/{bilgi['renk']}]", bilgi["aciklama"])
    console.print(Panel(tablo, title="[bold cyan]🤖 Model Seç[/bold cyan]", border_style="cyan"))
    while True:
        secim = Prompt.ask("[bold cyan]Seçim[/bold cyan]", default="1")
        if secim in MODELLER:
            AYARLAR["model_isim"] = MODELLER[secim]["isim"]
            return MODELLER[secim]["id"], MODELLER[secim]["isim"]
        console.print("[red]❌ 1 veya 2 gir[/red]")

def print_help():
    for grup, komutlar in KOMUT_GRUPLARI.items():
        tablo = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        tablo.add_column("Komut", style="bold cyan", width=14)
        tablo.add_column("Açıklama", style="dim")
        for cmd, desc in komutlar.items():
            tablo.add_row(cmd, desc)
        console.print(Panel(tablo, title=f"[bold]{grup}[/bold]", border_style="cyan", box=box.ROUNDED))
    console.print(durum_satiri())
    console.print()

def kod_cikart(response: str) -> str:
    if "```" in response:
        parts = response.split("```")
        if len(parts) >= 3:
            kod = parts[1]
            lines = kod.split("\n", 1)
            return (lines[1] if len(lines) > 1 else kod).strip()
    return response.strip()

def format_response(response: str):
    parts = response.split("```")
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                console.print(Markdown(part))
        else:
            lines = part.split("\n", 1)
            lang = lines[0].strip() if lines[0].strip() else "python"
            code = lines[1] if len(lines) > 1 else part
            syntax = Syntax(code.strip(), lang, theme="monokai", line_numbers=True, word_wrap=True)
            console.print(Panel(syntax, title=f"[bold green]{lang}[/bold green]",
                                border_style="green", box=box.ROUNDED))

def ai_cevap(ai, mesaj: str, max_len: int = None, komut_tipi: str = "genel") -> str:
    if max_len is None:
        max_len = AYARLAR["max_len"]
    baslangic = time.time()

    if AYARLAR["streaming"]:
        console.print()
        console.print(Panel("", title=f"[bold magenta]🤖 KodAI[/bold magenta] [dim]⚡ streaming...[/dim]",
                            border_style="magenta", box=box.ROUNDED))
        buffer = ""
        def yazdir(chunk):
            nonlocal buffer
            buffer += chunk
            if "```" not in buffer or buffer.count("```") % 2 == 0:
                console.print(chunk, end="", highlight=False)
        try:
            response = ai.stream_chat(mesaj, max_length=max_len,
                                      temperature=AYARLAR["temperature"],
                                      komut_tipi=komut_tipi, callback=yazdir)
        except AttributeError:
            with Live(Spinner("dots2", text="[cyan]Düşünüyor...[/cyan]"), refresh_per_second=10) as live:
                response = ai.chat(mesaj, max_length=max_len,
                                   temperature=AYARLAR["temperature"], komut_tipi=komut_tipi)
                live.stop()
        sure = time.time() - baslangic
        console.print()
        console.print(f"[dim]⏱ {sure:.1f}s[/dim]")
        if "```" in response:
            console.print(Rule(style="dim"))
            format_response(response)
        console.print()
    else:
        with Live(Spinner("dots2", text="[cyan]Düşünüyor...[/cyan]"), refresh_per_second=10) as live:
            response = ai.chat(mesaj, max_length=max_len,
                               temperature=AYARLAR["temperature"], komut_tipi=komut_tipi)
            live.stop()
        sure = time.time() - baslangic
        console.print()
        console.print(Panel("", title=f"[bold magenta]🤖 KodAI[/bold magenta] [dim]({sure:.1f}s)[/dim]",
                            border_style="magenta", box=box.ROUNDED))
        format_response(response)
        console.print()
    return response

# ─── SNIPPET ────────────────────────────────────────────────────────────────

def snippet_yukle() -> dict:
    if os.path.exists(SNIPPET_DOSYA):
        try:
            with open(SNIPPET_DOSYA, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {}

def snippet_kaydet_dosya(snippets: dict):
    with open(SNIPPET_DOSYA, "w", encoding="utf-8") as f:
        json.dump(snippets, f, ensure_ascii=False, indent=2)

def cmd_snippet(arg: str, son_cevap: str):
    snippets = snippet_yukle()
    parcalar = arg.strip().split(maxsplit=1)
    alt = parcalar[0] if parcalar else "list"

    if alt in ("list", ""):
        if not snippets:
            console.print("[dim]Henüz snippet yok. /snippet <isim> ile son cevabı kaydet.[/dim]"); return
        tablo = Table(box=box.ROUNDED, border_style="yellow")
        tablo.add_column("İsim", style="bold cyan")
        tablo.add_column("Önizleme", style="dim", max_width=50)
        tablo.add_column("Tarih", style="dim")
        for isim, veri in snippets.items():
            tablo.add_row(isim, veri["kod"][:50].replace("\n", " ") + "…", veri.get("tarih", "-"))
        console.print(Panel(tablo, title="[bold yellow]📦 Snippets[/bold yellow]", border_style="yellow"))

    elif alt == "sil":
        isim = parcalar[1] if len(parcalar) > 1 else ""
        if isim in snippets:
            del snippets[isim]
            snippet_kaydet_dosya(snippets)
            console.print(f"[green]✅ '{isim}' silindi.[/green]")
        else:
            console.print(f"[red]❌ '{isim}' bulunamadı.[/red]")

    elif alt in ("yukle", "yükle"):
        isim = parcalar[1] if len(parcalar) > 1 else ""
        if isim in snippets:
            kod = snippets[isim]["kod"]
            console.print(Panel(Syntax(kod, "python", theme="monokai", line_numbers=True),
                                title=f"[bold green]📦 {isim}[/bold green]", border_style="green"))
            subprocess.run(["pbcopy"], input=kod.encode())
            console.print("[dim]Panoya kopyalandı.[/dim]")
        else:
            console.print(f"[red]❌ '{isim}' bulunamadı.[/red]")

    else:
        if not son_cevap:
            console.print("[yellow]⚠️ Kaydedilecek cevap yok.[/yellow]"); return
        kod = kod_cikart(son_cevap)
        snippets[alt] = {"kod": kod, "ext": "python", "tarih": datetime.now().strftime("%d.%m.%Y %H:%M")}
        snippet_kaydet_dosya(snippets)
        console.print(f"[green]✅ '{alt}' kaydedildi ({len(kod)} karakter).[/green]")

# ─── KOMUTLAR ───────────────────────────────────────────────────────────────

def cmd_dosya(ai, dosya_yolu: str):
    if not os.path.exists(dosya_yolu):
        console.print(f"[red]❌ Dosya bulunamadı: {dosya_yolu}[/red]"); return
    try:
        with open(dosya_yolu, "r", encoding="utf-8") as f:
            icerik = f.read()
        ext = os.path.splitext(dosya_yolu)[1].lstrip(".")
        console.print(Panel(Syntax(icerik[:3000], ext or "text", theme="monokai", line_numbers=True),
                            title=f"[bold blue]📄 {dosya_yolu}[/bold blue]", border_style="blue"))
        ai_cevap(ai, f"Şu dosyayı analiz et ve Türkçe açıkla:\n\nDosya: {dosya_yolu}\n\n```{ext}\n{icerik[:3000]}\n```", max_len=500)
    except Exception as e:
        console.print(f"[red]❌ Hata: {e}[/red]")

def cmd_dosyalar(ai, dosya_listesi: list):
    icerikler = []
    for dosya in dosya_listesi:
        if not os.path.exists(dosya):
            console.print(f"[yellow]⚠️ Bulunamadı: {dosya}[/yellow]"); continue
        try:
            with open(dosya, "r", encoding="utf-8") as f:
                icerik = f.read()
            ext = os.path.splitext(dosya)[1].lstrip(".")
            icerikler.append(f"### {dosya}\n```{ext}\n{icerik[:1500]}\n```")
        except Exception as e:
            console.print(f"[red]❌ {dosya}: {e}[/red]")
    if not icerikler:
        console.print("[red]❌ Hiç dosya okunamadı.[/red]"); return
    ai_cevap(ai, f"Bu {len(icerikler)} dosyayı karşılaştır, ilişkisini Türkçe açıkla:\n\n" + "\n\n".join(icerikler), max_len=700)

def cmd_diff(dosya1: str, dosya2: str):
    for d in [dosya1, dosya2]:
        if not os.path.exists(d):
            console.print(f"[red]❌ Bulunamadı: {d}[/red]"); return
    try:
        with open(dosya1) as f1, open(dosya2) as f2:
            fark = list(difflib.unified_diff(f1.readlines(), f2.readlines(),
                                             fromfile=dosya1, tofile=dosya2, lineterm=""))
        if not fark:
            console.print("[green]✅ Dosyalar aynı.[/green]"); return
        console.print(Panel(Syntax("\n".join(fark[:80]), "diff", theme="monokai"),
                            title=f"[bold yellow]🔀 {dosya1} ↔ {dosya2}[/bold yellow]", border_style="yellow"))
    except Exception as e:
        console.print(f"[red]❌ Hata: {e}[/red]")

def cmd_ozet(ai, dosya_yolu: str):
    if not os.path.exists(dosya_yolu):
        console.print(f"[red]❌ Bulunamadı: {dosya_yolu}[/red]"); return
    with open(dosya_yolu, "r", encoding="utf-8") as f: icerik = f.read()
    ext = os.path.splitext(dosya_yolu)[1].lstrip(".")
    ai_cevap(ai, f"Bu kodu 3 cümleyle özetle:\n\n```{ext}\n{icerik[:2000]}\n```", max_len=200)

def cmd_acikla(ai, dosya_yolu: str):
    if not os.path.exists(dosya_yolu):
        console.print(f"[red]❌ Bulunamadı: {dosya_yolu}[/red]"); return
    with open(dosya_yolu, "r", encoding="utf-8") as f: icerik = f.read()
    ext = os.path.splitext(dosya_yolu)[1].lstrip(".")
    ai_cevap(ai, f"Bu kodu satır satır açıkla, yeni başlayanlar anlayabilsin:\n\n```{ext}\n{icerik[:2000]}\n```",
             max_len=700, komut_tipi="acikla")

def cmd_optimize(ai, dosya_yolu: str):
    """Performans analizi ve optimizasyon önerileri."""
    if not os.path.exists(dosya_yolu):
        console.print(f"[red]❌ Bulunamadı: {dosya_yolu}[/red]"); return
    try:
        with open(dosya_yolu, "r", encoding="utf-8") as f:
            icerik = f.read()
        ext = os.path.splitext(dosya_yolu)[1].lstrip(".")
        console.print(f"[cyan]⚡ Performans analizi: {dosya_yolu}[/cyan]")
        mesaj = (f"Bu kodu performans açısından analiz et. Türkçe yanıtla:\n"
                 f"1. **Yavaş Noktalar:** Hangi kısımlar yavaş?\n"
                 f"2. **O(n) Analizi:** Zaman karmaşıklığı nedir?\n"
                 f"3. **Optimizasyon:** Nasıl hızlandırılır? Örnek ver.\n\n"
                 f"```{ext}\n{icerik[:3000]}\n```")
        ai_cevap(ai, mesaj, max_len=700, komut_tipi="genel")
    except Exception as e:
        console.print(f"[red]❌ Hata: {e}[/red]")

def cmd_karmasiklik(dosya_yolu: str):
    """Kod karmaşıklık skoru — AST ile hesapla, AI gerektirmez."""
    if not os.path.exists(dosya_yolu):
        console.print(f"[red]❌ Bulunamadı: {dosya_yolu}[/red]"); return
    try:
        with open(dosya_yolu, "r", encoding="utf-8") as f:
            icerik = f.read()
        tree = ast.parse(icerik)

        fonksiyonlar = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Cyclomatic complexity: 1 + dallanma sayısı
                karmasiklik = 1
                for alt in ast.walk(node):
                    if isinstance(alt, (ast.If, ast.While, ast.For, ast.ExceptHandler,
                                        ast.With, ast.Assert, ast.comprehension)):
                        karmasiklik += 1
                    elif isinstance(alt, ast.BoolOp):
                        karmasiklik += len(alt.values) - 1

                satir = getattr(node, 'end_lineno', 0) - node.lineno + 1
                renk = "green" if karmasiklik <= 5 else "yellow" if karmasiklik <= 10 else "red"
                risk = "Düşük ✅" if karmasiklik <= 5 else "Orta ⚠️" if karmasiklik <= 10 else "Yüksek ❌"
                fonksiyonlar.append((node.name, karmasiklik, satir, renk, risk))

        if not fonksiyonlar:
            console.print("[yellow]⚠️ Fonksiyon bulunamadı.[/yellow]"); return

        tablo = Table(box=box.ROUNDED, border_style="cyan")
        tablo.add_column("Fonksiyon", style="cyan")
        tablo.add_column("Karmaşıklık", justify="center")
        tablo.add_column("Satır", justify="right", style="dim")
        tablo.add_column("Risk", justify="center")

        toplam = 0
        for isim, k, s, renk, risk in sorted(fonksiyonlar, key=lambda x: -x[1]):
            tablo.add_row(isim, f"[{renk}]{k}[/{renk}]", str(s), risk)
            toplam += k

        ort = toplam / len(fonksiyonlar)
        console.print(Panel(tablo,
            title=f"[bold cyan]📊 Karmaşıklık Analizi — {dosya_yolu}[/bold cyan]",
            border_style="cyan"))
        console.print(f"[dim]Ortalama: {ort:.1f} | Toplam fonksiyon: {len(fonksiyonlar)} | "
                      f"{'✅ İyi durumda' if ort <= 5 else '⚠️ Refactor önerilir' if ort <= 10 else '❌ Acil refactor gerekli'}[/dim]")
    except SyntaxError as e:
        console.print(f"[red]❌ Syntax hatası: {e}[/red]")
    except Exception as e:
        console.print(f"[red]❌ Hata: {e}[/red]")

def cmd_rename(ai, dosya_yolu: str):
    """Değişken/fonksiyon yeniden adlandırma."""
    if not os.path.exists(dosya_yolu):
        console.print(f"[red]❌ Bulunamadı: {dosya_yolu}[/red]"); return
    try:
        with open(dosya_yolu, "r", encoding="utf-8") as f:
            icerik = f.read()
        console.print(f"[cyan]Ne adını değiştirelim?[/cyan]")
        eski = input("  Eski isim → ").strip()
        if not eski:
            return
        yeni = input("  Yeni isim → ").strip()
        if not yeni:
            return

        # Basit rename: tüm dosyada değiştir
        sayac = icerik.count(eski)
        if sayac == 0:
            console.print(f"[yellow]⚠️ '{eski}' bulunamadı.[/yellow]"); return

        yeni_icerik = icerik.replace(eski, yeni)
        yedek = dosya_yolu + ".bak"
        with open(yedek, "w", encoding="utf-8") as f:
            f.write(icerik)
        with open(dosya_yolu, "w", encoding="utf-8") as f:
            f.write(yeni_icerik)

        console.print(f"[green]✅ '{eski}' → '{yeni}' ({sayac} yerde değiştirildi)[/green] [dim](yedek: {yedek})[/dim]")

        # AI ile doğrula
        ai_cevap(ai, f"Bu kodda '{eski}' yerine '{yeni}' kullanıldı. Mantıklı mı, sorun var mı? Kısaca Türkçe değerlendir:\n\n```python\n{yeni_icerik[:1500]}\n```",
                 max_len=300, komut_tipi="genel")
    except Exception as e:
        console.print(f"[red]❌ Hata: {e}[/red]")

def cmd_duzenle(ai, dosya_yolu: str):
    if not os.path.exists(dosya_yolu):
        console.print(f"[red]❌ Bulunamadı: {dosya_yolu}[/red]"); return
    try:
        with open(dosya_yolu, "r", encoding="utf-8") as f: icerik = f.read()
        ext = os.path.splitext(dosya_yolu)[1].lstrip(".")
        console.print(Panel(Syntax(icerik[:2000], ext or "text", theme="monokai", line_numbers=True),
                            title=f"[bold blue]📄 {dosya_yolu}[/bold blue]", border_style="blue"))
        console.print("[cyan]Ne değiştirelim?[/cyan]")
        degisiklik = input("→ ").strip()
        if not degisiklik: return
        response = ai_cevap(ai, f"Bu kodu şu şekilde düzenle: {degisiklik}\n\nSadece düzenlenmiş kodu yaz:\n\n```{ext}\n{icerik[:3000]}\n```",
                            max_len=900, komut_tipi="yaz")
        yeni_kod = kod_cikart(response)
        yedek = dosya_yolu + ".bak"
        with open(yedek, "w", encoding="utf-8") as f: f.write(icerik)
        with open(dosya_yolu, "w", encoding="utf-8") as f: f.write(yeni_kod)
        console.print(f"[green]✅ Kaydedildi: {dosya_yolu}[/green] [dim](yedek: {yedek})[/dim]")
    except Exception as e:
        console.print(f"[red]❌ Hata: {e}[/red]")

def cmd_test(ai, dosya_yolu: str):
    if not os.path.exists(dosya_yolu):
        console.print(f"[red]❌ Bulunamadı: {dosya_yolu}[/red]"); return
    with open(dosya_yolu, "r", encoding="utf-8") as f: icerik = f.read()
    ext = os.path.splitext(dosya_yolu)[1].lstrip(".")
    response = ai_cevap(ai, f"Bu kod için kapsamlı pytest testleri yaz. Türkçe yorum ekle:\n\n```{ext}\n{icerik[:3000]}\n```",
                        max_len=800, komut_tipi="test")
    kod = kod_cikart(response)
    if kod != response:
        test_dosya = dosya_yolu.replace(".py", "_test.py")
        with open(test_dosya, "w", encoding="utf-8") as f: f.write(kod)
        console.print(f"[green]✅ Test kaydedildi: {test_dosya}[/green]")

def cmd_refactor(ai, dosya_yolu: str):
    if not os.path.exists(dosya_yolu):
        console.print(f"[red]❌ Bulunamadı: {dosya_yolu}[/red]"); return
    with open(dosya_yolu, "r", encoding="utf-8") as f: icerik = f.read()
    ext = os.path.splitext(dosya_yolu)[1].lstrip(".")
    ai_cevap(ai, f"Bu kodu refactor et, daha temiz hale getir:\n\n```{ext}\n{icerik[:3000]}\n```",
             max_len=700, komut_tipi="refactor")

def cmd_hata(ai, hata_metni: str):
    if not hata_metni:
        console.print("[yellow]Hata metnini gir (boş satırla bitir):[/yellow]")
        satirlar = []
        while True:
            satir = input()
            if not satir: break
            satirlar.append(satir)
        hata_metni = "\n".join(satirlar)
    ai_cevap(ai, f"Bu hatayı analiz et:\n\n```\n{hata_metni}\n```", max_len=450, komut_tipi="hata")

def cmd_dok(ai, dosya_yolu: str):
    if not os.path.exists(dosya_yolu):
        console.print(f"[red]❌ Bulunamadı: {dosya_yolu}[/red]"); return
    with open(dosya_yolu, "r", encoding="utf-8") as f: icerik = f.read()
    response = ai_cevap(ai, f"Bu Python kodundaki tüm fonksiyonlara Türkçe docstring ekle. Sadece kodu yaz:\n\n```python\n{icerik[:3000]}\n```",
                        max_len=900, komut_tipi="yaz")
    yeni_kod = kod_cikart(response)
    yedek = dosya_yolu + ".bak"
    with open(yedek, "w", encoding="utf-8") as f: f.write(icerik)
    with open(dosya_yolu, "w", encoding="utf-8") as f: f.write(yeni_kod)
    console.print(f"[green]✅ Docstring'ler eklendi: {dosya_yolu}[/green] [dim](yedek: {yedek})[/dim]")

def cmd_todo(klasor: str = "."):
    console.print(f"[cyan]📋 TODO'lar aranıyor: {klasor}[/cyan]")
    try:
        result = subprocess.run(
            ["grep", "-r", "-n", "--include=*.py",
             "--exclude-dir=venv", "--exclude-dir=.git",
             "--exclude-dir=__pycache__", "--exclude-dir=.venv",
             "-E", "TODO|FIXME|HACK|XXX|NOTE", klasor],
            capture_output=True, text=True
        )
        if result.stdout:
            satirlar = result.stdout.strip().split("\n")
            tablo = Table(box=box.ROUNDED, border_style="yellow")
            tablo.add_column("Dosya", style="cyan", max_width=35)
            tablo.add_column("Satır", style="dim", width=6)
            tablo.add_column("İçerik", style="white")
            for satir in satirlar[:40]:
                parcalar = satir.split(":", 2)
                if len(parcalar) >= 3:
                    icerik = parcalar[2].strip()
                    etiket = next((e for e in ["FIXME", "TODO", "HACK", "NOTE", "XXX"] if e in icerik), "TODO")
                    renk = {"FIXME": "red", "TODO": "yellow", "HACK": "magenta"}.get(etiket, "dim")
                    tablo.add_row(parcalar[0], parcalar[1], f"[{renk}]{icerik}[/{renk}]")
            console.print(Panel(tablo,
                title=f"[bold yellow]📋 TODO'lar ({len(satirlar)} adet)[/bold yellow]", border_style="yellow"))
        else:
            console.print("[green]✅ Hiç TODO bulunamadı.[/green]")
    except Exception as e:
        console.print(f"[red]❌ Hata: {e}[/red]")

def cmd_bagim(klasor: str = "."):
    """Bağımlılık analizi — stdlib bug fix ile."""
    imports = set()
    for root, dirs, files in os.walk(klasor):
        dirs[:] = [d for d in dirs if d not in ["venv", ".git", "__pycache__", "node_modules", ".venv"]]
        for f in files:
            if not f.endswith(".py"):
                continue
            try:
                with open(os.path.join(root, f), "r", encoding="utf-8", errors="ignore") as fp:
                    for line in fp:
                        line = line.strip()
                        if line.startswith("import "):
                            # "import os, sys, re" → ["os", "sys", "re"]
                            rest = line[7:].split("#")[0]  # yorum kaldır
                            for pkg in rest.split(","):
                                pkg = pkg.strip().split(".")[0].split(" as ")[0].strip()
                                if pkg and not pkg.startswith("_"):
                                    imports.add(pkg)
                        elif line.startswith("from ") and " import " in line:
                            pkg = line.split()[1].split(".")[0]
                            if pkg and not pkg.startswith("_") and pkg != "__future__":
                                imports.add(pkg)
            except:
                pass

    # stdlib ve iç modülleri filtrele
    dis_paketler = imports - STDLIB

    req_dosya = os.path.join(klasor, "requirements.txt")
    req_paketler = set()
    if os.path.exists(req_dosya):
        with open(req_dosya) as f:
            for line in f:
                pkg = line.strip().split("==")[0].split(">=")[0].split("~=")[0].lower()
                if pkg and not pkg.startswith("#"):
                    req_paketler.add(pkg)

    tablo = Table(box=box.ROUNDED, border_style="blue")
    tablo.add_column("Paket", style="cyan")
    tablo.add_column("Durum", style="white")

    eksik_yuklü = []
    for pkg in sorted(dis_paketler):
        if pkg.lower() in req_paketler:
            tablo.add_row(pkg, "[green]✅ requirements.txt'te var[/green]")
        else:
            tablo.add_row(pkg, "[yellow]⚠️ requirements.txt'te yok[/yellow]")
            # Yüklü mü kontrol et
            try:
                r = subprocess.run(["pip", "show", pkg], capture_output=True, text=True)
                if r.returncode == 0:
                    versiyon = next((s.split(": ")[1] for s in r.stdout.splitlines() if s.startswith("Version:")), "?")
                    eksik_yuklü.append(f"{pkg}=={versiyon}")
            except:
                pass

    console.print(Panel(tablo,
        title=f"[bold blue]📦 Bağımlılıklar ({len(dis_paketler)} dış paket)[/bold blue]",
        border_style="blue"))

    if eksik_yuklü:
        console.print(f"[yellow]⚠️ Yüklü ama requirements.txt'te yok:[/yellow]")
        for p in eksik_yuklü:
            console.print(f"  [cyan]{p}[/cyan]")
        cevap = input("\nrequirements.txt'e eklensin mi? (e/h): ").strip().lower()
        if cevap == "e":
            with open(req_dosya, "a", encoding="utf-8") as f:
                for p in eksik_yuklü:
                    f.write(f"\n{p}")
            console.print(f"[green]✅ {len(eksik_yuklü)} paket eklendi![/green]")

def cmd_git(ai, git_komut: str):
    if not git_komut:
        console.print("[yellow]Git ile ne yapmak istiyorsun?[/yellow]")
        git_komut = input("→ ").strip()
    gercek = ["status", "log", "diff", "branch", "stash", "show"]
    if git_komut.split()[0] in gercek:
        try:
            result = subprocess.run(["git"] + git_komut.split(), capture_output=True, text=True)
            cikti = result.stdout or result.stderr
            console.print(Panel(cikti[:3000], title=f"[bold yellow]git {git_komut}[/bold yellow]", border_style="yellow"))
            if git_komut.startswith(("log", "diff")):
                ai_cevap(ai, f"Bu git çıktısını analiz et:\n\n{cikti[:2000]}", max_len=300, komut_tipi="git")
        except Exception as e:
            console.print(f"[red]❌ Git hatası: {e}[/red]")
    else:
        ai_cevap(ai, f"Git komutları hakkında yardım et: {git_komut}", max_len=400, komut_tipi="git")

def cmd_calistir(ai, dosya_yolu: str):
    if not os.path.exists(dosya_yolu):
        console.print(f"[red]❌ Bulunamadı: {dosya_yolu}[/red]"); return
    console.print(f"[yellow]▶ Çalıştırılıyor: {dosya_yolu}[/yellow]\n")
    try:
        result = subprocess.run([sys.executable, dosya_yolu], capture_output=True, text=True, timeout=30)
        if result.stdout:
            console.print(Panel(result.stdout, title="[bold green]✅ Çıktı[/bold green]", border_style="green"))
        if result.stderr:
            console.print(Panel(result.stderr, title="[bold red]❌ Hata[/bold red]", border_style="red"))
            ai_cevap(ai, f"Bu Python hatasını analiz et:\n\n```\n{result.stderr}\n```", max_len=400, komut_tipi="hata")
    except subprocess.TimeoutExpired:
        console.print("[red]❌ Zaman aşımı (30s)[/red]")
    except Exception as e:
        console.print(f"[red]❌ Hata: {e}[/red]")

def cmd_proje(ai, klasor: str):
    if not os.path.exists(klasor):
        console.print(f"[red]❌ Klasör bulunamadı: {klasor}[/red]"); return
    dosyalar = []
    toplam = 0
    for root, dirs, files in os.walk(klasor):
        dirs[:] = [d for d in dirs if d not in ["venv", ".git", "__pycache__", "node_modules", ".venv"]]
        for f in files:
            if f.endswith((".py", ".js", ".ts", ".html", ".css", ".json", ".md")):
                tam = os.path.join(root, f)
                dosyalar.append(tam)
                toplam += os.path.getsize(tam)
    tablo = Table(box=box.ROUNDED, border_style="blue")
    tablo.add_column("Dosya", style="cyan")
    tablo.add_column("Boyut", style="dim", justify="right")
    ozet = []
    for dosya in dosyalar[:20]:
        tablo.add_row(dosya, f"{os.path.getsize(dosya):,}B")
        try:
            with open(dosya, "r", encoding="utf-8", errors="ignore") as f:
                ozet.append(f"### {dosya}\n```\n{f.read()[:400]}\n```")
        except: pass
    console.print(Panel(tablo,
        title=f"[bold blue]📁 {klasor} — {len(dosyalar)} dosya • {toplam/1024:.1f}KB[/bold blue]",
        border_style="blue"))
    ai_cevap(ai, f"Bu projeyi kısaca analiz et, amacını Türkçe özetle:\n\n{''.join(ozet[:3])}", max_len=300)

def cmd_bul(pattern: str, klasor: str = "."):
    console.print(f"[cyan]🔍 '{pattern}' aranıyor: {klasor}[/cyan]")
    try:
        result = subprocess.run(
            ["grep", "-r", "-n", "--include=*.py", "--color=never",
             "--exclude-dir=venv", "--exclude-dir=.git", "--exclude-dir=__pycache__",
             pattern, klasor],
            capture_output=True, text=True
        )
        if result.stdout:
            satirlar = result.stdout.strip().split("\n")[:30]
            tablo = Table(box=box.SIMPLE, show_header=False)
            tablo.add_column("Dosya", style="cyan", max_width=40)
            tablo.add_column("Satır", style="dim", width=5)
            tablo.add_column("İçerik", style="white")
            for satir in satirlar:
                p = satir.split(":", 2)
                if len(p) >= 3:
                    tablo.add_row(p[0], p[1], p[2].strip())
            console.print(Panel(tablo,
                title=f"[bold yellow]🔍 '{pattern}' — {len(satirlar)} sonuç[/bold yellow]", border_style="yellow"))
        else:
            console.print(f"[dim]'{pattern}' bulunamadı.[/dim]")
    except Exception as e:
        console.print(f"[red]❌ Hata: {e}[/red]")

def cmd_yaz(ai, dosya_yolu: str):
    console.print(f"[cyan]Ne yazayım? [dim]({dosya_yolu})[/dim][/cyan]")
    aciklama = input("→ ").strip()
    if not aciklama: return
    response = ai_cevap(ai, f"Sadece kodu yaz, gereksiz açıklama ekleme:\n{aciklama}\n\nDosya: {dosya_yolu}",
                        max_len=900, komut_tipi="yaz")
    kod = kod_cikart(response)
    with open(dosya_yolu, "w", encoding="utf-8") as f: f.write(kod)
    ext = os.path.splitext(dosya_yolu)[1].lstrip(".") or "python"
    console.print(Panel(Syntax(kod, ext, theme="monokai", line_numbers=True),
                        title=f"[bold green]✅ Yazıldı: {dosya_yolu}[/bold green]", border_style="green"))

def cmd_kaydet_konusma(gecmis: list, dosya_adi: str):
    if not dosya_adi:
        dosya_adi = f"kodai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(dosya_adi, "w", encoding="utf-8") as f:
        f.write(f"# KodAI Konuşma — {datetime.now().strftime('%d.%m.%Y %H:%M')}\n\n")
        for soru, cevap in gecmis:
            f.write(f"## 👤 Siz\n{soru}\n\n## 🤖 KodAI\n{cevap}\n\n---\n\n")
    console.print(f"[green]✅ Kaydedildi: {dosya_adi}[/green]")

def gecmis_goster(gecmis: list, n: int = 5):
    if not gecmis:
        console.print("[dim]Henüz konuşma yok.[/dim]"); return
    son = gecmis[-n:]
    tablo = Table(box=box.ROUNDED, border_style="dim")
    tablo.add_column("#", style="dim", width=3)
    tablo.add_column("Siz", style="cyan", max_width=45)
    tablo.add_column("KodAI", style="white", max_width=45)
    for i, (soru, cevap) in enumerate(son, len(gecmis)-len(son)+1):
        tablo.add_row(str(i),
                      soru[:45] + ("…" if len(soru) > 45 else ""),
                      cevap[:45] + ("…" if len(cevap) > 45 else ""))
    console.print(Panel(tablo, title=f"[bold]📜 Son {len(son)} konuşma[/bold]", border_style="dim"))

def gecmis_ara(gecmis: list, kelime: str):
    kl = kelime.lower()
    bulunanlar = [(i+1, s, c) for i, (s, c) in enumerate(gecmis) if kl in s.lower() or kl in c.lower()]
    if not bulunanlar:
        console.print(f"[dim]'{kelime}' bulunamadı.[/dim]"); return
    tablo = Table(box=box.ROUNDED, border_style="yellow")
    tablo.add_column("#", style="dim", width=3)
    tablo.add_column("Soru", style="cyan", max_width=60)
    for no, soru, _ in bulunanlar[-10:]:
        tablo.add_row(str(no), soru[:60])
    console.print(Panel(tablo, title=f"[bold yellow]🔍 '{kelime}' — {len(bulunanlar)} sonuç[/bold yellow]", border_style="yellow"))

def gecmis_yukle():
    dosya = ".kodai_gecmis.json"
    if os.path.exists(dosya):
        try:
            with open(dosya, "r", encoding="utf-8") as f: return json.load(f)
        except: pass
    return []

def gecmis_kaydet(gecmis: list):
    try:
        with open(".kodai_gecmis.json", "w", encoding="utf-8") as f:
            json.dump(gecmis[-100:], f, ensure_ascii=False, indent=2)
    except: pass

# ─── ANA DÖNGÜ ──────────────────────────────────────────────────────────────

def main():
    print_banner()
    model_id, model_isim = model_sec()
    console.print(f"\n[cyan]Model yükleniyor: [bold]{model_isim}[/bold]...[/cyan]\n")
    with Live(Spinner("bouncingBall", text=f"[cyan]{model_isim} başlatılıyor...[/cyan]"), refresh_per_second=10):
        ai = TurkishAI(model_size="small")
        ai.initialize(model_name=model_id)
    console.print(f"[bold green]✅ {model_isim} hazır![/bold green]\n")
    print_help()

    session = PromptSession(
        history=FileHistory(".kodai_history"),
        auto_suggest=AutoSuggestFromHistory(),
        style=prompt_style,
    )
    gecmis = gecmis_yukle()
    son_cevap = ""
    if gecmis:
        console.print(f"[dim]📂 {len(gecmis)} önceki konuşma yüklendi.[/dim]")

    while True:
        try:
            user_input = session.prompt(
                HTML(f'\n<ansigreen><b>[{AYARLAR["model_isim"]}]</b></ansigreen><ansicyan>❯ </ansicyan>')
            ).strip()
            if not user_input:
                continue

            if user_input == "/exit":
                gecmis_kaydet(gecmis)
                console.print("[dim]💾 Geçmiş kaydedildi.[/dim]")
                console.print("\n[bold cyan]👋 Görüşmek üzere![/bold cyan]")
                break
            elif user_input == "/help":
                print_help()
            elif user_input == "/temizle":
                console.clear(); print_banner(); console.print(durum_satiri())
            elif user_input == "/reset":
                ai.reset_conversation(); gecmis.clear(); son_cevap = ""
                console.print("[green]✅ Konuşma sıfırlandı.[/green]")
            elif user_input == "/kopyala":
                if son_cevap:
                    subprocess.run(["pbcopy"], input=son_cevap.encode())
                    console.print("[green]✅ Kopyalandı![/green]")
                else:
                    console.print("[yellow]⚠️ Kopyalanacak cevap yok.[/yellow]")
            elif user_input == "/stream":
                AYARLAR["streaming"] = not AYARLAR["streaming"]
                console.print(f"[green]✅ Streaming {'açık ⚡' if AYARLAR['streaming'] else 'kapalı ⏳'}[/green]")
            elif user_input.startswith("/gecmis"):
                p = user_input.split()
                gecmis_goster(gecmis, int(p[1]) if len(p) > 1 and p[1].isdigit() else 5)
            elif user_input.startswith("/ara "):
                gecmis_ara(gecmis, user_input[5:].strip())
            elif user_input.startswith("/kaydet"):
                p = user_input.split(maxsplit=1)
                cmd_kaydet_konusma(gecmis, p[1] if len(p) > 1 else "")
            elif user_input == "/stats":
                stats = ai.get_stats()
                tablo = Table(box=box.ROUNDED, border_style="cyan")
                tablo.add_column("Özellik", style="cyan")
                tablo.add_column("Değer", style="white")
                for k, v in stats.items():
                    tablo.add_row(str(k), str(v))
                tablo.add_row("sıcaklık", str(AYARLAR["temperature"]))
                tablo.add_row("max_token", str(AYARLAR["max_len"]))
                tablo.add_row("streaming", "✅ açık" if AYARLAR["streaming"] else "❌ kapalı")
                tablo.add_row("konuşma", str(len(gecmis)))
                tablo.add_row("snippet", str(len(snippet_yukle())))
                console.print(Panel(tablo, title="[bold]📊 KodAI v6 İstatistikleri[/bold]", border_style="cyan"))
            elif user_input.startswith("/sicaklik "):
                try:
                    val = float(user_input.split()[1])
                    if 0.1 <= val <= 2.0:
                        AYARLAR["temperature"] = val
                        console.print(f"[green]✅ Sıcaklık: {val}[/green]")
                    else:
                        console.print("[red]❌ 0.1-2.0 arasında olmalı[/red]")
                except:
                    console.print("[red]❌ Geçersiz değer[/red]")
            elif user_input.startswith("/uzunluk "):
                try:
                    val = int(user_input.split()[1])
                    if 100 <= val <= 2000:
                        AYARLAR["max_len"] = val
                        console.print(f"[green]✅ Max token: {val}[/green]")
                    else:
                        console.print("[red]❌ 100-2000 arasında olmalı[/red]")
                except:
                    console.print("[red]❌ Geçersiz değer[/red]")
            elif user_input == "/model":
                model_id, model_isim = model_sec()
                with Live(Spinner("bouncingBall", text=f"[cyan]{model_isim}...[/cyan]"), refresh_per_second=10):
                    ai = TurkishAI(model_size="small")
                    ai.initialize(model_name=model_id)
                console.print(f"[bold green]✅ {model_isim} hazır![/bold green]")
            elif user_input.startswith("/snippet"):
                cmd_snippet(user_input[8:].strip(), son_cevap)
            elif user_input.startswith("/diff "):
                p = user_input[6:].split(maxsplit=1)
                if len(p) == 2: cmd_diff(p[0], p[1])
                else: console.print("[red]❌ /diff <dosya1> <dosya2>[/red]")
            elif user_input.startswith("/optimize "):
                cmd_optimize(ai, user_input[10:].strip())
            elif user_input.startswith("/karmasiklik "):
                cmd_karmasiklik(user_input[13:].strip())
            elif user_input.startswith("/rename "):
                cmd_rename(ai, user_input[8:].strip())
            elif user_input.startswith("/dok "):
                cmd_dok(ai, user_input[5:].strip())
            elif user_input.startswith("/todo"):
                cmd_todo(user_input[5:].strip() or ".")
            elif user_input.startswith("/bagim"):
                cmd_bagim(user_input[6:].strip() or ".")
            elif user_input.startswith("/dosyalar "):
                cmd_dosyalar(ai, user_input[10:].split())
            elif user_input.startswith("/dosya "):
                cmd_dosya(ai, user_input[7:].strip())
            elif user_input.startswith("/ozet "):
                cmd_ozet(ai, user_input[6:].strip())
            elif user_input.startswith("/acikla "):
                cmd_acikla(ai, user_input[8:].strip())
            elif user_input.startswith("/duzenle "):
                cmd_duzenle(ai, user_input[9:].strip())
            elif user_input.startswith("/calistir "):
                cmd_calistir(ai, user_input[10:].strip())
            elif user_input.startswith("/yaz "):
                cmd_yaz(ai, user_input[5:].strip())
            elif user_input.startswith("/test "):
                cmd_test(ai, user_input[6:].strip())
            elif user_input.startswith("/refactor "):
                cmd_refactor(ai, user_input[10:].strip())
            elif user_input.startswith("/hata"):
                cmd_hata(ai, user_input[5:].strip())
            elif user_input.startswith("/git"):
                cmd_git(ai, user_input[4:].strip())
            elif user_input.startswith("/proje "):
                cmd_proje(ai, user_input[7:].strip())
            elif user_input.startswith("/bul "):
                p = user_input[5:].split(maxsplit=1)
                cmd_bul(p[0], p[1] if len(p) > 1 else ".")
            elif user_input.startswith("/"):
                console.print(f"[red]❌ Bilinmeyen komut:[/red] [bold]{user_input.split()[0]}[/bold]  [dim]/help[/dim]")
            else:
                console.print(Panel(f"[white]{user_input}[/white]",
                    title="[bold blue]👤 Siz[/bold blue]", border_style="blue", box=box.ROUNDED))
                response = ai_cevap(ai, user_input, komut_tipi="genel")
                gecmis.append((user_input, response))
                son_cevap = response
                if len(gecmis) % 10 == 0:
                    gecmis_kaydet(gecmis)

        except KeyboardInterrupt:
            console.print("\n[bold cyan]👋 Görüşmek üzere![/bold cyan]")
            gecmis_kaydet(gecmis)
            break
        except EOFError:
            break
        except Exception as e:
            console.print(f"[red]❌ Hata: {e}[/red]")

if __name__ == "__main__":
    main()
