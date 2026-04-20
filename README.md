# 🤖 KodAI — Offline Turkish AI Coding Assistant

> A fully local Turkish AI assistant running on Mac M4.
> No internet connection required. Your data never leaves your device.

## 🚀 Features

- **Fully Offline** — Qwen2.5-Coder model runs locally via Apple MPS
- **Streaming Output** — Real-time responses via threading
- **35+ Commands** — Code writing, debugging, refactoring, testing, git and more
- **Command-specific AI** — 7 different system prompts (different mode for coding, different for error analysis)
- **AST-based Analysis** — Cyclomatic complexity scoring without AI, instant results
- **HTTP API Server** — Ollama-compatible API (port 5001)
- **Autonomous Learning** — Background data collection and learning engine
- **Scenario Engine** — Statistics-based pattern recognition and prediction

## 📋 Requirements

- macOS (Apple Silicon M1/M2/M3/M4 recommended)
- Python 3.10+
- ~6GB RAM (3B model) or ~14GB (7B model)

## ⚙️ Installation

```bash
git clone https://github.com/kaandevs-ops/kodai.git
cd turkish-ai
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

## 🛠 Commands

| Group | Command | Description |
|-------|---------|-------------|
| 💬 Chat | `/help` | Help menu |
| 💬 Chat | `/reset` | Reset conversation |
| 💬 Chat | `/gecmis` | Show history |
| 📁 File | `/dosya <path>` | File analysis |
| 📁 File | `/diff <path1> <path2>` | Compare files |
| 🛠 Code | `/test <path>` | Write pytest tests |
| 🛠 Code | `/refactor <path>` | Refactor code |
| 🛠 Code | `/optimize <path>` | Performance analysis |
| 🛠 Code | `/karmasiklik <path>` | Complexity score |
| 🛠 Code | `/rename <path>` | Rename variables/functions |
| 🛠 Code | `/hata` | Error analysis |
| 🔧 Project | `/proje <folder>` | Project analysis |
| 🔧 Project | `/git <command>` | Git assistance |
| 🔧 Project | `/todo` | List TODOs |
| 🔧 Project | `/bagim` | Dependency analysis |
| ⚙️ System | `/stream` | Toggle streaming |
| ⚙️ System | `/sicaklik 0.1-2.0` | Creativity setting |
| ⚙️ System | `/model` | Switch model |

## 🧠 Architecture

```
turkish_ai/
├── main.py                   # Terminal UI (KodAI v6)
├── server_api.py             # HTTP API server
├── autonomous_learner.py     # Autonomous learning engine
├── scenario_engine.py        # Scenario and prediction engine
├── prepare_data.py           # Data preparation
├── train_tokenizer.py        # BPE tokenizer training
├── configs/
│   └── default.yaml          # Configuration
└── src/
    ├── core/
    │   └── ai_engine.py      # Core AI engine (v6)
    ├── models/
    │   └── transformer.py    # Transformer architecture
    ├── training/
    │   └── trainer.py        # Training loop
    ├── tokenizer/
    │   └── bpe_tokenizer.py  # BPE tokenizer
    ├── memory/
    │   └── memory_system.py  # Long-term memory
    ├── data/
    │   └── data_collector.py # Data collection
    └── utils/
        └── logger.py         # Logging
```

## 📊 Model Options

| Model | Size | RAM | Use Case |
|-------|------|-----|----------|
| Qwen2.5-Coder-3B | ~6GB | 8GB+ | Daily use |
| Qwen2.5-Coder-7B | ~14GB | 16GB+ | Complex projects |

## 📄 License

MIT
