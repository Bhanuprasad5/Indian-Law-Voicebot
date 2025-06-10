# 🇮🇳 Indian Law Voicebot ⚖️  
A powerful voice-enabled assistant that helps users interact with Indian law through natural language. Speak your legal question and receive voice-based responses backed by law texts.

---

## 📌 Features

- 🎙️ **Voice Input** – Ask your legal queries using speech.
- 🧠 **Speech-to-Text** – Converts voice into text using `speech_recognition`.
- 📚 **RAG Pipeline** – Retrieves relevant Indian law sections using LangChain and Chroma DB.
- 🗣️ **Text-to-Speech** – Delivers answers back using `gTTS` (Google Text-to-Speech).
- 🧾 **Streamlit UI** – Clean, interactive web interface to interact with the bot.
- 📦 **Docker Deployment** – Containerized setup for easy deployment on Hugging Face Spaces.

---

## 🏗️ Tech Stack

| Layer              | Tools Used                                            |
|-------------------|--------------------------------------------------------|
| Frontend          | Streamlit                                             |
| Backend           | Python, LangChain, ChromaDB                           |
| Speech-to-Text    | `speech_recognition` library with Google API          |
| Text-to-Speech    | `gTTS` (Google Text-to-Speech)                        |
| Vector DB         | Chroma with sentence-transformer embeddings           |
| Containerization  | Docker                                                |
| Deployment        | Streamlit                                   |

---

## ⚙️ Installation

### 🔧 Prerequisites

- Python 3.9+
- `ffmpeg` installed (for audio processing)
- Docker (optional for containerized deployment)

### 📥 Clone the Repository

```bash
git clone https://github.com/Bhanuprasad5/Indian-Law-Voicebot.git
cd Indian-Law-Voicebot
