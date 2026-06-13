# 🕹️ Arca-de: Multi-Capability AI Chatbot

Arca-de is a web-based AI chatbot built with **Streamlit**, **LangChain**, and **LangGraph**. The application integrates Google's **Gemini** model with multiple external tools, enabling the chatbot to perform real-time web searches, generate images, answer questions from PDF documents, and analyze uploaded images.

To maintain responsiveness and efficiency, Arca-de implements a **Hybrid Router System** that automatically distinguishes between simple conversational requests (fast path) and complex tasks requiring agent reasoning and tool usage (agent path).

---

# ✨ Features

## ⚡ Fast Casual Chat
Responds instantly to greetings, casual conversations, and simple questions without invoking the full agent reasoning workflow.

## 🌐 Real-Time Web Search
Retrieves up-to-date information, news, and fact verification using the **Exa Search API**.

## 🎨 AI Image Generation
Creates images directly from text prompts using **Google Gemini** image generation capabilities.

## 📄 PDF Document Q&A (RAG)
Upload PDF documents and allow the chatbot to learn their contents using **Chroma Vector Store**, enabling natural language questions and answers based on the uploaded files.

## 👁️ Image Analysis & Description
Upload PNG or JPG images and ask the chatbot to analyze and describe objects, scenes, and visual content.

## 🔄 Session-Safe Reset
Reset conversations, temporary files, and vector database memory with a single click from the sidebar.

---

# 🏗️ System Architecture

```text
User
 │
 ▼
Hybrid Router
 │
 ├── Fast Path
 │     └── Casual Conversations
 │
 └── Agent Path
       │
       ├── Web Search Tool
       ├── Image Generation Tool
       ├── PDF RAG Tool
       └── Image Analysis Tool
               │
               ▼
           Gemini Model
```

---

# 🛠️ Technology Stack

| Technology | Purpose |
|------------|---------|
| Streamlit | Interactive web application interface |
| LangGraph | Agent orchestration and workflow management |
| LangChain | LLM application framework |
| Google Gemini | Large Language Model |
| ChromaDB | Vector database for Retrieval-Augmented Generation (RAG) |
| PyPDF | PDF document processing |
| Exa API | AI-powered web search |
| Python | Core programming language |

---

# 🚀 Local Installation

## 1. Clone the Repository

```bash
git clone https://github.com/bagussam/arca_de.git
cd arca_de
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Configure API Keys

Create a `.streamlit` directory:

```bash
mkdir .streamlit
```

Create the file:

```text
.streamlit/secrets.toml
```

Add your API credentials:

```toml
GOOGLE_API_KEY = "AIzaSy..."
EXA_API_KEY = "your-exa-api-key-here"
```

> ⚠️ Never commit API keys or credentials to GitHub.

---

# ▶️ Run the Application

```bash
streamlit run arcade.py
```

By default, the application will be available at:

```text
http://localhost:8501
```

---

# ☁️ Deployment on Streamlit Community Cloud

To deploy Arca-de on Streamlit Community Cloud:

### 1. Open Streamlit Community Cloud

Sign in and select your deployed application.

### 2. Configure Secrets

Navigate to:

```text
Settings → Secrets
```

Add the following configuration:

```toml
GOOGLE_API_KEY = "AIzaSy..."
EXA_API_KEY = "your-exa-api-key-here"
```

### 3. Save and Reboot

Save the configuration and reboot the application if required.

---

# 📂 Project Structure

```text
arca_de/
│
├── .streamlit/
│   └── secrets.toml          # Local API keys (do not push to GitHub)
│
├── arcade.py                 # Main Streamlit application and AI agent
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

# 🔒 Security Best Practices

- Store API keys only in `.streamlit/secrets.toml`
- Add sensitive files to `.gitignore`
- Never expose credentials publicly
- Rotate API keys periodically
- Use environment-specific secrets for production deployments

Example `.gitignore`:

```gitignore
.streamlit/secrets.toml

__pycache__/
*.pyc
.env
venv/
```

---

# 📋 Requirements

- Python 3.10+
- Google Gemini API Key
- Exa API Key
- Internet connection

---

# 🎯 Supported Capabilities

| Capability | Supported |
|------------|------------|
| Conversational Chat | ✅ |
| Real-Time Web Search | ✅ |
| PDF Question Answering (RAG) | ✅ |
| Image Analysis | ✅ |
| AI Image Generation | ✅ |
| Session Reset | ✅ |

---

# 🧠 How the Hybrid Router Works

Arca-de improves performance by classifying incoming requests into two categories:

### Fast Path
Used for:

- Greetings
- Small talk
- Simple questions
- General conversation

These requests are sent directly to Gemini for immediate responses.

### Agent Path
Used for:

- Web searches
- Document-based questions
- Image generation
- Image analysis
- Multi-step reasoning

These requests activate the LangGraph agent and relevant tools before generating a response.

This architecture significantly reduces latency while preserving advanced capabilities when needed.

---

# 📈 Future Enhancements

- Multi-PDF knowledge base
- Persistent user memory
- Voice input and output
- YouTube content analysis
- Database connectivity
- Multi-agent collaboration
- Authentication and user accounts
- Conversation export functionality

---

# 📄 License

This project is licensed under the MIT License.

Feel free to use, modify, and distribute this software in accordance with the license terms.

---

# 👨‍💻 Author

**Bagus Samudro Aji Luhur**
