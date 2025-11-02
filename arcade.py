# (Dummy commit) Memaksa Streamlit untuk rebuild
# Import library yang diperlukan
import streamlit as st
import google.generativeai as genai
from exa_py import Exa
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from PIL import Image
import io
import os
import tempfile
import textwrap # <- TAMBAHAN BARU UNTUK MEMBERSIHKAN PROMPT

# --- 1. Konfigurasi Halaman dan Judul ---
st.set_page_config(page_title="Arca-de", page_icon="🕹️", layout="wide")
st.title("🕹️ Arca-de")
st.caption("AI Chatbot dengan beragam kemampuan")

# --- 2. Inisialisasi API Keys dan Tools ---

# Inisialisasi API keys dari Streamlit Secrets
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    EXA_API_KEY = st.secrets["EXA_API_KEY"]
    
    # PERBAIKAN: Pastikan keys tidak hanya ada, tapi juga tidak kosong
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY di Streamlit Secrets ditemukan, tetapi nilainya kosong. Harap isi nilainya.")
        st.stop()
    if not EXA_API_KEY:
        st.error("EXA_API_KEY di Streamlit Secrets ditemukan, tetapi nilainya kosong. Harap isi nilainya.")
        st.stop()
        
    genai.configure(api_key=GOOGLE_API_KEY)

except KeyError as e:
    # Error jika key-nya tidak ada sama sekali
    st.error(f"API Key '{e.args[0]}' tidak ditemukan di Streamlit Secrets. Harap tambahkan.")
    st.stop()
except Exception as e:
    # Error umum lainnya
    st.error(f"Error saat memuat API Keys: {e}")
    st.stop()

# Inisialisasi model Embeddings sekali per sesi
if "embeddings" not in st.session_state:
    # PERBAIKAN: Teruskan juga API key ke model embeddings
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

# Fungsi Tools untuk Agen
@tool
def web_search(query: str):
    """Gunakan tool ini untuk mencari informasi terbaru atau faktual di internet."""
    try:
        exa = Exa(api_key=EXA_API_KEY)
        results = exa.search_and_contents(query, use_autoprompt=True, num_results=3, text=True)
        return str(results)
    except Exception as e:
        return f"Error saat melakukan pencarian: {e}"

@tool
def generate_image(prompt: str):
    """Gunakan tool ini untuk membuat atau menghasilkan gambar berdasarkan deskripsi teks."""
    try:
        # PENTING: Gunakan model 'pro' untuk membuat gambar
        image_model = genai.GenerativeModel('gemini-1.5-pro')
        response = image_model.generate_content([prompt], generation_config={"response_mime_type": "image/png"})
        
        image_data = response.parts[0].blob
        image = Image.open(io.BytesIO(image_data))
        
        temp_image_path = "temp_generated_image.png"
        image.save(temp_image_path)
        return f"Gambar berhasil dibuat dan disimpan sebagai {temp_image_path}. Tampilkan gambar ini dan berikan deskripsi singkat."
    except Exception as e:
        return f"Error saat membuat gambar: {e}"

@tool
def process_document(file_path: str):
    """Gunakan tool ini untuk memproses dan mempelajari isi dokumen PDF yang di-upload pengguna."""
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        st.session_state.vector_store = Chroma.from_documents(documents=chunks, embedding=st.session_state.embeddings)
        return f"Dokumen '{os.path.basename(file_path)}' berhasil diproses. Sekarang pengguna bisa bertanya tentang isinya."
    except Exception as e:
        return f"Gagal memproses dokumen: {e}"

@tool
def answer_from_document(query: str):
    """Gunakan tool ini SETELAH dokumen diproses untuk menjawab pertanyaan spesifik tentang isi dokumen tersebut."""
    if "vector_store" not in st.session_state:
        return "Dokumen belum diproses. Harap proses dokumen terlebih dahulu menggunakan tool 'process_document'."
    
    try:
        relevant_docs = st.session_state.vector_store.similarity_search(query)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        prompt_template = f"Berdasarkan konteks berikut, jawab pertanyaan pengguna.\nKonteks:\n{context}\n\nPertanyaan: {query}"
        # PENTING: Gunakan model 'flash' atau 'pro' yang benar
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt_template)
        return response.text
    except Exception as e:
        return f"Terjadi kesalahan saat menjawab dari dokumen: {e}"

@tool
def describe_image(file_path: str):
    """Gunakan tool ini untuk menganalisis dan mendeskripsikan isi dari sebuah gambar yang di-upload."""
    try:
        # PENTING: Gunakan model 'pro' yang mampu memproses gambar
        model = genai.GenerativeModel('gemini-1.5-pro')
        image = Image.open(file_path)
        prompt = "Deskripsikan gambar ini secara detail."
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"Gagal mendeskripsikan gambar: {e}"

# --- 3. Inisialisasi Agen LangGraph (PERBAIKAN INDENTASI PROMPT) ---
if "agent" not in st.session_state:
    try:
        # PERBAIKAN: Teruskan GOOGLE_API_KEY secara eksplisit ke ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0.7, 
            google_api_key=GOOGLE_API_KEY # <-- Gunakan variabel yang sudah divalidasi
        )
        tools = [web_search, generate_image, process_document, answer_from_document, describe_image]
        
        # 1. Gunakan textwrap.dedent untuk membersihkan spasi ekstra
        system_prompt_text = textwrap.dedent("""
            You are a helpful assistant with powerful tools.

            IMPORTANT:
            - If the user asks to 'generate', 'create', 'draw', or 'make an image of' something, you MUST use the 'generate_image' tool.
            - For factual or recent questions, use the 'web_search' tool.
            - For questions about a document, use the 'answer_from_document' tool after it has been processed.
            - To describe a user-uploaded image, use the 'describe_image' tool.
            - Otherwise, answer like a friendly chatbot.
        """)
        
        # 2. Panggil 'create_react_agent' dengan argumen yang benar
        st.session_state.agent = create_react_agent(
            model=llm,
            tools=tools,
            messages_modifier=SystemMessage(content=system_prompt_text)
        )
    
    except Exception as e:
        st.error(f"Gagal menginisialisasi agen AI: {e}")
        st.stop()

# --- 4. Sidebar dengan File Uploader ---
with st.sidebar:
    st.header("Pengaturan")
    if st.button("Reset Percakapan"):
        st.session_state.messages = []
        if os.path.exists("temp_generated_image.png"): os.remove("temp_generated_image.png")
        st.rerun()

    uploaded_file = st.file_uploader("Upload PDF atau Gambar", type=["pdf", "png", "jpg", "jpeg"])
    
    if uploaded_file and ("processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name):
        with st.spinner(f"Memproses file {uploaded_file.name}..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                file_path = tmp_file.name

            if uploaded_file.type == "application/pdf":
                tool_to_use = "process_document"
            else:
                tool_to_use = "describe_image"
            
            # Buat pesan untuk memicu agen memproses file
            process_prompt = f"Pengguna telah mengupload file bernama '{uploaded_file.name}'. Gunakan tool '{tool_to_use}' untuk memproses file yang ada di path: {file_path}"
            
            # Pastikan st.session_state.messages ada sebelum di-append
            if "messages" not in st.session_state:
                st.session_state.messages = []
                
            st.session_state.messages.append({"role": "user", "content": process_prompt})
            
            # Panggil agen untuk mendapatkan konfirmasi
            history = [HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]) for msg in st.session_state.messages]
            response = st.session_state.agent.invoke({"messages": history})
            result_message = response['messages'][-1].content
            
            st.session_state.messages.append({"role": "assistant", "content": result_message})
            st.session_state.processed_file = uploaded_file.name
            st.rerun()

# --- 5. Manajemen dan Tampilan Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "image":
            st.image(msg["content"], caption="Gambar yang dihasilkan AI")
        else:
            st.markdown(msg["content"])

# --- 6. Logika Input dan Respons ---
prompt = st.chat_input("Tanya saya, apa saja...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Sedang berpikir..."):
            try:
                history = [HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]) for msg in st.session_state.messages]
                response = st.session_state.agent.invoke({"messages": history})
                answer = response['messages'][-1].content

                if "Gambar berhasil dibuat" in answer:
                    image_path = "temp_generated_image.png"
                    if os.path.exists(image_path):
                        st.image(image_path, caption="Gambar yang dihasilkan AI")
                        st.session_state.messages.append({"role": "assistant", "content": image_path, "type": "image"})
                        os.remove(image_path)
                    else:
                        st.error("Gagal menemukan gambar yang telah dibuat.")
                        st.session_state.messages.append({"role": "assistant", "content": "Maaf, terjadi kesalahan saat mencoba menampilkan gambar.", "type": "text"})
                else:
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer, "type": "text"})
            except Exception as e:
                error_message = f"Terjadi kesalahan: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message, "type": "text"})


