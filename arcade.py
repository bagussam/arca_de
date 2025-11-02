# (Dummy commit) Memaksa Streamlit untuk rebuild
# Import library yang diperlukan
import streamlit as st
import google.generativeai as genai
from exa_py import Exa
# SystemMessage dihapus karena tidak lagi diperlukan oleh agen versi baru
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
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
import textwrap # <- Untuk membersihkan indentasi prompt
import traceback # <- Untuk menampilkan error lengkap

# --- 1. Konfigurasi Halaman dan Judul ---
st.set_page_config(page_title="Arca-de", page_icon="🕹️", layout="wide")
st.title("🕹️ Arca-de")
st.caption("AI Chatbot dengan beragam kemampuan")

# --- 2. Inisialisasi API Keys dan Tools ---

# Inisialisasi API keys dari Streamlit Secrets
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    EXA_API_KEY = st.secrets["EXA_API_KEY"]
    
    # Validasi keys (memastikan tidak hanya ada, tapi juga tidak kosong)
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
    try:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
    except Exception as e:
        st.error(f"Gagal menginisialisasi model embeddings: {e}")
        st.stop()

# --- Fungsi Tools untuk Agen ---
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
        image_model = genai.GenerativeModel('gemini-1.5-pro')
        response = image_model.generate_content([prompt], generation_config={"response_mime_type": "image/png"})
        
        # Ekstrak data gambar
        if not response.parts:
            return "Gagal menghasilkan gambar: Tidak ada data gambar yang diterima dari API."
            
        # PERBAIKAN: Menggunakan .blob, bukan .inline_data.data
        image_data = response.parts[0].blob
        image = Image.open(io.BytesIO(image_data))
        
        # Simpan gambar sementara
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
        
        if "embeddings" not in st.session_state:
             return "Error: Model Embeddings belum siap."
             
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
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt_template)
        return response.text
    except Exception as e:
        return f"Terjadi kesalahan saat menjawab dari dokumen: {e}"

@tool
def describe_image(file_path: str):
    """Gunakan tool ini untuk menganalisis dan mendeskripsikan isi dari sebuah gambar yang di-upload."""
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        image = Image.open(file_path)
        prompt = "Deskripsikan gambar ini secara detail."
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"Gagal mendeskripsikan gambar: {e}"

# --- 3. Inisialisasi Agen LangGraph & Model Chat Sederhana ---
# Definisikan kata kunci yang akan memicu agen canggih
AGENT_KEYWORDS = ["gambar", "buatkan", "ciptakan", "cari", "carikan", "internet", "pdf", "dokumen", "analisis", "deskripsikan"]

if "agent" not in st.session_state:
    try:
        # 1. Inisialisasi Model "Otak" Agen (yang akan memanggil tools)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0.7, 
            google_api_key=GOOGLE_API_KEY
        )
        tools = [web_search, generate_image, process_document, answer_from_document, describe_image]
        
        # Membersihkan indentasi pada system prompt
        system_prompt_text = textwrap.dedent("""
            You are a helpful assistant with powerful tools.

            IMPORTANT:
            - If the user asks to 'generate', 'create', 'draw', or 'make an image of' something, you MUST use the 'generate_image' tool.
            - For factual or recent questions, use the 'web_search' tool.
            - For questions about a document, use the 'answer_from_document' tool after it has been processed.
            - To describe a user-uploaded image, use the 'describe_image' tool.
            - Otherwise, answer like a friendly chatbot.
            """)
        
        # 2. Inisialisasi Agen (Canggih tapi Lambat)
        st.session_state.agent = create_react_agent(
            model=llm,
            tools=tools,
            # PERBAIKAN FINAL: Menggunakan 'system_message', bukan 'messages_modifier'
            system_message=system_prompt_text
        )
        
        # 3. Inisialisasi Model Chat Sederhana (Cepat)
        st.session_state.simple_chat = genai.GenerativeModel('gemini-1.5-flash')

    except Exception as e:
        # Kita tambahkan traceback untuk melihat error lengkapnya
        tb_str = traceback.format_exc()
        st.error(f"Gagal menginisialisasi agen AI: {e}\n\nTraceback lengkap:\n{tb_str}")
        st.stop()


# --- 4. Sidebar dengan File Uploader ---
with st.sidebar:
    st.header("Pengaturan")
    if st.button("Reset Percakapan"):
        st.session_state.messages = []
        if os.path.exists("temp_generated_image.png"): 
            try:
                os.remove("temp_generated_image.png")
            except:
                pass # Abaikan jika file tidak bisa dihapus
        # Hapus juga vector store jika ada
        if "vector_store" in st.session_state:
            del st.session_state.vector_store
        if "processed_file" in st.session_state:
            del st.session_state.processed_file
        st.rerun()

    uploaded_file = st.file_uploader("Upload PDF atau Gambar", type=["pdf", "png", "jpg", "jpeg"])
    
    if uploaded_file and ("processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name):
        with st.spinner(f"Memproses file {uploaded_file.name}..."):
            try:
                # PERBAIKAN: Menghapus typo 'J'
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    file_path = tmp_file.name

                if uploaded_file.type == "application/pdf":
                    tool_to_use = "process_document"
                else:
                    tool_to_use = "describe_image"
                
                # Buat pesan untuk memicu agen memproses file
                process_prompt = f"Pengguna telah mengupload file bernama '{uploaded_file.name}'. Gunakan tool '{tool_to_use}' untuk memproses file yang ada di path: {file_path}"
                
                # Inisialisasi messages jika belum ada
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                    
                st.session_state.messages.append({"role": "user", "content": process_prompt})
                
                # Panggil agen untuk mendapatkan konfirmasi
                history = [HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]) for msg in st.session_state.messages]
                
                if "agent" not in st.session_state:
                     st.error("Agen AI belum siap. Silakan refresh halaman.")
                     st.stop()
                     
                response = st.session_state.agent.invoke({"messages": history})
                
                # Pastikan response adalah AIMessage
                if isinstance(response.get('messages', [])[-1], AIMessage):
                    result_message = response['messages'][-1].content
                else:
                    result_message = str(response) # Fallback
                
                st.session_state.messages.append({"role": "assistant", "content": result_message})
                st.session_state.processed_file = uploaded_file.name
                
                # Hapus file temp setelah diproses
                if os.path.exists(file_path):
                    os.remove(file_path)
                    
                st.rerun()
            
            except Exception as e:
                tb_str = traceback.format_exc()
                st.error(f"Gagal memproses file upload: {e}\n\nTraceback:\n{tb_str}")
                # Hapus file temp jika terjadi error
                if 'file_path' in locals() and os.path.exists(file_path):
                    os.remove(file_path)


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
                # PERBAIKAN: Logika "Router"
                # Cek apakah ada kata kunci yang butuh agen
                use_agent = any(keyword in prompt.lower() for keyword in AGENT_KEYWORDS)
                
                if use_agent:
                    # Jika ada kata kunci, panggil AGEN (Lambat tapi canggih)
                    history = [HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]) for msg in st.session_state.messages]
                    
                    if "agent" not in st.session_state:
                        st.error("Agen AI belum siap. Silakan refresh halaman.")
                        st.stop()
                        
                    response = st.session_state.agent.invoke({"messages": history})
                    
                    if isinstance(response.get('messages', [])[-1], AIMessage):
                        answer = response['messages'][-1].content
                    else:
                        answer = str(response) # Fallback
                
                else:
                    # Jika tidak ada kata kunci, panggil CHAT SEDERHANA (Cepat)
                    if "simple_chat" not in st.session_state:
                        st.error("Model chat belum siap. Silakan refresh halaman.")
                        st.stop()
                    
                    # Buat riwayat chat sederhana
                    chat_history = []
                    for msg in st.session_state.messages[:-1]: # Ambil semua KECUALI prompt terakhir
                        chat_history.append({"role": "user" if msg["role"] == "user" else "model", "parts": [msg["content"]]})
                    
                    chat_session = st.session_state.simple_chat.start_chat(history=chat_history)
                    response = chat_session.send_message(prompt)
                    answer = response.text

                # --- Logika Menampilkan Respons ---
                if "Gambar berhasil dibuat" in answer:
                    image_path = "temp_generated_image.png"
                    if os.path.exists(image_path):
                        st.image(image_path, caption="Gambar yang dihasilkan AI")
                        st.session_state.messages.append({"role": "assistant", "content": image_path, "type": "image"})
                        st.session_state.messages.append({"role": "assistant", "content": "Berikut adalah gambarnya.", "type": "text"})
                        os.remove(image_path) # Hapus setelah ditampilkan
                    else:
                        st.error("Gambar seharusnya dibuat, tapi file sementara tidak ditemukan.")
                        st.session_state.messages.append({"role": "assistant", "content": "Maaf, terjadi kesalahan saat menampilkan gambar.", "type": "text"})
                else:
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer, "type": "text"})
            
            except Exception as e:
                tb_str = traceback.format_exc()
                error_message = f"Terjadi kesalahan: {e}\n\nTraceback:\n{tb_str}"
                st.error(error_message)
                # Ini adalah baris 252 yang telah diperbaiki:
                st.session_state.messages.append({"role": "assistant", "content": error_message, "type": "text"})

