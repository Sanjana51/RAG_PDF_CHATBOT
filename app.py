import os
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from uuid import uuid4
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = 'uploads'
DATA_FOLDER = 'data'
CHROMA_DB_DIR = 'chroma_db'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_vector_store():
    # Always load fresh vectorstore from disk (persistent)
    return Chroma(
        collection_name="pdf_docs",
        embedding_function=embedding_model,
        persist_directory=CHROMA_DB_DIR
    )

chat_template = """
You are a highly intelligent assistant specialized in answering questions from PDF documents. Use the following extracted content from a document to answer the user's question.

Follow these guidelines:
- Answer clearly and concisely.
- Use bullet points or numbering when needed.
- If the context doesn't have the answer, say: "I couldn't find the answer in the document."
- Do not fabricate information. Use only the context provided.

Document Context:
{context}

User Question:
{question}

Answer:
"""

summary_template = """
You are an intelligent assistant helping summarize long documents like PDFs. Your goal is to extract the **most important and relevant content** and present it clearly.

Instructions:
- Start with a 2–3 line **executive summary**.
- Then provide a **detailed summary** in bullet points or paragraphs.
- Focus on **key ideas, major facts, and arguments**.
- Avoid adding information not in the document.

Here is the document content:
{context}

Summarize:
"""

def get_llm_response_gemini(prompt: str) -> str:
    generation_config = {
        "temperature": 0.3,
        "response_mime_type": "text/plain"
    }
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config
    )
    response = model.generate_content(prompt)
    return response.text

def upload_pdf(file):
    filename = secure_filename(file.filename)
    file_path = os.path.join(DATA_FOLDER, filename)
    file.save(file_path)
    return file_path

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()

def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
    return splitter.split_documents(documents)

def index_docs(documents):
    # Remove and recreate collection properly
    from chromadb import PersistentClient
    client = PersistentClient(path=CHROMA_DB_DIR)

    # Drop existing collection if it exists
    try:
        client.delete_collection("pdf_docs")
    except:
        pass  # safe to ignore if collection doesn't exist

    # Recreate it and re-index
    vector_store = Chroma(
        collection_name="pdf_docs",
        embedding_function=embedding_model,
        persist_directory=CHROMA_DB_DIR
    )

    ids = [str(uuid4()) for _ in documents]
    vector_store.add_documents(documents=documents, ids=ids)
    vector_store.persist()


def retrieve_docs(query):
    vector_store = get_vector_store()
    return vector_store.similarity_search(query, k=6)

def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])[:28000]
    prompt = chat_template.format(question=question, context=context)
    return get_llm_response_gemini(prompt)

def summarize_document(documents):
    full_context = "\n\n".join([doc.page_content for doc in documents])[:30000]
    prompt = summary_template.format(context=full_context)
    return get_llm_response_gemini(prompt)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'pdf_file' not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files['pdf_file']
        if file.filename == '':
            flash("No selected file")
            return redirect(request.url)
        if file:
            file_path = upload_pdf(file)
            documents = load_pdf(file_path)
            chunked_documents = split_text(documents)
            index_docs(chunked_documents)
            session['file_uploaded'] = True  # Mark upload done
            flash("PDF uploaded and indexed successfully!")
            return redirect(url_for('index'))
    return render_template('index.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if not session.get('file_uploaded'):
        flash("Please upload a PDF first.")
        return redirect(url_for('index'))
    answer = None
    question = None
    if request.method == 'POST':
        question = request.form.get('question', '').strip()
        if question:
            relevant_docs = retrieve_docs(question)
            answer = answer_question(question, relevant_docs)
    return render_template('chat.html', question=question, answer=answer)

@app.route('/summary', methods=['GET', 'POST'])
def summary():
    if not session.get('file_uploaded'):
        flash("Please upload a PDF first.")
        return redirect(url_for('index'))
    summary = None
    if request.method == 'POST':
        vector_store = get_vector_store()
        documents = vector_store.similarity_search(" ", k=100)
        if not documents:
            flash("No documents indexed yet.")
            return redirect(url_for('index'))
        summary = summarize_document(documents)
    return render_template('summary.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
