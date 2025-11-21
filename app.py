import os
import tempfile

import dotenv
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate

# ========================
# Config & Helpers
# ========================

dotenv.load_dotenv()  # Load GROQ_API_KEY from .env

# Make sure GROQ_API_KEY is set
if not os.getenv("GROQ_API_KEY"):
    st.warning("⚠️ GROQ_API_KEY not found in environment. Set it in a .env file or environment variable.")

# Text splitter (reuse)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

def load_docs_from_uploaded_files(uploaded_files):
    """Save uploaded PDFs to temp files and load them as LangChain documents."""
    docs = []
    for uploaded_file in uploaded_files:
        if uploaded_file.name.lower().endswith(".pdf"):
            # Save to a temp file so PyPDFLoader can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            file_docs = loader.load()
            # Add source metadata as the original uploaded filename
            for d in file_docs:
                d.metadata["source"] = uploaded_file.name
            docs.extend(file_docs)

    return docs


def build_vectorstore_from_docs(docs):
    """Create Chroma vector store from loaded documents."""

    if not docs:
        # No docs loaded – avoid calling Chroma with empty list
        raise ValueError("No text could be loaded from the uploaded PDFs. "
                         "Make sure they are not empty or password-protected.")

    chunks = text_splitter.split_documents(docs)

    if not chunks:
        raise ValueError("No text chunks were created from the documents. "
                         "Your PDFs may not contain extractable text.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
    )
    return vectordb


def build_qa_chain(vectordb, memory):
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
    )

    # Rephrase follow-up questions -> standalone question
    condense_question_prompt = PromptTemplate.from_template(
        "Given the following conversation history and a follow-up question, "
        "rephrase the follow-up question to be a standalone question that includes "
        "relevant context from the history. Keep it concise and in the original language.\n\n"
        "Chat History:\n{chat_history}\n\n"
        "Follow-Up Question: {input}\n\n"
        "Standalone Question:"
    )

    # Grounded answer prompt
    answer_prompt = ChatPromptTemplate.from_template(
        """You are a helpful AI tutor for college students.

Use ONLY the following context from the documents to answer the question.
If the answer is not in the context, say:
"I don't know from the documents. You may need to refer to your textbook or teacher."

Be clear, concise, and explain in simple terms. Use bullet points or steps if helpful.

Context:
{context}

Question:
{input}

Answer:"""
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, condense_question_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, answer_prompt)
    qa_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return qa_chain


# ========================
# Streamlit App
# ========================

st.set_page_config(page_title="Campus Knowledge Assistant", page_icon="📚", layout="wide")
st.title("📚 Campus Knowledge Assistant (RAG + LangChain + Groq)")

st.markdown(
    """
Upload your **college PDFs** (notes, syllabus, question papers) and ask questions.
The assistant uses **RAG (Retrieval-Augmented Generation)** with:
- 🧠 Groq LLaMA 3.1 as the LLM  
- 🔍 Chroma + sentence-transformers for semantic search  
- 💬 Conversational memory + custom prompts  
"""
)

# Sidebar – Upload & Process
st.sidebar.header("📂 Document Setup")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []  # for UI chat history


def process_documents():
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
        return

    with st.spinner("Processing documents... This may take a moment."):
        docs = load_docs_from_uploaded_files(uploaded_files)

        try:
            vectordb = build_vectorstore_from_docs(docs)
        except ValueError as e:
            st.error(f"❌ {e}")
            return

        st.session_state.vectordb = vectordb
        st.session_state.memory.clear()
        st.session_state.qa_chain = build_qa_chain(vectordb, st.session_state.memory)

    st.success("✅ Documents processed! You can start chatting now.")


if st.sidebar.button("⚙️ Process Documents"):
    process_documents()


# Chat UI
st.subheader("💬 Ask Questions")

if st.session_state.qa_chain is None:
    st.info("Upload PDFs and click **Process Documents** to start.")
else:
    # Display previous messages
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(content)

    # Chat input
    user_query = st.chat_input("Ask a question about your documents...")
    if user_query:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Get answer from chain
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.qa_chain.invoke({"input": user_query, "chat_history": st.session_state.memory.chat_memory.messages})
                answer = result["answer"]
                sources = result.get("context", [])

                st.markdown(answer)

                # Show sources in expanders
                if sources:
                    with st.expander("📚 Sources used"):
                        for i, doc in enumerate(sources, start=1):
                            metadata = doc.metadata or {}
                            source = metadata.get("source", "Unknown source")
                            page = metadata.get("page", "Unknown page")
                            snippet = doc.page_content[:500].replace("\n", " ")

                            st.markdown(f"**[{i}] {source} — page {page}**")
                            st.markdown(f"> {snippet}...")
                            st.markdown("---")
                else:
                    st.markdown("_No sources returned._")

        # Save assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.memory.chat_memory.add_user_message(user_query)
        st.session_state.memory.chat_memory.add_ai_message(answer)
