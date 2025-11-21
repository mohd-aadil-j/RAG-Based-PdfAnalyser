import os
import dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate  # Added for custom prompt

# 1. Config
dotenv.load_dotenv()  # Loads GROQ_API_KEY from .env if present

# If not using .env, uncomment and set key manually:
# os.environ["GROQ_API_KEY"] = "your-groq-api-key-here"

DATA_PATH = "data"          # folder with your PDFs
CHROMA_PATH = "chroma_db"   # folder for vector store


def load_and_split_documents():
    docs = []
    for filename in os.listdir(DATA_PATH):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(DATA_PATH, filename)
            print(f"[+] Loading {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            docs.extend(loader.load())

    print(f"[+] Loaded {len(docs)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)
    print(f"[+] Split into {len(chunks)} chunks")
    return chunks


def build_vector_store(chunks):
    # Real embedding model (local, no API cost)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )
    vectordb.persist()
    print("[+] Vector store built and persisted")
    return vectordb


def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )
    print("[+] Loaded existing vector store")
    return vectordb


def build_qa_chain(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    # Custom prompt to better condense questions with chat history
    condense_question_prompt = PromptTemplate.from_template(
        "Given the following conversation history and a follow-up question, "
        "rephrase the follow-up question to be a standalone question that includes "
        "relevant context from the history. Keep it concise and in the original language.\n\n"
        "Chat History:\n{chat_history}\n\n"
        "Follow-Up Question: {question}\n\n"
        "Standalone Question:"
    )

    # Custom answer prompt (for grounded, student-friendly answers)
    answer_prompt = PromptTemplate.from_template(
        """You are a helpful AI tutor for college students.

Use ONLY the following context from the documents to answer the question.
If the answer is not in the context, say:
"I don't know from the documents. You may need to refer to your textbook or teacher."

Be clear, concise, and explain in simple terms. Use bullet points or steps if helpful.

Context:
{context}

Question:
{question}

Answer:"""
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={"prompt": answer_prompt},
        return_source_documents=True,
        verbose=False,
    )
    return qa_chain



def ingest():
    chunks = load_and_split_documents()
    build_vector_store(chunks)


def chat():
    vectordb = load_vector_store()
    qa_chain = build_qa_chain(vectordb)

    print("=== Campus Knowledge Assistant (RAG) ===")
    print("Ask your question (type 'exit' to quit)")
    print("The AI will remember our conversation!\n")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break

        result = qa_chain.invoke({"question": query})
        answer = result["answer"]
        sources = result.get("source_documents", [])

        print("\nAI:", answer)
        print("\n--- Sources used ---")
        if not sources:
            print("No sources returned.")
        else:
            for i, doc in enumerate(sources, start=1):
                metadata = doc.metadata or {}
                source = metadata.get("source", "Unknown source")
                page = metadata.get("page", "Unknown page")

                snippet = doc.page_content[:300].replace("\n", " ")
                print(f"\n[{i}] {source} (page {page})")
                print(f"    \"{snippet}...\"")

        print("-" * 80)



if __name__ == "__main__":
    # First time: run ingest() once to build vector store
    if not os.path.exists(CHROMA_PATH) or not os.listdir(CHROMA_PATH):
        ingest()
    chat()