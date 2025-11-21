# 📚 Campus Knowledge Assistant

A powerful **Retrieval-Augmented Generation (RAG)** system built with LangChain, Groq, and Streamlit that allows you to chat with your PDF documents using advanced AI. Perfect for students, researchers, and professionals who want to extract insights from their documents through natural conversation.

## ✨ Features

### 🤖 AI-Powered Q&A
- **Conversational Memory**: Remembers previous questions and answers for contextual follow-ups
- **Smart Question Rephrasing**: Automatically understands references like "explain its types" from previous context
- **Source Citations**: Shows exactly which documents and pages were used for each answer

### 📄 Document Processing
- **Multi-PDF Support**: Upload and process multiple PDF documents simultaneously
- **Intelligent Chunking**: Splits documents into optimal 1000-character chunks with 200-character overlap
- **Metadata Preservation**: Maintains source filenames and page numbers for accurate citations

### 🔍 Advanced Search
- **Semantic Search**: Uses vector embeddings for meaning-based retrieval (not just keyword matching)
- **Local Embeddings**: Privacy-focused - no data sent to external embedding services
- **Configurable Retrieval**: Adjustable number of relevant chunks (k=4 by default)

### 🎨 User Interfaces
- **Web App (Streamlit)**: Modern, intuitive web interface with drag-and-drop PDF uploads
- **Command-Line Tool**: Lightweight terminal interface for quick queries
- **Real-time Chat**: Interactive conversation with instant responses

### 🛠️ Technical Stack
- **LLM**: Groq's Llama 3.1 8B Instant (fast, cost-effective inference)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2) for semantic similarity
- **Vector Store**: ChromaDB for efficient vector storage and retrieval
- **Framework**: LangChain for orchestration, Streamlit for web UI

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Groq API key ([get one here](https://console.groq.com/))

### Installation

1. **Clone or download the project**
   ```bash
   cd your-project-directory
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API key**
   Create a `.env` file in the project root:
   ```bash
   GROQ_API_KEY=your-groq-api-key-here
   ```

## 🎯 Usage Options

### Option 1: Web Interface (Recommended)

Launch the modern Streamlit web application:

```bash
streamlit run app.py
```

**Features:**
- Drag-and-drop PDF uploads
- Real-time chat interface
- Source document citations
- Session-based conversation memory
- No installation required for end users

**Workflow:**
1. Open the web app in your browser
2. Upload your PDF documents
3. Click "Process Documents"
4. Start asking questions!

### Option 2: Command-Line Interface

For quick testing or integration:

```bash
python rag.py
```

**Features:**
- Terminal-based interaction
- Persistent vector store
- Batch document processing

**Setup:**
- Place PDFs in the `data/` folder
- Run `python rag.py` (first run processes documents)
- Subsequent runs reuse the vector store

## 📁 Project Structure

```
Simple-RAG/
├── app.py                 # 🌐 Streamlit web application
├── rag.py                 # 💻 Command-line interface
├── requirements.txt       # 📦 Python dependencies
├── .env                   # 🔑 API keys (create this)
├── .env.example          # 📝 API key template
├── README.md             # 📖 This file
├── .gitignore           # 🚫 Git ignore rules
├── data/                 # 📄 PDF storage (for CLI)
│   └── README.md
└── chroma_db/            # 🗄️ Vector store (auto-generated)
```

## 🏗️ How It Works

### Architecture Overview

```
User Query → Question Rephrasing → Semantic Search → Context Retrieval → LLM Generation → Answer
     ↓              ↓                      ↓              ↓              ↓              ↓
  "its types" → "Multiclass Classification types" → Vector Search → Top 4 Chunks → Groq Llama → Final Answer
```

### Detailed Flow

1. **Document Ingestion**
   - PDFs are loaded and split into overlapping text chunks
   - Each chunk is converted to a vector embedding using Sentence Transformers
   - Embeddings are stored in ChromaDB for fast retrieval

2. **Query Processing**
   - User question is rephrased using conversation history for context
   - Query is converted to embedding and compared against document chunks
   - Top-k most similar chunks are retrieved as context

3. **Answer Generation**
   - Context chunks + rephrased question sent to Groq's Llama model
   - Model generates answer based only on provided context
   - Source documents are cited for transparency

4. **Memory Management**
   - Conversation history maintained across interactions
   - Follow-up questions automatically incorporate previous context

## ⚙️ Configuration

### Model Settings (in `app.py` or `rag.py`)

```python
# LLM Configuration
llm = ChatGroq(
    model="llama-3.1-8b-instant",  # Fast and cost-effective
    temperature=0.2,               # Low creativity for factual answers
)

# Retrieval Settings
retriever = vectordb.as_retriever(search_kwargs={"k": 4})  # Top 4 chunks

# Text Splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,     # Characters per chunk
    chunk_overlap=200,   # Overlap between chunks
)
```

### Available Models

- `llama-3.1-8b-instant` ⭐ (Recommended - fast and accurate)
- `llama3-8b-8192` (Legacy, may be deprecated)
- `llama3-70b-8192` (More capable but slower)
- `mixtral-8x7b-32768` (Good for complex reasoning)

## 🔧 Customization

### Adding New Features

**Custom Prompts:**
```python
custom_prompt = PromptTemplate.from_template("""
You are a {role} assistant. Use the context to answer...

Context: {context}
Question: {question}
Answer:
""")
```

**Different Embeddings:**
```python
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()  # Requires OpenAI API key
```

**External Vector Stores:**
```python
from langchain_pinecone import PineconeVectorStore
# For cloud-based vector storage
```

## 🐛 Troubleshooting

### Common Issues

**❌ "ModuleNotFoundError"**
```bash
pip install -r requirements.txt
# Or install specific package: pip install langchain-groq
```

**❌ "GROQ_API_KEY not found"**
- Create `.env` file with: `GROQ_API_KEY=your-key-here`
- Or set environment variable: `export GROQ_API_KEY=your-key`

**❌ "No text could be loaded from PDFs"**
- Ensure PDFs are not password-protected
- Check if PDFs contain selectable text (not just images)
- Try different PDF files

**❌ "HuggingFace embeddings error"**
- The app uses `FakeEmbeddings` to avoid dependency issues
- For production, consider using OpenAI embeddings or fixing HF dependencies

**❌ Streamlit app not loading**
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

**❌ Memory not working in conversations**
- Check if `ConversationBufferMemory` is properly initialized
- Ensure `output_key="answer"` matches your chain's output

### Performance Tips

- **Smaller chunks**: Reduce `chunk_size` for more precise retrieval
- **More context**: Increase `k` parameter for broader context
- **Caching**: Vector stores persist automatically for faster subsequent runs
- **Model selection**: Use `llama-3.1-8b-instant` for speed, larger models for complexity

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with a clear description

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest

# Format code
black . && isort .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain** - For the RAG framework
- **Groq** - For fast LLM inference
- **Sentence Transformers** - For semantic embeddings
- **ChromaDB** - For vector storage
- **Streamlit** - For the web interface

## 📞 Support

- 📧 **Issues**: Open a GitHub issue for bugs or feature requests
- 💬 **Discussions**: Use GitHub Discussions for questions
- 📖 **Documentation**: Check this README and inline code comments

---

**Happy Learning! 🎓** Transform your PDFs into interactive knowledge bases with AI-powered conversations.
