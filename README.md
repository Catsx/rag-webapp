# RAG Application - Catarina Cardoso

A simple RAG (Retrieval-Augmented Generation) application that lets you chat with your documents.

## What it does

- Upload PDF, Word, or text files
- Add web links
- Ask questions about your content
- Get AI-powered answers with sources

## How to run

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set your OpenAI API key:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

3. **Run the web interface:**
```bash
python web_interface.py
```

4. **Or use the command line:**
```bash
python src/cli.py interactive
```

## Features

- **Web Interface**: Easy-to-use web app with Gradio
- **Command Line**: Full CLI for advanced users
- **Multiple Formats**: PDF, DOCX, TXT, Markdown, URLs
- **Smart Processing**: Automatic text cleaning and chunking
- **Vector Search**: Semantic search using embeddings
- **AI Answers**: GPT-powered responses with source citations
- **Works Offline**: Similarity search works without OpenAI key

## Project Structure

```
src/
├── rag_pipeline.py     # Main RAG implementation
├── document_loader.py  # Load files and URLs
├── text_processor.py   # Clean and chunk text
├── vector_store.py     # Vector database with embeddings
└── cli.py             # Command line interface
web_interface.py        # Web app interface
requirements.txt        # Python dependencies
```

## Quick Example

```python
from src.rag_pipeline import RAGPipeline

# Initialize
rag = RAGPipeline()

# Add documents
rag.index_documents(["document.pdf", "https://example.com"])

# Ask questions
response = rag.query("What is the main topic?")
print(response['answer'])
```

## Requirements

- Python 3.8+
- OpenAI API key (optional, for AI answers)
- See `requirements.txt` for all dependencies

---

**Author:** Catarina Cardoso