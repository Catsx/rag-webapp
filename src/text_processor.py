import re
import string
from typing import List, Dict, Any, Optional
import logging

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter
)
from langchain.schema import Document

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Text processor implementing data cleaning and chunking strategies.
    Optimized for RAG performance.
    """
    
    def __init__(self, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200,
                 chunking_strategy: str = "recursive"):
        """
        Initialize text processor with chunking parameters.
        
        Args:
            chunk_size (int): Size of each chunk in characters
            chunk_overlap (int): Overlap between chunks to preserve context
            chunking_strategy (str): Strategy for chunking ('recursive', 'character', 'token', 'markdown', 'html')
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        
        # Initialize text splitters with optimal configurations
        self.splitters = {
            "recursive": RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            ),
            "character": CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator="\n"
            ),
            "token": TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            ),
            "markdown": MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3"),
                ]
            ),
            "html": HTMLHeaderTextSplitter(
                headers_to_split_on=[
                    ("h1", "Header 1"),
                    ("h2", "Header 2"),
                    ("h3", "Header 3"),
                ]
            )
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean text following data cleaning best practices.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        # Step 1: Handle encoding issues and special characters
        text = self._fix_encoding(text)
        
        # Step 2: Remove excessive whitespace and normalize
        text = self._normalize_whitespace(text)
        
        # Step 3: Clean formatting artifacts
        text = self._remove_formatting_artifacts(text)
        
        # Step 4: Standardize terminology
        text = self._standardize_terminology(text)
        
        # Step 5: Remove duplicates and redundant content
        text = self._remove_redundancy(text)
        
        logger.info("Limpeza de texto concluída")
        return text
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents using the specified strategy.
        
        Args:
            documents (List[Document]): List of documents to chunk
            
        Returns:
            List[Document]: List of chunked documents
        """
        if self.chunking_strategy not in self.splitters:
            raise ValueError(f"Estratégia de chunking não suportada: {self.chunking_strategy}")
        
        splitter = self.splitters[self.chunking_strategy]
        
        all_chunks = []
        for doc in documents:
            # Clean the document content first
            cleaned_content = self.clean_text(doc.page_content)
            
            # Create a new document with cleaned content
            cleaned_doc = Document(
                page_content=cleaned_content,
                metadata=doc.metadata
            )
            
            # Split the document
            if self.chunking_strategy in ["markdown", "html"]:
                # These splitters work differently
                chunks = splitter.split_text(cleaned_content)
                chunk_docs = [Document(page_content=chunk, metadata=doc.metadata) for chunk in chunks]
            else:
                chunk_docs = splitter.split_documents([cleaned_doc])
            
            # Add chunk metadata
            for i, chunk_doc in enumerate(chunk_docs):
                chunk_doc.metadata.update({
                    "chunk_id": i,
                    "total_chunks": len(chunk_docs),
                    "chunk_size": len(chunk_doc.page_content),
                    "chunking_strategy": self.chunking_strategy
                })
            
            all_chunks.extend(chunk_docs)
        
        logger.info(f"Criados {len(all_chunks)} chunks de {len(documents)} documentos usando estratégia {self.chunking_strategy}")
        return all_chunks
    
    def _fix_encoding(self, text: str) -> str:
        """Fix common encoding issues."""
        # Handle common encoding problems
        encoding_fixes = {
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€"': '—',
            'â€"': '–',
            'Ã¡': 'á',
            'Ã©': 'é',
            'Ã­': 'í',
            'Ã³': 'ó',
            'Ãº': 'ú'
        }
        
        for bad_char, good_char in encoding_fixes.items():
            text = text.replace(bad_char, good_char)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace and line breaks."""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _remove_formatting_artifacts(self, text: str) -> str:
        """Remove common formatting artifacts."""
        # Remove page numbers
        text = re.sub(r'Page \d+', '', text)
        text = re.sub(r'\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove headers/footers patterns
        text = re.sub(r'^[A-Z\s]+$', '', text, flags=re.MULTILINE)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{3,}', '---', text)
        
        # Remove URLs if they're not relevant
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        return text
    
    def _standardize_terminology(self, text: str) -> str:
        """
        Standardize terminology for consistency.
        Unify variations like "ML," "Machine Learning," and "machine learning".
        """
        # AI/ML terminology standardization
        ml_variations = [
            (r'\bML\b', 'Machine Learning'),
            (r'\bmachine learning\b', 'Machine Learning'),
            (r'\bAI\b', 'Artificial Intelligence'),
            (r'\bartificial intelligence\b', 'Artificial Intelligence'),
            (r'\bNLP\b', 'Natural Language Processing'),
            (r'\bnatural language processing\b', 'Natural Language Processing'),
            (r'\bLLM\b', 'Large Language Model'),
            (r'\blarge language model\b', 'Large Language Model'),
        ]
        
        for pattern, replacement in ml_variations:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _remove_redundancy(self, text: str) -> str:
        """Remove redundant and duplicate content."""
        # Split into sentences
        sentences = text.split('.')
        
        # Remove duplicate sentences (simple approach)
        unique_sentences = []
        seen = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence.lower() not in seen and len(sentence) > 10:
                unique_sentences.append(sentence)
                seen.add(sentence.lower())
        
        return '. '.join(unique_sentences)
    
    def get_chunk_statistics(self, chunks: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about the chunks for analysis and optimization.
        
        Args:
            chunks (List[Document]): List of chunked documents
            
        Returns:
            Dict[str, Any]: Statistics about the chunks
        """
        if not chunks:
            return {}
        
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        
        stats = {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "total_characters": sum(chunk_sizes),
            "chunking_strategy": self.chunking_strategy,
            "chunk_size_setting": self.chunk_size,
            "chunk_overlap_setting": self.chunk_overlap
        }
        
        return stats
    
    def update_chunking_parameters(self, chunk_size: int, chunk_overlap: int, strategy: str):
        """
        Update chunking parameters and reinitialize splitters.
        
        Args:
            chunk_size (int): New chunk size
            chunk_overlap (int): New chunk overlap
            strategy (str): New chunking strategy
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = strategy
        
        # Reinitialize splitters with new parameters
        self.splitters["recursive"] = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        self.splitters["character"] = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n"
        )
        self.splitters["token"] = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        logger.info(f"Parâmetros de chunking atualizados: tamanho={chunk_size}, sobreposição={chunk_overlap}, estratégia={strategy}")

