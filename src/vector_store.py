import os
from typing import List, Dict, Any, Optional, Tuple
import logging
import pickle
from pathlib import Path

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.base import VectorStore
from sentence_transformers import SentenceTransformer
import numpy as np

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Vector store manager implementing embedding and retrieval strategies.
    Supports multiple embedding models and vector databases.
    """
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 vector_store_type: str = "faiss",
                 persist_directory: str = "./vector_store"):
        """
        Initialize vector store manager.
        
        Args:
            embedding_model (str): Name of the embedding model to use
            vector_store_type (str): Type of vector store ('faiss', 'chroma')
            persist_directory (str): Directory to persist the vector store
        """
        self.embedding_model_name = embedding_model
        self.vector_store_type = vector_store_type
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        self.embeddings = self._initialize_embeddings()
        self.vector_store: Optional[VectorStore] = None
        
        # Retrieval parameters for optimization
        self.retrieval_params = {
            "k": 4,  # Number of documents to retrieve
            "search_type": "similarity",  # similarity, mmr, similarity_score_threshold
            "score_threshold": 0.7,  # For similarity_score_threshold search
            "fetch_k": 20,  # For MMR search
            "lambda_mult": 0.5  # For MMR search diversity
        }
    
    def _initialize_embeddings(self):
        """Initialize embedding model based on configuration."""
        try:
            if self.embedding_model_name.startswith("text-embedding"):
                # OpenAI embeddings
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    logger.warning("Chave API do OpenAI não encontrada. A usar embeddings HuggingFace.")
                    return self._get_huggingface_embeddings()
                return OpenAIEmbeddings(
                    model=self.embedding_model_name,
                    openai_api_key=api_key
                )
            else:
                # HuggingFace embeddings (default and fallback)
                return self._get_huggingface_embeddings()
                
        except Exception as e:
            logger.error(f"Erro ao inicializar embeddings: {e}")
            logger.info("A usar embeddings HuggingFace por defeito")
            return self._get_huggingface_embeddings()
    
    def _get_huggingface_embeddings(self):
        """Get HuggingFace embeddings with error handling."""
        try:
            return HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            logger.warning(f"Falha ao carregar {self.embedding_model_name}, a usar modelo por defeito")
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
    
    def create_vector_store(self, documents: List[Document]) -> VectorStore:
        """
        Create vector store from documents.
        
        Args:
            documents (List[Document]): List of documents to index
            
        Returns:
            VectorStore: Created vector store
        """
        if not documents:
            raise ValueError("Nenhum documento fornecido para criação do vector store")
        
        try:
            logger.info(f"A criar vector store com {len(documents)} documentos")
            
            if self.vector_store_type == "faiss":
                self.vector_store = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
            else:
                raise ValueError(f"Tipo de vector store não suportado: {self.vector_store_type}")
            
            logger.info("Vector store criado com sucesso")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Erro ao criar vector store: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add new documents to existing vector store.
        
        Args:
            documents (List[Document]): Documents to add
        """
        if not self.vector_store:
            raise ValueError("Vector store não inicializado. Cria primeiro o vector store.")
        
        try:
            self.vector_store.add_documents(documents)
            logger.info(f"Adicionados {len(documents)} documentos ao vector store")
        except Exception as e:
            logger.error(f"Erro ao adicionar documentos: {e}")
            raise
    
    def similarity_search(self, 
                         query: str, 
                         k: Optional[int] = None,
                         filter_metadata: Optional[Dict] = None) -> List[Document]:
        """
        Perform similarity search in vector store.
        
        Args:
            query (str): Query string
            k (int): Number of documents to retrieve
            filter_metadata (Dict): Metadata filters
            
        Returns:
            List[Document]: Retrieved documents
        """
        if not self.vector_store:
            raise ValueError("Vector store não inicializado")
        
        k = k or self.retrieval_params["k"]
        
        try:
            if filter_metadata:
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter=filter_metadata
                )
            else:
                results = self.vector_store.similarity_search(query=query, k=k)
            
            logger.info(f"Recuperados {len(results)} documentos para pergunta: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Erro na pesquisa por similaridade: {e}")
            raise
    
    def similarity_search_with_score(self, 
                                   query: str, 
                                   k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with relevance scores.
        
        Args:
            query (str): Query string
            k (int): Number of documents to retrieve
            
        Returns:
            List[Tuple[Document, float]]: Documents with similarity scores
        """
        if not self.vector_store:
            raise ValueError("Vector store não inicializado")
        
        k = k or self.retrieval_params["k"]
        
        try:
            results = self.vector_store.similarity_search_with_score(query=query, k=k)
            logger.info(f"Recuperados {len(results)} documentos com pontuações")
            return results
            
        except Exception as e:
            logger.error(f"Erro na pesquisa por similaridade com pontuação: {e}")
            raise
    
    def mmr_search(self, 
                   query: str, 
                   k: Optional[int] = None,
                   fetch_k: Optional[int] = None,
                   lambda_mult: Optional[float] = None) -> List[Document]:
        """
        Perform Maximum Marginal Relevance (MMR) search for diversity.
        
        Args:
            query (str): Query string
            k (int): Number of documents to retrieve
            fetch_k (int): Number of documents to fetch before MMR
            lambda_mult (float): Diversity parameter (0=diverse, 1=relevant)
            
        Returns:
            List[Document]: Retrieved documents with diversity
        """
        if not self.vector_store:
            raise ValueError("Vector store não inicializado")
        
        k = k or self.retrieval_params["k"]
        fetch_k = fetch_k or self.retrieval_params["fetch_k"]
        lambda_mult = lambda_mult or self.retrieval_params["lambda_mult"]
        
        try:
            results = self.vector_store.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult
            )
            logger.info(f"Recuperados {len(results)} documentos usando MMR")
            return results
            
        except Exception as e:
            logger.error(f"Erro na pesquisa MMR: {e}")
            raise
    
    def save_vector_store(self, name: str = "vector_store") -> str:
        """
        Save vector store to disk.
        
        Args:
            name (str): Name for the saved vector store
            
        Returns:
            str: Path where vector store was saved
        """
        if not self.vector_store:
            raise ValueError("Vector store não inicializado")
        
        save_path = self.persist_directory / name
        
        try:
            if self.vector_store_type == "faiss":
                self.vector_store.save_local(str(save_path))
            
            # Save metadata
            metadata = {
                "embedding_model": self.embedding_model_name,
                "vector_store_type": self.vector_store_type,
                "retrieval_params": self.retrieval_params
            }
            
            with open(save_path / "metadata.pkl", "wb") as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Vector store guardado em {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Erro ao guardar vector store: {e}")
            raise
    
    def load_vector_store(self, name: str = "vector_store") -> VectorStore:
        """
        Load vector store from disk.
        
        Args:
            name (str): Name of the saved vector store
            
        Returns:
            VectorStore: Loaded vector store
        """
        load_path = self.persist_directory / name
        
        if not load_path.exists():
            raise FileNotFoundError(f"Vector store não encontrado em {load_path}")
        
        try:
            # Load metadata
            metadata_path = load_path / "metadata.pkl"
            if metadata_path.exists():
                with open(metadata_path, "rb") as f:
                    metadata = pickle.load(f)
                    self.retrieval_params.update(metadata.get("retrieval_params", {}))
            
            if self.vector_store_type == "faiss":
                self.vector_store = FAISS.load_local(
                    str(load_path),
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
            
            logger.info(f"Vector store carregado de {load_path}")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Erro ao carregar vector store: {e}")
            raise
    
    def get_retriever(self, search_type: str = "similarity", **kwargs):
        """
        Get a retriever object for the vector store.
        
        Args:
            search_type (str): Type of search ('similarity', 'mmr', 'similarity_score_threshold')
            **kwargs: Additional parameters for the retriever
            
        Returns:
            VectorStoreRetriever: Configured retriever
        """
        if not self.vector_store:
            raise ValueError("Vector store não inicializado")
        
        # Merge default params with provided kwargs
        search_kwargs = self.retrieval_params.copy()
        search_kwargs.update(kwargs)
        
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
    def update_retrieval_params(self, **params):
        """
        Update retrieval parameters.
        
        Args:
            **params: Parameters to update
        """
        self.retrieval_params.update(params)
        logger.info(f"Parâmetros de recuperação atualizados: {self.retrieval_params}")
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dict[str, Any]: Vector store statistics
        """
        if not self.vector_store:
            return {"status": "not_initialized"}
        
        stats = {
            "embedding_model": self.embedding_model_name,
            "vector_store_type": self.vector_store_type,
            "retrieval_params": self.retrieval_params,
            "persist_directory": str(self.persist_directory)
        }
        
        # Add FAISS-specific stats
        if hasattr(self.vector_store, 'index') and self.vector_store.index:
            stats.update({
                "total_vectors": self.vector_store.index.ntotal,
                "vector_dimension": self.vector_store.index.d
            })
        
        return stats
