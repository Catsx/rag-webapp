import os
import requests
from typing import List, Optional
from urllib.parse import urlparse
import logging

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    WebBaseLoader
)
from langchain.schema import Document
from bs4 import BeautifulSoup

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentLoader:
    
    def __init__(self):
        self.supported_extensions = {'.pdf', '.docx', '.txt', '.md'}
        
    def load_from_file(self, file_path: str) -> List[Document]:
        """
        Carrega documentos de ficheiros locais.
        
        Args:
            file_path (str): Caminho para o ficheiro
            
        Returns:
            List[Document]: Lista de documentos carregados
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Ficheiro não encontrado: {file_path}")
            
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension not in self.supported_extensions:
            raise ValueError(f"Extensão de ficheiro não suportada: {file_extension}")
            
        try:
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension == '.docx':
                loader = Docx2txtLoader(file_path)
            elif file_extension in ['.txt', '.md']:
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                raise ValueError(f"Nenhum carregador disponível para {file_extension}")
                
            documents = loader.load()
            logger.info(f"Carregados com sucesso {len(documents)} documentos de {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Erro ao carregar ficheiro {file_path}: {str(e)}")
            raise
    
    def load_from_url(self, url: str) -> List[Document]:
        """
        Carrega documentos de URLs da web.
        
        Args:
            url (str): URL para carregar conteúdo
            
        Returns:
            List[Document]: Lista de documentos carregados
        """
        if not self._is_valid_url(url):
            raise ValueError(f"URL inválido: {url}")
            
        try:
            # Usa o WebBaseLoader do LangChain para conteúdo web
            loader = WebBaseLoader(url)
            documents = loader.load()
            
            # Limpeza adicional para conteúdo web
            for doc in documents:
                doc.page_content = self._clean_web_content(doc.page_content)
                
            logger.info(f"Carregados com sucesso {len(documents)} documentos de {url}")
            return documents
            
        except Exception as e:
            logger.error(f"Erro ao carregar URL {url}: {str(e)}")
            raise
    
    def load_from_multiple_sources(self, sources: List[str]) -> List[Document]:
        """
        Carrega documentos de múltiplas fontes (ficheiros e URLs).
        
        Args:
            sources (List[str]): Lista de caminhos de ficheiros e URLs
            
        Returns:
            List[Document]: Lista combinada de todos os documentos carregados
        """
        all_documents = []
        
        for source in sources:
            try:
                if self._is_valid_url(source):
                    docs = self.load_from_url(source)
                else:
                    docs = self.load_from_file(source)
                all_documents.extend(docs)
            except Exception as e:
                logger.warning(f"Falha ao carregar fonte {source}: {str(e)}")
                continue
                
        logger.info(f"Carregados com sucesso {len(all_documents)} documentos no total de {len(sources)} fontes")
        return all_documents
    
    def _is_valid_url(self, url: str) -> bool:
        """Verifica se a string é um URL válido."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _clean_web_content(self, content: str) -> str:
        """
        Limpa conteúdo web seguindo as diretrizes de limpeza de dados.
        """
        # Remove espaços em branco excessivos
        content = ' '.join(content.split())
        
        # Remove artefactos web comuns
        artifacts_to_remove = [
            'Cookie Policy', 'Privacy Policy', 'Terms of Service',
            'Subscribe to newsletter', 'Follow us on', 'Share this',
            'Advertisement', 'Sponsored content'
        ]
        
        for artifact in artifacts_to_remove:
            content = content.replace(artifact, '')
        
        return content.strip()
    
    def get_supported_extensions(self) -> set:
        """Retorna as extensões de ficheiro suportadas."""
        return self.supported_extensions.copy()

