"""
RAG (Retrieval-Augmented Generation) service using FAISS.

Handles:
- Embedding generation using SentenceTransformers
- FAISS index creation and persistence
- Vector similarity search
- Index caching for performance
"""
import logging
import os
from typing import List, Optional, Tuple

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from sentence_transformers import SentenceTransformer

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RAGServiceError(Exception):
    """Custom exception for RAG service errors."""
    pass


class RAGService:
    """
    Service for RAG operations using FAISS and SentenceTransformers.
    
    Provides:
    - Document chunk embedding and indexing
    - Vector similarity search
    - Safe index caching
    """

    def __init__(self) -> None:
        """
        Initialize RAG service with embedding model.
        
        Raises:
            RAGServiceError: If FAISS is not installed or model loading fails.
        """
        if faiss is None:
            raise RAGServiceError("FAISS is not installed. Install with: pip install faiss-cpu")

        self.model_name = settings.RAG_MODEL
        self.index_dir = settings.RAG_INDEX_DIR
        self.top_k = settings.RAG_TOP_K

        # Create index directory if it doesn't exist
        try:
            os.makedirs(self.index_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create index directory {self.index_dir}: {str(e)}")
            raise RAGServiceError(f"Failed to create index directory: {str(e)}")

        # Load embedding model
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise RAGServiceError(f"Failed to load embedding model: {str(e)}")

        # Cache for loaded indices
        self._index_cache: dict = {}

    def _get_index_path(self, document_id: int) -> str:
        """
        Get the safe file path for a document's FAISS index.
        
        Args:
            document_id: ID of the document.
            
        Returns:
            Safe file path for the index.
        """
        # Sanitize document_id to prevent path traversal
        if document_id <= 0:
            raise ValueError(f"Invalid document_id: {document_id}")
        return os.path.join(self.index_dir, f"doc_{document_id}.index")

    def build_index(self, document_id: int, chunks: List[str]) -> None:
        """
        Build and save a FAISS index for document chunks.
        
        Handles edge cases:
        - Empty chunk list
        - Single chunk
        - Large chunk lists

        Args:
            document_id: ID of the document (must be positive).
            chunks: List of text chunks to index (non-empty).

        Raises:
            RAGServiceError: If index building fails.
            ValueError: If document_id is invalid or chunks is empty.
        """
        if document_id <= 0:
            raise ValueError(f"Invalid document_id: {document_id}")
        
        if not chunks:
            logger.warning(f"No chunks provided for document {document_id}")
            return
        
        # Filter out empty chunks
        valid_chunks = [c for c in chunks if c and c.strip()]
        if not valid_chunks:
            logger.warning(f"All chunks are empty for document {document_id}")
            return

        try:
            # Encode chunks into embeddings
            logger.debug(f"Encoding {len(valid_chunks)} chunks for document {document_id}")
            embeddings = self.model.encode(valid_chunks, convert_to_numpy=True)
            embeddings = embeddings.astype(np.float32)

            # Validate embeddings
            if embeddings.shape[0] != len(valid_chunks):
                raise RAGServiceError("Embedding count mismatch")

            # Create FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)

            # Save index to disk
            index_path = self._get_index_path(document_id)
            faiss.write_index(index, index_path)

            # Cache the index
            self._index_cache[document_id] = index

            logger.info(f"Built and saved FAISS index for document {document_id} with {len(valid_chunks)} chunks")

        except Exception as e:
            logger.error(f"Failed to build index for document {document_id}: {str(e)}")
            raise RAGServiceError(f"Failed to build index: {str(e)}")

    def retrieve_top_k(
        self,
        document_id: int,
        query: str,
        k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """
        Retrieve top-k most relevant chunks for a query.
        
        Handles:
        - Missing indices (graceful fallback)
        - Empty queries
        - Index caching
        - Invalid document IDs

        Args:
            document_id: ID of the document to search (must be positive).
            query: Query text (must be non-empty).
            k: Number of top results to return. Defaults to settings.RAG_TOP_K.

        Returns:
            List of tuples (chunk_index, distance_score).
            Empty list if index not found or query is invalid.

        Raises:
            RAGServiceError: If retrieval fails unexpectedly.
            ValueError: If document_id is invalid or query is empty.
        """
        if document_id <= 0:
            raise ValueError(f"Invalid document_id: {document_id}")
        
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or whitespace")
        
        if k is None:
            k = self.top_k
        
        if k <= 0:
            raise ValueError("k must be positive")

        try:
            # Load or get cached index
            if document_id in self._index_cache:
                index = self._index_cache[document_id]
                logger.debug(f"Using cached index for document {document_id}")
            else:
                index_path = self._get_index_path(document_id)
                if not os.path.exists(index_path):
                    logger.warning(f"Index not found for document {document_id} at {index_path}")
                    return []

                try:
                    index = faiss.read_index(index_path)
                    self._index_cache[document_id] = index
                    logger.debug(f"Loaded index for document {document_id} from disk")
                except Exception as e:
                    logger.error(f"Failed to load index from {index_path}: {str(e)}")
                    return []

            # Encode query
            query_embedding = self.model.encode([query.strip()], convert_to_numpy=True)
            query_embedding = query_embedding.astype(np.float32)

            # Search (limit k to index size)
            index_size = index.ntotal
            search_k = min(k, index_size)
            
            if search_k <= 0:
                logger.warning(f"Index for document {document_id} is empty")
                return []

            distances, indices = index.search(query_embedding, search_k)

            # Return results as list of (chunk_index, distance)
            results = list(zip(indices[0].tolist(), distances[0].tolist()))
            logger.debug(f"Retrieved {len(results)} chunks for document {document_id}")
            return results

        except Exception as e:
            logger.error(f"Failed to retrieve chunks for document {document_id}: {str(e)}")
            raise RAGServiceError(f"Failed to retrieve chunks: {str(e)}")

    def clear_cache(self) -> None:
        """
        Clear the index cache.
        
        Useful for memory management or testing.
        """
        self._index_cache.clear()
        logger.info("Cleared RAG index cache")


# Singleton instance
_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """
    Get or create RAG service instance (singleton).
    
    Returns:
        RAGService: Singleton instance.
    """
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
