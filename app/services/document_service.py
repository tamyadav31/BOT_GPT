"""
Document service for managing documents and RAG indexing.

Handles:
- Document creation and deletion
- Text chunking
- RAG index building
- Document ownership validation
"""
import logging
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from app.db.models import Document, DocumentChunk
from app.utils.chunking import chunk_text
from app.services.rag_service import get_rag_service, RAGServiceError
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class DocumentServiceError(Exception):
    """Custom exception for document service errors."""
    pass


class DocumentService:
    """
    Service for managing documents and RAG indexing.
    
    Handles:
    - Document CRUD operations
    - Text chunking and persistence
    - RAG index building
    - Document ownership validation
    """

    def __init__(self, db: Session) -> None:
        """
        Initialize document service.
        
        Args:
            db: SQLAlchemy database session.
        """
        self.db = db
        self.rag_service = get_rag_service()

    def create_document(
        self,
        user_id: int,
        title: str,
        content: str,
    ) -> Tuple[Document, int]:
        """
        Create a new document and build its RAG index.
        
        Handles:
        - Document creation
        - Text chunking
        - Chunk persistence
        - RAG index building (with graceful fallback)

        Args:
            user_id: ID of the user (must be positive).
            title: Document title (non-empty).
            content: Document content (raw text, non-empty).

        Returns:
            Tuple of (Document object, number of chunks).

        Raises:
            DocumentServiceError: If document creation fails.
            ValueError: If parameters are invalid.
        """
        # Validate parameters
        if user_id <= 0:
            raise ValueError(f"Invalid user_id: {user_id}")
        if not title or not title.strip():
            raise ValueError("title cannot be empty")
        if not content or not content.strip():
            raise ValueError("content cannot be empty")
        
        try:
            # Create document (no content field in Document model)
            document = Document(
                user_id=user_id,
                title=title.strip(),
                path=None  # We're not storing files, just text content
            )
            self.db.add(document)
            self.db.flush()  # Get the ID

            # Chunk the content
            try:
                chunks = chunk_text(
                    content.strip(),
                    chunk_size=settings.CHUNK_SIZE,
                    overlap=settings.CHUNK_OVERLAP,
                )
            except ValueError as e:
                logger.error(f"Chunking error: {str(e)}")
                raise DocumentServiceError(f"Failed to chunk document: {str(e)}")

            if not chunks:
                logger.warning(f"No chunks created for document {document.id}")
                self.db.commit()
                logger.info(f"Created document {document.id} with 0 chunks (content too small)")
                return document, 0

            # Create DocumentChunk entries
            for idx, chunk_content in enumerate(chunks):
                doc_chunk = DocumentChunk(
                    document_id=document.id,
                    chunk_index=idx,
                    text=chunk_content,
                )
                self.db.add(doc_chunk)

            self.db.flush()

            # Build RAG index (graceful fallback if fails)
            try:
                self.rag_service.build_index(document.id, chunks)
                logger.info(f"Built RAG index for document {document.id}")
            except RAGServiceError as e:
                logger.warning(f"Failed to build RAG index for document {document.id}: {str(e)}")
                # Continue anyway - chunks are stored in DB
                # RAG retrieval will fail but document is still usable

            self.db.commit()
            logger.info(f"Created document {document.id} with {len(chunks)} chunks")

            return document, len(chunks)

        except (ValueError, DocumentServiceError):
            self.db.rollback()
            raise
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to create document: {str(e)}", exc_info=True)
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception args: {e.args}")
            raise DocumentServiceError(f"Failed to create document: {str(e)}")

    def get_document(self, document_id: int, user_id: int) -> Optional[Document]:
        """
        Get a document by ID.

        Args:
            document_id: ID of the document.
            user_id: ID of the user (for validation).

        Returns:
            Document object or None.
        """
        return self.db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == user_id,
        ).first()

    def list_documents(
        self,
        user_id: int,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[List[Document], int]:
        """
        List documents for a user.

        Args:
            user_id: ID of the user.
            limit: Number of results to return.
            offset: Number of results to skip.

        Returns:
            Tuple of (list of Document objects, total count).
        """
        query = self.db.query(Document).filter(
            Document.user_id == user_id,
        ).order_by(Document.created_at.desc())

        total = query.count()
        documents = query.limit(limit).offset(offset).all()

        return documents, total

    def delete_document(self, document_id: int, user_id: int) -> bool:
        """
        Delete a document.

        Args:
            document_id: ID of the document.
            user_id: ID of the user (for validation).

        Returns:
            True if deleted, False if not found.
        """
        try:
            document = self.db.query(Document).filter(
                Document.id == document_id,
                Document.user_id == user_id,
            ).first()

            if not document:
                return False

            self.db.delete(document)
            self.db.commit()
            logger.info(f"Deleted document {document_id}")
            return True

        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to delete document: {str(e)}")
            raise DocumentServiceError(f"Failed to delete document: {str(e)}")
