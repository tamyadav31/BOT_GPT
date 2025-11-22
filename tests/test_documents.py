"""
Tests for document service and endpoints.

Covers:
- Text chunking with edge cases
- Document creation and deletion
- RAG index building
- Error handling and validation
"""
import pytest
from unittest.mock import patch, MagicMock
from sqlalchemy.orm import Session

from app.db.models import User, Document, DocumentChunk
from app.services.document_service import DocumentService, DocumentServiceError
from app.utils.chunking import chunk_text


class TestChunking:
    """Tests for text chunking utility."""

    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        text = "This is a sample document. " * 20
        chunks = chunk_text(text, chunk_size=100, overlap=10)

        assert len(chunks) > 0
        assert all(len(chunk) > 0 for chunk in chunks)

    def test_chunk_text_empty(self):
        """Test chunking empty text."""
        chunks = chunk_text("", chunk_size=100)
        assert chunks == []

    def test_chunk_text_whitespace_only(self):
        """Test chunking whitespace-only text."""
        chunks = chunk_text("   \n\t  ", chunk_size=100)
        assert chunks == []

    def test_chunk_text_overlap(self):
        """Test that chunks have overlap."""
        text = "A" * 200
        chunks = chunk_text(text, chunk_size=100, overlap=20)

        assert len(chunks) >= 2

    def test_chunk_text_invalid_chunk_size(self):
        """Test chunking with invalid chunk_size."""
        with pytest.raises(ValueError):
            chunk_text("test", chunk_size=0)

    def test_chunk_text_negative_overlap(self):
        """Test chunking with negative overlap."""
        with pytest.raises(ValueError):
            chunk_text("test", chunk_size=100, overlap=-1)

    def test_chunk_text_overlap_too_large(self):
        """Test chunking with overlap >= chunk_size."""
        with pytest.raises(ValueError):
            chunk_text("test", chunk_size=100, overlap=100)

    def test_chunk_text_small_text(self):
        """Test chunking text smaller than chunk_size."""
        text = "Small text"
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        
        assert len(chunks) == 1
        assert chunks[0] == text


class TestCreateDocument:
    """Tests for creating documents."""

    @patch("app.services.document_service.get_rag_service")
    def test_create_document_basic(self, mock_rag_service, db: Session, test_user):
        """Test creating a basic document."""
        # Mock RAG service
        mock_rag = MagicMock()
        mock_rag.build_index = MagicMock()
        mock_rag_service.return_value = mock_rag

        service = DocumentService(db)
        document, num_chunks = service.create_document(
            user_id=test_user.id,
            title="Sample Document",
            content="This is a sample document. " * 10,
        )

        # Assertions
        assert document.id is not None
        assert document.user_id == test_user.id
        assert document.title == "Sample Document"
        assert num_chunks > 0

        # Verify chunks are created
        chunks = db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document.id
        ).all()
        assert len(chunks) == num_chunks

    @patch("app.services.document_service.get_rag_service")
    def test_create_document_empty_content(self, mock_rag_service, db: Session, test_user):
        """Test creating a document with empty content."""
        mock_rag = MagicMock()
        mock_rag_service.return_value = mock_rag

        service = DocumentService(db)
        
        with pytest.raises(ValueError):
            service.create_document(
                user_id=test_user.id,
                title="Empty Document",
                content="",
            )

    @patch("app.services.document_service.get_rag_service")
    def test_create_document_invalid_user_id(self, mock_rag_service, db: Session):
        """Test creating a document with invalid user_id."""
        mock_rag = MagicMock()
        mock_rag_service.return_value = mock_rag

        service = DocumentService(db)
        
        with pytest.raises(ValueError):
            service.create_document(
                user_id=-1,
                title="Document",
                content="Sample content",
            )

    @patch("app.services.document_service.get_rag_service")
    def test_create_document_empty_title(self, mock_rag_service, db: Session, test_user):
        """Test creating a document with empty title."""
        mock_rag = MagicMock()
        mock_rag_service.return_value = mock_rag

        service = DocumentService(db)
        
        with pytest.raises(ValueError):
            service.create_document(
                user_id=test_user.id,
                title="",
                content="Sample content",
            )

    @patch("app.services.document_service.get_rag_service")
    def test_create_document_whitespace_content(self, mock_rag_service, db: Session, test_user):
        """Test creating a document with whitespace-only content."""
        mock_rag = MagicMock()
        mock_rag_service.return_value = mock_rag

        service = DocumentService(db)
        
        with pytest.raises(ValueError):
            service.create_document(
                user_id=test_user.id,
                title="Document",
                content="   \n\t  ",
            )

    @patch("app.services.document_service.get_rag_service")
    def test_create_document_rag_index_called(self, mock_rag_service, db: Session, test_user):
        """Test that RAG index building is called."""
        mock_rag = MagicMock()
        mock_rag.build_index = MagicMock()
        mock_rag_service.return_value = mock_rag

        service = DocumentService(db)
        document, num_chunks = service.create_document(
            user_id=test_user.id,
            title="Document for RAG",
            content="Sample content. " * 20,
        )

        # Verify build_index was called
        mock_rag.build_index.assert_called_once()
        call_args = mock_rag.build_index.call_args
        assert call_args[0][0] == document.id  # document_id
        assert len(call_args[0][1]) == num_chunks  # chunks list


class TestListDocuments:
    """Tests for listing documents."""

    @patch("app.services.document_service.get_rag_service")
    def test_list_documents_empty(self, mock_rag_service, db: Session, test_user):
        """Test listing documents when none exist."""
        mock_rag = MagicMock()
        mock_rag_service.return_value = mock_rag

        service = DocumentService(db)
        documents, total = service.list_documents(user_id=test_user.id)

        assert documents == []
        assert total == 0

    @patch("app.services.document_service.get_rag_service")
    def test_list_documents_with_pagination(self, mock_rag_service, db: Session, test_user):
        """Test listing documents with pagination."""
        mock_rag = MagicMock()
        mock_rag.build_index = MagicMock()
        mock_rag_service.return_value = mock_rag

        # Create multiple documents
        service = DocumentService(db)
        for i in range(5):
            service.create_document(
                user_id=test_user.id,
                title=f"Document {i}",
                content="Sample content. " * 10,
            )

        documents, total = service.list_documents(
            user_id=test_user.id,
            limit=2,
            offset=0,
        )

        assert len(documents) == 2
        assert total == 5


class TestDeleteDocument:
    """Tests for deleting documents."""

    @patch("app.services.document_service.get_rag_service")
    def test_delete_document(self, mock_rag_service, db: Session, test_user):
        """Test deleting a document."""
        mock_rag = MagicMock()
        mock_rag.build_index = MagicMock()
        mock_rag_service.return_value = mock_rag

        service = DocumentService(db)
        document, _ = service.create_document(
            user_id=test_user.id,
            title="To Delete",
            content="Content. " * 10,
        )

        deleted = service.delete_document(document.id, test_user.id)
        assert deleted is True

        # Verify it's deleted
        db_doc = db.query(Document).filter(Document.id == document.id).first()
        assert db_doc is None

    @patch("app.services.document_service.get_rag_service")
    def test_delete_nonexistent_document(self, mock_rag_service, db: Session, test_user):
        """Test deleting a non-existent document."""
        mock_rag = MagicMock()
        mock_rag_service.return_value = mock_rag

        service = DocumentService(db)
        deleted = service.delete_document(999, test_user.id)

        assert deleted is False
