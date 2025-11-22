"""
Document API endpoints.

Provides REST endpoints for:
- Uploading/registering documents
- Listing documents with pagination
- Deleting documents
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.schemas import (
    DocumentCreate,
    DocumentRead,
    DocumentUploadResponse,
    PaginationMeta,
    StatusResponse,
)
from app.services.document_service import (
    DocumentService,
    DocumentServiceError,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document",
    responses={
        400: {"description": "Invalid request parameters"},
        500: {"description": "Internal server error"},
    },
)
async def upload_document(
    payload: DocumentCreate,
    db: Session = Depends(get_db),
) -> DocumentUploadResponse:
    """
    Upload/register a new document for RAG.
    
    Chunks the document and builds a FAISS index for vector search.
    Gracefully handles RAG index failures (document still usable).

    Args:
        payload: Document creation payload with:
            - user_id: ID of the user
            - title: Document title
            - content: Raw text content
        db: Database session (injected).

    Returns:
        DocumentUploadResponse with document ID and chunk count.

    Raises:
        HTTPException 400: If parameters are invalid or service error.
        HTTPException 500: If unexpected error occurs.
    """
    try:
        service = DocumentService(db)
        document, num_chunks = service.create_document(
            user_id=payload.user_id,
            title=payload.title,
            content=payload.content,
        )

        logger.info(f"Uploaded document {document.id} with {num_chunks} chunks")
        return DocumentUploadResponse(
            document_id=document.id,
            num_chunks=num_chunks,
        )

    except ValueError as e:
        logger.warning(f"Invalid request: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except DocumentServiceError as e:
        logger.error(f"Document service error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in upload_document: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload document"
        )


@router.get(
    "",
    summary="List documents for user",
    responses={
        500: {"description": "Internal server error"},
    },
)
async def list_documents(
    user_id: int = Query(..., gt=0, description="ID of the user"),
    limit: int = Query(20, ge=1, le=100, description="Number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    db: Session = Depends(get_db),
) -> dict:
    """
    List all documents for a user with pagination.

    Args:
        user_id: ID of the user (query parameter, must be positive).
        limit: Number of results to return (1-100, default 20).
        offset: Number of results to skip (default 0).
        db: Database session (injected).

    Returns:
        Dict with:
            - documents: List of DocumentRead with chunk counts
            - pagination: PaginationMeta with total, limit, offset
    """
    try:
        service = DocumentService(db)
        documents, total = service.list_documents(
            user_id=user_id,
            limit=limit,
            offset=offset,
        )

        # Add chunk count to each document
        docs_with_chunks = []
        for doc in documents:
            doc_read = DocumentRead.model_validate(doc)
            doc_read.num_chunks = len(doc.chunks)
            docs_with_chunks.append(doc_read)

        logger.debug(f"Listed {len(documents)} documents for user {user_id}")
        return {
            "documents": docs_with_chunks,
            "pagination": PaginationMeta(
                total=total,
                limit=limit,
                offset=offset,
            ),
        }

    except Exception as e:
        logger.error(f"Unexpected error in list_documents: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list documents"
        )


@router.delete(
    "/{document_id}",
    response_model=StatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete a document",
    responses={
        404: {"description": "Document not found"},
        500: {"description": "Internal server error"},
    },
)
async def delete_document(
    document_id: int = ...,
    user_id: int = Query(..., gt=0, description="ID of the user"),
    db: Session = Depends(get_db),
) -> StatusResponse:
    """
    Delete a document and all associated chunks.
    
    Validates user ownership before deletion.
    Associated FAISS index is also deleted.

    Args:
        document_id: ID of the document (path parameter).
        user_id: ID of the user (query parameter, must be positive).
        db: Database session (injected).

    Returns:
        StatusResponse with status="deleted".

    Raises:
        HTTPException 404: If document not found or access denied.
        HTTPException 500: If unexpected error occurs.
    """
    try:
        service = DocumentService(db)
        deleted = service.delete_document(document_id, user_id)

        if not deleted:
            logger.warning(f"Document {document_id} not found for user {user_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        logger.info(f"Deleted document {document_id}")
        return StatusResponse(status="deleted")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in delete_document: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document"
        )
