"""
Conversation API endpoints.

Provides REST endpoints for:
- Creating conversations (open or RAG mode)
- Adding messages to conversations
- Listing conversations with pagination
- Retrieving full conversation history
- Deleting conversations
"""
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.schemas import (
    ConversationCreate,
    ConversationRead,
    ConversationListItem,
    MessageCreate,
    MessageAddResponse,
    ConversationStartResponse,
    PaginationMeta,
    StatusResponse,
)
from app.services.conversation_service import (
    ConversationService,
    ConversationServiceError,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post(
    "",
    response_model=ConversationStartResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start a new conversation",
    responses={
        400: {"description": "Invalid request parameters"},
        500: {"description": "Internal server error"},
    },
)
async def start_conversation(
    payload: ConversationCreate,
    db: Session = Depends(get_db),
) -> ConversationStartResponse:
    """
    Start a new conversation with the first message.
    
    Creates a new conversation in either 'open' or 'rag' mode.
    In RAG mode, associates the conversation with specified documents.
    Generates an LLM response to the first message.

    Args:
        payload: Conversation creation payload with:
            - user_id: ID of the user
            - title: Conversation title
            - mode: "open" or "rag"
            - first_message: Initial user message
            - document_ids: Optional list of document IDs for RAG mode
        db: Database session (injected).

    Returns:
        ConversationStartResponse with conversation ID and initial messages.

    Raises:
        HTTPException 400: If parameters are invalid or LLM call fails.
        HTTPException 500: If unexpected error occurs.
    """
    try:
        service = ConversationService(db)
        conversation, messages = service.create_conversation(
            user_id=payload.user_id,
            title=payload.title,
            mode=payload.mode,
            first_message=payload.first_message,
            document_ids=payload.document_ids,
        )

        logger.info(f"Created conversation {conversation.id} for user {payload.user_id}")
        return ConversationStartResponse(
            conversation_id=conversation.id,
            messages=messages,
        )

    except ValueError as e:
        logger.warning(f"Invalid request: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ConversationServiceError as e:
        logger.error(f"Conversation service error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in start_conversation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create conversation"
        )


@router.post(
    "/{conversation_id}/messages",
    response_model=MessageAddResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add message to conversation",
    responses={
        400: {"description": "Invalid request or conversation not found"},
        404: {"description": "Conversation not found"},
        500: {"description": "Internal server error"},
    },
)
async def add_message(
    conversation_id: int,
    payload: MessageCreate,
    user_id: int = Query(..., gt=0, description="ID of the user"),
    db: Session = Depends(get_db),
) -> MessageAddResponse:
    """
    Add a new message to an existing conversation.
    
    Appends a user message and generates an LLM response.
    Validates user ownership of the conversation.

    Args:
        conversation_id: ID of the conversation (path parameter).
        payload: Message creation payload with content.
        user_id: ID of the user (query parameter, must be positive).
        db: Database session (injected).

    Returns:
        MessageAddResponse with new user and assistant messages.

    Raises:
        HTTPException 400: If parameters are invalid or service error.
        HTTPException 404: If conversation not found or access denied.
        HTTPException 500: If unexpected error occurs.
    """
    try:
        service = ConversationService(db)
        messages = service.add_message(
            conversation_id=conversation_id,
            user_id=user_id,
            content=payload.content,
        )

        logger.info(f"Added message to conversation {conversation_id}")
        return MessageAddResponse(messages=messages)

    except ValueError as e:
        logger.warning(f"Invalid request: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ConversationServiceError as e:
        error_msg = str(e)
        if "not found" in error_msg.lower():
            logger.warning(f"Conversation not found: {error_msg}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error_msg)
        logger.error(f"Conversation service error: {error_msg}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg)
    except Exception as e:
        logger.error(f"Unexpected error in add_message: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add message"
        )


@router.get(
    "",
    summary="List conversations for user",
    responses={
        500: {"description": "Internal server error"},
    },
)
async def list_conversations(
    user_id: int = Query(..., gt=0, description="ID of the user"),
    limit: int = Query(20, ge=1, le=100, description="Number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    db: Session = Depends(get_db),
) -> dict:
    """
    List all conversations for a user with pagination.

    Args:
        user_id: ID of the user (query parameter, must be positive).
        limit: Number of results to return (1-100, default 20).
        offset: Number of results to skip (default 0).
        db: Database session (injected).

    Returns:
        Dict with:
            - conversations: List of ConversationListItem
            - pagination: PaginationMeta with total, limit, offset
    """
    try:
        service = ConversationService(db)
        conversations, total = service.list_conversations(
            user_id=user_id,
            limit=limit,
            offset=offset,
        )

        logger.debug(f"Listed {len(conversations)} conversations for user {user_id}")
        return {
            "conversations": [
                ConversationListItem.model_validate(c) for c in conversations
            ],
            "pagination": PaginationMeta(
                total=total,
                limit=limit,
                offset=offset,
            ),
        }

    except Exception as e:
        logger.error(f"Unexpected error in list_conversations: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list conversations"
        )


@router.get(
    "/{conversation_id}",
    response_model=ConversationRead,
    summary="Get conversation history",
    responses={
        404: {"description": "Conversation not found"},
        500: {"description": "Internal server error"},
    },
)
async def get_conversation(
    conversation_id: int = ...,
    user_id: int = Query(..., gt=0, description="ID of the user"),
    db: Session = Depends(get_db),
) -> ConversationRead:
    """
    Get full conversation history with all messages.
    
    Validates user ownership before returning conversation.

    Args:
        conversation_id: ID of the conversation (path parameter).
        user_id: ID of the user (query parameter, must be positive).
        db: Database session (injected).

    Returns:
        ConversationRead with all messages ordered by creation time.

    Raises:
        HTTPException 404: If conversation not found or access denied.
        HTTPException 500: If unexpected error occurs.
    """
    try:
        service = ConversationService(db)
        conversation = service.get_conversation(conversation_id, user_id)

        if not conversation:
            logger.warning(f"Conversation {conversation_id} not found for user {user_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )

        logger.debug(f"Retrieved conversation {conversation_id}")
        return ConversationRead.model_validate(conversation)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_conversation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversation"
        )


@router.delete(
    "/{conversation_id}",
    response_model=StatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete a conversation",
    responses={
        404: {"description": "Conversation not found"},
        500: {"description": "Internal server error"},
    },
)
async def delete_conversation(
    conversation_id: int = ...,
    user_id: int = Query(..., gt=0, description="ID of the user"),
    db: Session = Depends(get_db),
) -> StatusResponse:
    """
    Delete a conversation and all associated messages.
    
    Validates user ownership before deletion.
    Associated documents are not deleted.

    Args:
        conversation_id: ID of the conversation (path parameter).
        user_id: ID of the user (query parameter, must be positive).
        db: Database session (injected).

    Returns:
        StatusResponse with status="deleted".

    Raises:
        HTTPException 404: If conversation not found or access denied.
        HTTPException 500: If unexpected error occurs.
    """
    try:
        service = ConversationService(db)
        deleted = service.delete_conversation(conversation_id, user_id)

        if not deleted:
            logger.warning(f"Conversation {conversation_id} not found for user {user_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )

        logger.info(f"Deleted conversation {conversation_id}")
        return StatusResponse(status="deleted")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in delete_conversation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete conversation"
        )
