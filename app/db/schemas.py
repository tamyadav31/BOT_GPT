"""
Pydantic schemas for request/response validation.

Schemas are organized by entity:
- User: User account schemas
- Message: Message schemas
- Conversation: Conversation schemas
- Document: Document schemas
- Response: API response schemas
- Pagination: Pagination helper schemas
"""
from datetime import datetime
from typing import Optional, List, Generic, TypeVar
from pydantic import BaseModel, EmailStr, Field, field_validator

T = TypeVar("T")


# ============================================================================
# User Schemas
# ============================================================================

class UserCreate(BaseModel):
    """Schema for creating a user."""
    name: str = Field(..., min_length=1, max_length=255)
    email: EmailStr


class UserRead(BaseModel):
    """Schema for reading a user."""
    id: int
    name: str
    email: str
    created_at: datetime

    class Config:
        from_attributes = True


# Alias for API responses
UserResponse = UserRead


# ============================================================================
# Message Schemas
# ============================================================================

class MessageCreate(BaseModel):
    """Schema for creating a message."""
    content: str = Field(..., min_length=1)

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        """Ensure content is not just whitespace."""
        if not v.strip():
            raise ValueError("Message content cannot be empty or whitespace")
        return v.strip()


class MessageRead(BaseModel):
    """Schema for reading a message."""
    id: int
    role: str
    content: str
    tokens_used: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True


# ============================================================================
# Conversation Schemas
# ============================================================================

class ConversationCreate(BaseModel):
    """Schema for creating a conversation."""
    user_id: int = Field(..., gt=0)
    title: str = Field(..., min_length=1, max_length=255)
    mode: str = Field("open", pattern="^(open|rag)$")
    first_message: str = Field(..., min_length=1)
    document_ids: Optional[List[int]] = None

    @field_validator("first_message")
    @classmethod
    def first_message_not_empty(cls, v: str) -> str:
        """Ensure first message is not just whitespace."""
        if not v.strip():
            raise ValueError("First message cannot be empty or whitespace")
        return v.strip()


class ConversationRead(BaseModel):
    """Schema for reading a conversation."""
    id: int
    user_id: int
    title: str
    mode: str
    created_at: datetime
    messages: List[MessageRead] = []

    class Config:
        from_attributes = True


class ConversationListItem(BaseModel):
    """Schema for listing conversations."""
    id: int
    title: str
    mode: str
    created_at: datetime

    class Config:
        from_attributes = True


# ============================================================================
# Document Schemas
# ============================================================================

class DocumentCreate(BaseModel):
    """Schema for creating a document."""
    user_id: int = Field(..., gt=0)
    title: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1)

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        """Ensure content is not just whitespace."""
        if not v.strip():
            raise ValueError("Document content cannot be empty or whitespace")
        return v.strip()


class DocumentRead(BaseModel):
    """Schema for reading a document."""
    id: int
    user_id: int
    title: str
    created_at: datetime
    num_chunks: int = 0

    class Config:
        from_attributes = True


class DocumentChunkRead(BaseModel):
    """Schema for reading a document chunk."""
    id: int
    chunk_index: int
    text: str

    class Config:
        from_attributes = True


# ============================================================================
# Pagination Schemas
# ============================================================================

class PaginationMeta(BaseModel):
    """Pagination metadata."""
    total: int = Field(..., ge=0)
    limit: int = Field(..., gt=0)
    offset: int = Field(..., ge=0)


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response wrapper."""
    items: List[T]
    pagination: PaginationMeta


# ============================================================================
# API Response Schemas
# ============================================================================

class ConversationStartResponse(BaseModel):
    """Response for starting a new conversation."""
    conversation_id: int
    messages: List[MessageRead]


class MessageAddResponse(BaseModel):
    """Response for adding a message to a conversation."""
    messages: List[MessageRead]


class DocumentUploadResponse(BaseModel):
    """Response for uploading a document."""
    document_id: int
    num_chunks: int


class StatusResponse(BaseModel):
    """Generic status response."""
    status: str = Field(..., pattern="^[a-z_]+$")
