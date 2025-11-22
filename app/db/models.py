"""
SQLAlchemy ORM models for BOT GPT.

Models represent core entities:
- User: Platform users
- Conversation: Chat sessions (open or RAG mode)
- Message: Individual messages in conversations
- Document: Uploaded documents for RAG
- DocumentChunk: Text chunks from documents
- ConversationDocument: Link table for RAG associations
"""
from datetime import datetime
import enum
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Enum, Index
from sqlalchemy.orm import relationship

from app.db.database import Base


class ConversationMode(str, enum.Enum):
    """Conversation mode enumeration."""
    OPEN = "open"
    RAG = "rag"


class User(Base):
    """
    User model.
    
    Represents a platform user with their conversations and documents.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    conversations = relationship(
        "Conversation",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="select"
    )
    documents = relationship(
        "Document",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="select"
    )


class Conversation(Base):
    """
    Conversation model.
    
    Represents a chat session that can operate in two modes:
    - OPEN: Uses conversation history only
    - RAG: Uses conversation history + document context
    """
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    mode = Column(Enum(ConversationMode), default=ConversationMode.OPEN, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        lazy="select"
    )
    documents = relationship(
        "ConversationDocument",
        back_populates="conversation",
        cascade="all, delete-orphan",
        lazy="select"
    )


class Message(Base):
    """
    Message model.
    
    Represents a single message in a conversation.
    Role can be: "user", "assistant", or "system".
    """
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False, index=True)
    role = Column(String(50), nullable=False)  # "user", "assistant", "system"
    content = Column(Text, nullable=False)
    tokens_used = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")


class Document(Base):
    """
    Document model.
    
    Represents an uploaded document used for RAG.
    Documents are chunked and indexed for vector search.
    """
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    path = Column(String(512), nullable=True)  # Optional: filesystem path
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    user = relationship("User", back_populates="documents")
    chunks = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="select"
    )
    conversations = relationship(
        "ConversationDocument",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="select"
    )


class DocumentChunk(Base):
    """
    Document chunk model for RAG.
    
    Represents a text chunk from a document.
    Chunks are embedded and indexed in FAISS for vector search.
    """
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)

    # Relationships
    document = relationship("Document", back_populates="chunks")


class ConversationDocument(Base):
    """
    Link table: associates documents with conversations for RAG.
    
    Enables conversations to reference multiple documents
    for context retrieval.
    """
    __tablename__ = "conversation_documents"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)

    # Relationships
    conversation = relationship("Conversation", back_populates="documents")
    document = relationship("Document")
