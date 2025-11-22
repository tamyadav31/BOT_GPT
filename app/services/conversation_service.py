"""
Conversation service for managing conversations and messages.

Orchestrates:
- Conversation CRUD operations
- Message persistence
- LLM integration
- RAG context retrieval
- Conversation history management
"""
import logging
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from app.db.models import (
    Conversation, Message, ConversationMode, ConversationDocument, DocumentChunk
)
from app.db.schemas import MessageRead
from app.services.llm_service import get_llm_service, LLMServiceError
from app.services.rag_service import get_rag_service, RAGServiceError
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ConversationServiceError(Exception):
    """Custom exception for conversation service errors."""
    pass


class ConversationService:
    """
    Service for managing conversations and messages.
    
    Handles:
    - Conversation creation and deletion
    - Message persistence
    - LLM orchestration
    - RAG context retrieval
    - Conversation history management
    """

    def __init__(self, db: Session) -> None:
        """
        Initialize conversation service.
        
        Args:
            db: SQLAlchemy database session.
        """
        self.db = db
        self.llm_service = get_llm_service()
        self.rag_service = get_rag_service()

    def create_conversation(
        self,
        user_id: int,
        title: str,
        mode: str,
        first_message: str,
        document_ids: Optional[List[int]] = None,
    ) -> Tuple[Conversation, List[MessageRead]]:
        """
        Create a new conversation with the first message.
        
        Handles:
        - Mode validation (open or rag)
        - Document association for RAG mode
        - LLM response generation
        - Transaction rollback on failure

        Args:
            user_id: ID of the user (must be positive).
            title: Conversation title (non-empty).
            mode: "open" or "rag".
            first_message: The user's first message (non-empty).
            document_ids: Optional list of document IDs for RAG mode.

        Returns:
            Tuple of (Conversation object, list of MessageRead).

        Raises:
            ConversationServiceError: If conversation creation fails.
            ValueError: If parameters are invalid.
        """
        # Validate parameters
        if user_id <= 0:
            raise ValueError(f"Invalid user_id: {user_id}")
        if not title or not title.strip():
            raise ValueError("title cannot be empty")
        if mode not in ("open", "rag"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'open' or 'rag'")
        if not first_message or not first_message.strip():
            raise ValueError("first_message cannot be empty")
        
        try:
            # Create conversation
            conversation = Conversation(
                user_id=user_id,
                title=title.strip(),
                mode=ConversationMode(mode),
            )
            self.db.add(conversation)
            self.db.flush()  # Get the ID without committing

            # Create user message
            user_msg = Message(
                conversation_id=conversation.id,
                role="user",
                content=first_message.strip(),
            )
            self.db.add(user_msg)

            # Attach documents if in RAG mode
            if mode == "rag" and document_ids:
                for doc_id in document_ids:
                    if doc_id <= 0:
                        logger.warning(f"Skipping invalid document_id: {doc_id}")
                        continue
                    conv_doc = ConversationDocument(
                        conversation_id=conversation.id,
                        document_id=doc_id,
                    )
                    self.db.add(conv_doc)

            self.db.flush()

            # Get LLM response
            try:
                assistant_response = self._get_llm_response(
                    conversation_id=conversation.id,
                    user_message=first_message.strip(),
                    mode=mode,
                )
            except LLMServiceError as e:
                logger.error(f"LLM service error: {str(e)}")
                raise ConversationServiceError(f"Failed to get LLM response: {str(e)}")

            # Create assistant message
            assistant_msg = Message(
                conversation_id=conversation.id,
                role="assistant",
                content=assistant_response,
            )
            self.db.add(assistant_msg)
            self.db.commit()

            # Refresh to get all relationships
            self.db.refresh(conversation)

            messages = [
                MessageRead.model_validate(user_msg),
                MessageRead.model_validate(assistant_msg),
            ]

            logger.info(f"Created conversation {conversation.id} for user {user_id} in {mode} mode")
            return conversation, messages

        except (ValueError, ConversationServiceError):
            self.db.rollback()
            raise
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to create conversation: {str(e)}")
            raise ConversationServiceError(f"Failed to create conversation: {str(e)}")

    def add_message(
        self,
        conversation_id: int,
        user_id: int,
        content: str,
    ) -> List[MessageRead]:
        """
        Add a new message to an existing conversation.
        
        Validates user ownership and generates LLM response.

        Args:
            conversation_id: ID of the conversation (must be positive).
            user_id: ID of the user (must be positive, for validation).
            content: Message content (non-empty).

        Returns:
            List of new messages (user + assistant).

        Raises:
            ConversationServiceError: If message addition fails.
            ValueError: If parameters are invalid.
        """
        # Validate parameters
        if conversation_id <= 0:
            raise ValueError(f"Invalid conversation_id: {conversation_id}")
        if user_id <= 0:
            raise ValueError(f"Invalid user_id: {user_id}")
        if not content or not content.strip():
            raise ValueError("content cannot be empty")
        
        try:
            # Get conversation
            conversation = self.db.query(Conversation).filter(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            ).first()

            if not conversation:
                raise ConversationServiceError(f"Conversation {conversation_id} not found or access denied")

            # Create user message
            user_msg = Message(
                conversation_id=conversation_id,
                role="user",
                content=content,
            )
            self.db.add(user_msg)
            self.db.flush()

            # Get LLM response
            try:
                assistant_response = self._get_llm_response(
                    conversation_id=conversation_id,
                    user_message=content,
                    mode=conversation.mode.value,
                )
            except LLMServiceError as e:
                logger.error(f"LLM service error: {str(e)}")
                raise ConversationServiceError(f"Failed to get LLM response: {str(e)}")

            # Create assistant message
            assistant_msg = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=assistant_response,
            )
            self.db.add(assistant_msg)
            self.db.commit()

            messages = [
                MessageRead.model_validate(user_msg),
                MessageRead.model_validate(assistant_msg),
            ]

            return messages

        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to add message: {str(e)}")
            raise ConversationServiceError(f"Failed to add message: {str(e)}")

    def _get_llm_response(
        self,
        conversation_id: int,
        user_message: str,
        mode: str,
    ) -> str:
        """
        Get LLM response for a message.

        Args:
            conversation_id: ID of the conversation.
            user_message: The user's message.
            mode: "open" or "rag".

        Returns:
            Assistant response text.
        """
        # Build message history
        messages = self._build_message_history(conversation_id, user_message, mode)

        # Call LLM
        response = self.llm_service.chat_completion(messages)
        return response

    def _build_message_history(
        self,
        conversation_id: int,
        user_message: str,
        mode: str,
    ) -> List[dict]:
        """
        Build the message history for LLM context.
        
        Constructs prompt with:
        1. System prompt (mode-specific)
        2. RAG context (if RAG mode)
        3. Conversation history (sliding window)
        4. Current user message

        Args:
            conversation_id: ID of the conversation.
            user_message: The current user message.
            mode: "open" or "rag".

        Returns:
            List of message dicts for LLM API (role + content).
        """
        messages = []

        # 1. System prompt (mode-specific)
        system_prompt = "You are a helpful AI assistant."
        if mode == "rag":
            system_prompt += " You have access to document context provided below."

        messages.append({"role": "system", "content": system_prompt})

        # 2. Add RAG context if applicable
        if mode == "rag":
            context = self._retrieve_rag_context(conversation_id, user_message)
            if context:
                messages.append({
                    "role": "system",
                    "content": f"Document Context:\n{context}",
                })
            else:
                logger.debug(f"No RAG context found for conversation {conversation_id}")

        # 3. Add conversation history (sliding window for efficiency)
        history = self._get_conversation_history(conversation_id)
        history_window = history[-settings.MAX_HISTORY_MESSAGES:]
        for msg in history_window:
            messages.append({"role": msg.role, "content": msg.content})

        # 4. Add current user message
        messages.append({"role": "user", "content": user_message})

        logger.debug(f"Built message history with {len(messages)} messages for conversation {conversation_id}")
        return messages

    def _retrieve_rag_context(self, conversation_id: int, query: str) -> str:
        """
        Retrieve RAG context for a conversation.

        Args:
            conversation_id: ID of the conversation.
            query: Query text.

        Returns:
            Formatted context string.
        """
        try:
            # Get associated documents
            conv_docs = self.db.query(ConversationDocument).filter(
                ConversationDocument.conversation_id == conversation_id,
            ).all()

            if not conv_docs:
                return ""

            context_parts = []

            for conv_doc in conv_docs:
                try:
                    # Retrieve top-k chunks
                    results = self.rag_service.retrieve_top_k(
                        conv_doc.document_id,
                        query,
                        k=settings.RAG_TOP_K,
                    )

                    # Fetch chunk texts from DB
                    for chunk_idx, _ in results:
                        chunk = self.db.query(DocumentChunk).filter(
                            DocumentChunk.document_id == conv_doc.document_id,
                            DocumentChunk.chunk_index == chunk_idx,
                        ).first()

                        if chunk:
                            context_parts.append(chunk.text)

                except RAGServiceError as e:
                    logger.warning(f"RAG retrieval failed for document {conv_doc.document_id}: {str(e)}")

            return "\n\n".join(context_parts)

        except Exception as e:
            logger.error(f"Failed to retrieve RAG context: {str(e)}")
            return ""

    def _get_conversation_history(self, conversation_id: int) -> List[Message]:
        """
        Get all messages in a conversation.

        Args:
            conversation_id: ID of the conversation.

        Returns:
            List of Message objects ordered by creation time.
        """
        return self.db.query(Message).filter(
            Message.conversation_id == conversation_id,
        ).order_by(Message.created_at).all()

    def get_conversation(self, conversation_id: int, user_id: int) -> Optional[Conversation]:
        """
        Get a conversation by ID.

        Args:
            conversation_id: ID of the conversation.
            user_id: ID of the user (for validation).

        Returns:
            Conversation object or None.
        """
        return self.db.query(Conversation).filter(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        ).first()

    def list_conversations(
        self,
        user_id: int,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[List[Conversation], int]:
        """
        List conversations for a user.

        Args:
            user_id: ID of the user.
            limit: Number of results to return.
            offset: Number of results to skip.

        Returns:
            Tuple of (list of Conversation objects, total count).
        """
        query = self.db.query(Conversation).filter(
            Conversation.user_id == user_id,
        ).order_by(Conversation.created_at.desc())

        total = query.count()
        conversations = query.limit(limit).offset(offset).all()

        return conversations, total

    def delete_conversation(self, conversation_id: int, user_id: int) -> bool:
        """
        Delete a conversation.

        Args:
            conversation_id: ID of the conversation.
            user_id: ID of the user (for validation).

        Returns:
            True if deleted, False if not found.
        """
        try:
            conversation = self.db.query(Conversation).filter(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            ).first()

            if not conversation:
                return False

            self.db.delete(conversation)
            self.db.commit()
            return True

        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to delete conversation: {str(e)}")
            raise ConversationServiceError(f"Failed to delete conversation: {str(e)}")
