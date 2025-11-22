"""
Tests for conversation service and endpoints.

Covers:
- Conversation creation (open and RAG modes)
- Message addition
- Conversation listing with pagination
- Conversation deletion
- Error handling and validation
"""
import pytest
from unittest.mock import patch, AsyncMock
from sqlalchemy.orm import Session

from app.db.models import User, Conversation, Message, ConversationMode
from app.services.conversation_service import ConversationService, ConversationServiceError


class TestStartConversation:
    """Tests for starting a new conversation."""

    @patch("app.services.conversation_service.get_llm_service")
    def test_start_conversation_open_mode(self, mock_llm_service, db: Session, test_user):
        """Test starting a conversation in open mode."""
        # Mock LLM service
        mock_llm = AsyncMock()
        mock_llm.chat_completion = AsyncMock(return_value="Hello! How can I help?")
        mock_llm_service.return_value = mock_llm

        service = ConversationService(db)
        conversation, messages = service.create_conversation(
            user_id=test_user.id,
            title="Test Conversation",
            mode="open",
            first_message="Hello, BOT GPT",
        )

        # Assertions
        assert conversation.id is not None
        assert conversation.user_id == test_user.id
        assert conversation.title == "Test Conversation"
        assert conversation.mode == ConversationMode.OPEN

        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "Hello, BOT GPT"
        assert messages[1].role == "assistant"
        assert messages[1].content == "Hello! How can I help?"

    @patch("app.services.conversation_service.get_llm_service")
    def test_start_conversation_rag_mode(self, mock_llm_service, db: Session, test_user):
        """Test starting a conversation in RAG mode."""
        # Mock LLM service
        mock_llm = AsyncMock()
        mock_llm.chat_completion = AsyncMock(return_value="Based on the documents...")
        mock_llm_service.return_value = mock_llm

        service = ConversationService(db)
        conversation, messages = service.create_conversation(
            user_id=test_user.id,
            title="RAG Conversation",
            mode="rag",
            first_message="What's in the documents?",
            document_ids=[],  # No documents for this test
        )

        # Assertions
        assert conversation.mode == ConversationMode.RAG
        assert len(messages) == 2
        assert messages[1].role == "assistant"

    def test_start_conversation_invalid_mode(self, db: Session, test_user):
        """Test starting a conversation with invalid mode."""
        service = ConversationService(db)

        with pytest.raises(ValueError):
            service.create_conversation(
                user_id=test_user.id,
                title="Invalid",
                mode="invalid_mode",
                first_message="Hello",
            )

    def test_start_conversation_invalid_user_id(self, db: Session):
        """Test starting a conversation with invalid user_id."""
        service = ConversationService(db)

        with pytest.raises(ValueError):
            service.create_conversation(
                user_id=-1,
                title="Invalid",
                mode="open",
                first_message="Hello",
            )

    def test_start_conversation_empty_title(self, db: Session, test_user):
        """Test starting a conversation with empty title."""
        service = ConversationService(db)

        with pytest.raises(ValueError):
            service.create_conversation(
                user_id=test_user.id,
                title="",
                mode="open",
                first_message="Hello",
            )

    def test_start_conversation_empty_message(self, db: Session, test_user):
        """Test starting a conversation with empty first message."""
        service = ConversationService(db)

        with pytest.raises(ValueError):
            service.create_conversation(
                user_id=test_user.id,
                title="Test",
                mode="open",
                first_message="",
            )

    def test_start_conversation_whitespace_message(self, db: Session, test_user):
        """Test starting a conversation with whitespace-only message."""
        service = ConversationService(db)

        with pytest.raises(ValueError):
            service.create_conversation(
                user_id=test_user.id,
                title="Test",
                mode="open",
                first_message="   ",
            )


class TestAddMessage:
    """Tests for adding messages to a conversation."""

    @pytest.fixture
    def test_conversation(self, db: Session, test_user):
        """Create a test conversation."""
        conv = Conversation(
            user_id=test_user.id,
            title="Test Conv",
            mode=ConversationMode.OPEN,
        )
        db.add(conv)
        db.commit()
        db.refresh(conv)
        return conv

    @patch("app.services.conversation_service.get_llm_service")
    def test_add_message_to_conversation(self, mock_llm_service, db: Session, test_user, test_conversation):
        """Test adding a message to an existing conversation."""
        # Mock LLM service
        mock_llm = AsyncMock()
        mock_llm.chat_completion = AsyncMock(return_value="Response to your question")
        mock_llm_service.return_value = mock_llm

        service = ConversationService(db)
        messages = service.add_message(
            conversation_id=test_conversation.id,
            user_id=test_user.id,
            content="What's the answer?",
        )

        # Assertions
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "What's the answer?"
        assert messages[1].role == "assistant"
        assert messages[1].content == "Response to your question"

        # Verify messages are persisted
        db_messages = db.query(Message).filter(
            Message.conversation_id == test_conversation.id
        ).all()
        assert len(db_messages) == 2

    def test_add_message_to_nonexistent_conversation(self, db: Session, test_user):
        """Test adding a message to a non-existent conversation."""
        service = ConversationService(db)

        with pytest.raises(ConversationServiceError):
            service.add_message(
                conversation_id=999,
                user_id=test_user.id,
                content="Hello",
            )

    def test_add_message_invalid_conversation_id(self, db: Session, test_user):
        """Test adding a message with invalid conversation_id."""
        service = ConversationService(db)

        with pytest.raises(ValueError):
            service.add_message(
                conversation_id=-1,
                user_id=test_user.id,
                content="Hello",
            )

    def test_add_message_invalid_user_id(self, db: Session, test_conversation):
        """Test adding a message with invalid user_id."""
        service = ConversationService(db)

        with pytest.raises(ValueError):
            service.add_message(
                conversation_id=test_conversation.id,
                user_id=-1,
                content="Hello",
            )

    def test_add_message_empty_content(self, db: Session, test_user, test_conversation):
        """Test adding a message with empty content."""
        service = ConversationService(db)

        with pytest.raises(ValueError):
            service.add_message(
                conversation_id=test_conversation.id,
                user_id=test_user.id,
                content="",
            )

    def test_add_message_whitespace_content(self, db: Session, test_user, test_conversation):
        """Test adding a message with whitespace-only content."""
        service = ConversationService(db)

        with pytest.raises(ValueError):
            service.add_message(
                conversation_id=test_conversation.id,
                user_id=test_user.id,
                content="   ",
            )

    def test_add_message_wrong_user(self, db: Session, test_user, test_conversation):
        """Test adding a message with wrong user_id (access denied)."""
        service = ConversationService(db)
        other_user = User(name="Other User", email="other@example.com")
        db.add(other_user)
        db.commit()

        with pytest.raises(ConversationServiceError):
            service.add_message(
                conversation_id=test_conversation.id,
                user_id=other_user.id,
                content="Hello",
            )


class TestListConversations:
    """Tests for listing conversations."""

    def test_list_conversations_empty(self, db: Session, test_user):
        """Test listing conversations when none exist."""
        service = ConversationService(db)
        conversations, total = service.list_conversations(user_id=test_user.id)

        assert conversations == []
        assert total == 0

    def test_list_conversations_with_pagination(self, db: Session, test_user):
        """Test listing conversations with pagination."""
        # Create multiple conversations
        for i in range(5):
            conv = Conversation(
                user_id=test_user.id,
                title=f"Conversation {i}",
                mode=ConversationMode.OPEN,
            )
            db.add(conv)
        db.commit()

        service = ConversationService(db)
        conversations, total = service.list_conversations(
            user_id=test_user.id,
            limit=2,
            offset=0,
        )

        assert len(conversations) == 2
        assert total == 5

        # Test offset
        conversations, total = service.list_conversations(
            user_id=test_user.id,
            limit=2,
            offset=2,
        )
        assert len(conversations) == 2


class TestDeleteConversation:
    """Tests for deleting conversations."""

    def test_delete_conversation(self, db: Session, test_user):
        """Test deleting a conversation."""
        # Create conversation
        conv = Conversation(
            user_id=test_user.id,
            title="To Delete",
            mode=ConversationMode.OPEN,
        )
        db.add(conv)
        db.commit()

        service = ConversationService(db)
        deleted = service.delete_conversation(conv.id, test_user.id)

        assert deleted is True

        # Verify it's deleted
        db_conv = db.query(Conversation).filter(Conversation.id == conv.id).first()
        assert db_conv is None

    def test_delete_nonexistent_conversation(self, db: Session, test_user):
        """Test deleting a non-existent conversation."""
        service = ConversationService(db)
        deleted = service.delete_conversation(999, test_user.id)

        assert deleted is False
