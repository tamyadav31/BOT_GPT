"""
Pytest configuration and fixtures.

Provides:
- In-memory SQLite database for testing
- Database session fixtures
- FastAPI test client
- Test data factories
"""
import logging
from typing import Generator

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from fastapi.testclient import TestClient

from app.main import app
from app.db.database import Base, get_db
from app.db.models import User, Document, Conversation, ConversationMode

logger = logging.getLogger(__name__)

# Create in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
)

# Enable foreign key constraints for SQLite
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    """Enable foreign key constraints for SQLite."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db() -> Generator[Session, None, None]:
    """
    Create a fresh database for each test.
    
    Yields:
        SQLAlchemy session for testing.
    """
    Base.metadata.create_all(bind=engine)
    db_session = TestingSessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db: Session) -> Generator[TestClient, None, None]:
    """
    Create a test client with a fresh database.
    
    Args:
        db: Database session fixture.
        
    Yields:
        FastAPI TestClient for making requests.
    """
    def override_get_db() -> Generator[Session, None, None]:
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def test_user(db: Session) -> User:
    """
    Create a test user.
    
    Args:
        db: Database session fixture.
        
    Returns:
        User object with test data.
    """
    user = User(
        name="Test User",
        email="test@example.com",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.debug(f"Created test user {user.id}")
    return user


@pytest.fixture
def test_document(db: Session, test_user: User) -> Document:
    """
    Create a test document.
    
    Args:
        db: Database session fixture.
        test_user: Test user fixture.
        
    Returns:
        Document object with test data.
    """
    document = Document(
        user_id=test_user.id,
        title="Test Document",
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    logger.debug(f"Created test document {document.id}")
    return document


@pytest.fixture
def test_conversation(db: Session, test_user: User) -> Conversation:
    """
    Create a test conversation.
    
    Args:
        db: Database session fixture.
        test_user: Test user fixture.
        
    Returns:
        Conversation object with test data.
    """
    conversation = Conversation(
        user_id=test_user.id,
        title="Test Conversation",
        mode=ConversationMode.OPEN,
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    logger.debug(f"Created test conversation {conversation.id}")
    return conversation
