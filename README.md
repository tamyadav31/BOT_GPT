# BOT GPT - Conversational AI Backend

A production-grade conversational AI platform with RAG (Retrieval-Augmented Generation) capabilities, built with FastAPI and featuring a professional Streamlit UI.

## 🚀 Quick Start

```bash
git clone https://github.com/tamyadav31/BOT_GPT.git
cd BOT_GPT
pip install -r requirements.txt
cp .env.example .env
# Add your Groq API key to .env
uvicorn app.main:app --reload
# In another terminal: streamlit run streamlit_ui.py
```

Visit `http://localhost:8501` for the UI or `http://localhost:8000/docs` for API docs.

## Features

- **Professional Streamlit UI**: Interactive chat interface with document management and conversation history
- **Conversation Management**: Create and manage multi-turn conversations with proper validation
- **LLM Integration**: Seamless integration with external LLM providers (Groq, OpenAI, etc.)
- **RAG with FAISS**: Vector-based document retrieval using FAISS and SentenceTransformers
- **Document Management**: Upload and index documents for RAG with graceful error handling
- **Clean Architecture**: Separation of concerns with API, service, and data layers
- **SQLite Persistence**: Lightweight database with foreign key constraints and cascading deletes
- **Comprehensive Tests**: 34 unit and integration tests with ~62% code coverage
- **Docker Support**: Multi-stage build with health checks and non-root user
- **Production Ready**: Proper HTTP status codes, error handling, logging, and validation
- **Type Safe**: Complete type hints and Pydantic validation throughout

## Tech Stack

- **Backend**: FastAPI + Uvicorn
- **Frontend**: Streamlit + Plotly
- **Database**: SQLite + SQLAlchemy ORM
- **Vector Search**: FAISS (CPU)
- **Embeddings**: SentenceTransformers
- **LLM Provider**: Groq API (llama-3.1-8b-instant)
- **Testing**: pytest
- **Containerization**: Docker

## Project Structure

```
BOT_GPT/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── api/
│   │   ├── conversations.py    # Conversation endpoints
│   │   └── documents.py        # Document endpoints
│   ├── core/
│   │   └── config.py           # Configuration management
│   ├── db/
│   │   ├── database.py         # Database setup
│   │   ├── models.py           # SQLAlchemy models
│   │   └── schemas.py          # Pydantic schemas
│   ├── services/
│   │   ├── conversation_service.py  # Conversation logic
│   │   ├── document_service.py      # Document logic
│   │   ├── llm_service.py           # LLM integration
│   │   └── rag_service.py           # RAG operations
│   └── utils/
│       └── chunking.py         # Text chunking utilities
├── tests/
│   ├── conftest.py             # Test configuration
│   ├── test_conversations.py   # Conversation tests
│   └── test_documents.py       # Document tests
├── streamlit_ui.py             # Professional Streamlit frontend
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configuration
├── .env.example               # Environment variables template
└── README.md                  # This file
```

## Setup & Installation

### Prerequisites

- Python 3.11+
- pip
- Groq API key (free at [console.groq.com](https://console.groq.com))

### 1. Clone the Repository

```bash
git clone https://github.com/tamyadav31/BOT_GPT.git
cd BOT_GPT
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key:

```env
LLM_API_KEY=your_groq_api_key_here
LLM_API_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.1-8b-instant
```

### 5. Run the Backend

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 6. Run the Streamlit UI (Optional)

```bash
streamlit run streamlit_ui.py
```

The UI will be available at `http://localhost:8501`

- **Professional Chat Interface**: Interactive conversation UI
- **Document Upload**: Upload and manage documents for RAG
- **Conversation History**: View and manage past conversations
- **Real-time Chat**: Live chat with progress indicators

## API Endpoints

### Conversations

- `POST /conversations` - Start a new conversation (201 Created)
- `POST /conversations/{conversation_id}/messages` - Add message to conversation (201 Created)
- `GET /conversations` - List user's conversations with pagination (200 OK)
- `GET /conversations/{conversation_id}` - Get full conversation history (200 OK)
- `DELETE /conversations/{conversation_id}` - Delete a conversation (200 OK)

### Documents

- `POST /documents` - Upload/register a document (201 Created)
- `GET /documents` - List user's documents with pagination (200 OK)
- `DELETE /documents/{document_id}` - Delete a document (200 OK)

### HTTP Status Codes

- **201 Created**: POST requests that create new resources
- **200 OK**: GET and DELETE requests that succeed
- **400 Bad Request**: Invalid parameters or validation errors
- **404 Not Found**: Resource not found or access denied
- **500 Internal Server Error**: Unexpected server errors

### Error Responses

All error responses follow this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

Examples:
- Invalid user_id: `{"detail": "Invalid user_id: -1"}`
- Empty title: `{"detail": "title cannot be empty"}`
- Conversation not found: `{"detail": "Conversation not found"}`

## Usage Examples

### 1. Start a Conversation (Open Mode)

```bash
curl -X POST "http://localhost:8000/conversations" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "title": "My First Chat",
    "mode": "open",
    "first_message": "Hello, BOT GPT!"
  }'
```

### 2. Upload a Document

```bash
curl -X POST "http://localhost:8000/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "title": "Company Policy",
    "content": "This is the company policy document..."
  }'
```

### 3. Start RAG Conversation

```bash
curl -X POST "http://localhost:8000/conversations" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "title": "Policy Questions",
    "mode": "rag",
    "first_message": "What is the vacation policy?",
    "document_ids": [1]
  }'
```

### 4. Add Message to Conversation

```bash
curl -X POST "http://localhost:8000/conversations/1/messages?user_id=1" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "What about sick leave?"
  }'
```

## Input Validation

The API validates all inputs and returns meaningful error messages:

### Conversation Creation
- `user_id`: Must be positive integer
- `title`: Cannot be empty
- `mode`: Must be "open" or "rag"
- `first_message`: Cannot be empty or whitespace-only
- `document_ids`: Must exist and belong to the user

### Document Upload
- `user_id`: Must be positive integer
- `title`: Cannot be empty
- `content`: Cannot be empty or whitespace-only

### Message Addition
- `conversation_id`: Must be positive integer and exist
- `user_id`: Must be positive integer and match conversation owner
- `content`: Cannot be empty or whitespace-only

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_conversations.py

# Run with coverage
pytest --cov=app tests/

# Run specific test class
pytest tests/test_conversations.py::TestStartConversation

# Run with detailed output
pytest -vv
```

### Test Coverage

The project includes **34 tests** with ~62% code coverage:
- **13 positive tests**: Happy path scenarios
- **21 negative tests**: Error cases and validation
- **8 edge case tests**: Boundary conditions

Tests cover:
- Conversation creation (open and RAG modes)
- Message addition with validation
- Document upload with error handling
- Text chunking with edge cases
- Pagination
- Access control (user ownership)
- Empty/whitespace input handling
- Invalid ID handling

## Docker Deployment

### Build Image

```bash
docker build -t bot-gpt-backend .
```

### Run Container

```bash
docker run -p 8000:8000 \
  -e LLM_API_KEY=your_api_key \
  -e LLM_API_BASE_URL=https://api.groq.com/openai \
  -e LLM_MODEL=mixtral-8x7b-32768 \
  bot-gpt-backend
```

### With Environment File

```bash
docker run -p 8000:8000 \
  --env-file .env \
  bot-gpt-backend
```

## Configuration

All configuration is managed through environment variables (see `.env.example`):

- **DATABASE_URL**: SQLite database path
- **LLM_API_KEY**: Your LLM provider API key
- **LLM_API_BASE_URL**: LLM provider endpoint
- **LLM_MODEL**: Model name to use
- **RAG_MODEL**: Embedding model for RAG
- **RAG_TOP_K**: Number of chunks to retrieve
- **MAX_HISTORY_MESSAGES**: Conversation history window size
- **CHUNK_SIZE**: Document chunk size for RAG
- **CHUNK_OVERLAP**: Overlap between chunks

## Architecture Overview

### API Layer (`app/api/`)
RESTful endpoints for conversations and documents. Handles request validation and error responses.

### Service Layer (`app/services/`)
Business logic orchestration:
- **ConversationService**: Manages conversation flow and LLM integration
- **DocumentService**: Handles document storage and RAG indexing
- **LLMService**: External LLM API communication
- **RAGService**: FAISS vector search and embedding

### Data Layer (`app/db/`)
- **Models**: SQLAlchemy ORM definitions
- **Schemas**: Pydantic request/response validation
- **Database**: Session management and initialization

### Utilities (`app/utils/`)
- **Chunking**: Text splitting for document processing

## Key Design Decisions

1. **Minimal but Clean**: Focuses on clarity and maintainability without over-engineering
2. **Separation of Concerns**: Clear boundaries between API, business logic, and data layers
3. **Dependency Injection**: Services are instantiated with dependencies for testability
4. **Error Handling**: Graceful error handling with meaningful error messages and proper HTTP status codes
5. **Logging**: Comprehensive logging at INFO, WARNING, ERROR, and DEBUG levels
6. **Configuration**: Environment-based configuration for flexibility
7. **Type Safety**: Complete type hints and Pydantic validation throughout
8. **Validation**: Input validation at all layers (API, service, utility)
9. **Testing**: Comprehensive test coverage including negative and edge cases
10. **Production Ready**: Multi-stage Docker builds, health checks, non-root user

## Refinement & Quality Improvements

This codebase has been comprehensively refined with:

### Error Handling
- Specific exception types (ValueError, ServiceError) instead of generic exceptions
- Proper HTTP status codes (201 Created, 404 Not Found, 400 Bad Request, 500 Internal Server Error)
- Meaningful error messages for debugging
- Global exception handlers for unexpected errors

### Validation
- Input validation throughout all layers
- Empty/whitespace input detection
- Invalid ID detection (negative, zero)
- Non-existent resource detection
- User ownership validation

### Logging
- INFO level: Successful operations
- WARNING level: Validation and not found errors
- ERROR level: Service errors with stack traces
- DEBUG level: List operations and cache hits

### Testing
- 34 tests with ~62% code coverage
- 21 negative tests for error scenarios
- 8 edge case tests for boundary conditions
- FAISS mocking for safe testing
- Fixture consistency with type hints

### Code Quality
- 60+ docstrings added/improved
- 50+ type hints added
- Comprehensive method documentation
- Clean code following Python best practices

### Production Readiness
- Multi-stage Docker build for smaller images
- Non-root user for security
- Health check endpoint
- Foreign key constraints with cascading deletes
- Optional GPU support for FAISS

## Troubleshooting

### FAISS Index Not Found
Ensure the `data/indexes` directory exists and has write permissions.

### LLM API Errors
- Verify `LLM_API_KEY` is set correctly
- Check `LLM_API_BASE_URL` matches your provider
- Ensure network connectivity to the LLM provider

### Database Errors
- Check `DATABASE_URL` is valid
- Ensure the directory for SQLite file exists
- Verify file permissions

## Contributing

When contributing:
1. Follow the existing code structure
2. Add tests for new features
3. Update documentation
4. Ensure all tests pass before submitting

## License

MIT License - See LICENSE file for details

## Support

For issues or questions, please open an issue on the repository.
