"""
Main FastAPI application for BOT GPT backend.

Provides:
- Application setup and configuration
- CORS middleware
- Database initialization
- Global exception handlers
- Health check and root endpoints
"""
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

# Load environment variables from .env file
load_dotenv()

from app.core.config import get_settings
from app.db.database import init_db
from app.api import conversations, documents, users

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifespan events.
    
    Startup: Initialize database
    Shutdown: Cleanup resources
    """
    # Startup
    try:
        logger.info("Starting BOT GPT Backend...")
        init_db()
        logger.info("Database initialized successfully")
        if settings.llm_configured:
            logger.info("LLM is configured and ready")
        else:
            logger.warning("LLM_API_KEY not configured - LLM calls will fail")
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down BOT GPT Backend...")
    logger.info("Cleanup complete")


# Create FastAPI app with lifespan
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Backend for BOT GPT conversational AI platform",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handlers
@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc: ValidationError):
    """Handle Pydantic validation errors."""
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Invalid request data"},
    )


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request, exc: RequestValidationError):
    """Handle FastAPI request validation errors."""
    logger.warning(f"Request validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Invalid request parameters"},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


# Include routers
app.include_router(users.router)
app.include_router(conversations.router)
app.include_router(documents.router)


# Health check endpoint
@app.get(
    "/health",
    summary="Health check",
    tags=["system"],
    responses={
        200: {"description": "Service is healthy"},
    },
)
async def health_check() -> dict:
    """
    Health check endpoint.
    
    Returns:
        Dict with status and configuration info.
    """
    return {
        "status": "ok",
        "version": settings.API_VERSION,
        "llm_configured": settings.llm_configured,
    }


@app.get(
    "/",
    summary="Root endpoint",
    tags=["system"],
)
async def root() -> dict:
    """
    Root endpoint with API information.
    
    Returns:
        Dict with welcome message and useful links.
    """
    return {
        "message": "Welcome to BOT GPT Backend",
        "version": settings.API_VERSION,
        "docs": "/docs",
        "openapi": "/openapi.json",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
    )
