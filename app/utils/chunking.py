"""
Text chunking utilities for document processing.

Provides functions to split text into overlapping chunks
suitable for RAG embedding and retrieval.
"""
import logging
from typing import List

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Handles edge cases:
    - Empty or whitespace-only text
    - Small text (smaller than chunk_size)
    - Invalid parameters

    Args:
        text: The text to chunk.
        chunk_size: Maximum characters per chunk (must be positive).
        overlap: Number of overlapping characters between chunks (must be non-negative).

    Returns:
        List of text chunks (non-empty, stripped).
        
    Raises:
        ValueError: If chunk_size <= 0 or overlap >= chunk_size.
    """
    # Validate parameters
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")
    
    # Handle empty or whitespace-only text
    if not text or not text.strip():
        logger.warning("Attempted to chunk empty or whitespace-only text")
        return []

    # Clean text
    text = text.strip()
    
    # Handle text smaller than chunk_size
    if len(text) <= chunk_size:
        logger.debug(f"Text length ({len(text)}) <= chunk_size ({chunk_size}), returning single chunk")
        return [text]

    chunks = []
    start = 0
    max_chunks = 1000  # Safety limit to prevent memory issues

    while start < len(text) and len(chunks) < max_chunks:
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        
        # Only add non-empty chunks
        if chunk:
            chunks.append(chunk)

        # Move start position, accounting for overlap
        start = end - overlap
        
        # Prevent infinite loop if overlap is too large
        if start <= 0:
            break
    
    # Log warning if we hit the chunk limit
    if len(chunks) >= max_chunks:
        logger.warning(f"Hit maximum chunk limit ({max_chunks}). Document may be truncated.")

    if not chunks:
        logger.warning("No chunks produced after processing text")
        return []
    
    logger.debug(f"Chunked text into {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")
    return chunks
