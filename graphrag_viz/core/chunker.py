"""
Text chunking module for document processing.
Implements transparent, interpretable document chunking with overlap.
"""
import logging
from typing import List, Dict, Any
import tiktoken

logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Glass Box document chunker that provides transparent text segmentation.
    
    Attributes:
        chunk_size: Maximum tokens per chunk
        chunk_overlap: Number of overlapping tokens between chunks
        encoding: Tokenizer encoding for the model
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, model: str = "gpt-3.5-turbo"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning(f"Model {model} not found, using cl100k_base encoding")
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        logger.info(f"Initialized DocumentChunker: chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def chunk_document(self, text: str, document_id: str = "doc_0") -> List[Dict[str, Any]]:
        """
        Split document into overlapping chunks with full traceability.
        
        Args:
            text: Input document text
            document_id: Unique identifier for the document
            
        Returns:
            List of chunk dictionaries with metadata for interpretability
        """
        if not text:
            logger.warning("Empty text provided to chunker")
            return []
        
        # Tokenize the entire document
        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)
        
        logger.info(f"Chunking document {document_id}: {total_tokens} tokens")
        
        chunks = []
        chunk_id = 0
        start_idx = 0
        
        while start_idx < total_tokens:
            # Calculate end index for this chunk
            end_idx = min(start_idx + self.chunk_size, total_tokens)
            
            # Extract chunk tokens and decode
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Create interpretable chunk metadata
            chunk_info = {
                "chunk_id": f"{document_id}_chunk_{chunk_id}",
                "document_id": document_id,
                "text": chunk_text,
                "start_token": start_idx,
                "end_token": end_idx,
                "token_count": len(chunk_tokens),
                "char_count": len(chunk_text),
                "overlap_with_previous": min(self.chunk_overlap, start_idx),
                "overlap_with_next": self.chunk_overlap if end_idx < total_tokens else 0
            }
            
            chunks.append(chunk_info)
            chunk_id += 1
            
            # Move to next chunk with overlap
            start_idx = end_idx - self.chunk_overlap if end_idx < total_tokens else end_idx
            
            # Prevent infinite loop
            if start_idx >= end_idx:
                break
        
        logger.info(f"Created {len(chunks)} chunks from document {document_id}")
        return chunks
    
    def get_chunk_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Provide statistical summary of chunking process for transparency.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Dictionary with chunking statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_tokens": 0,
                "avg_tokens_per_chunk": 0,
                "min_tokens": 0,
                "max_tokens": 0
            }
        
        token_counts = [c["token_count"] for c in chunks]
        
        return {
            "total_chunks": len(chunks),
            "total_tokens": sum(token_counts),
            "avg_tokens_per_chunk": sum(token_counts) / len(chunks),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "total_overlap_tokens": sum(c["overlap_with_previous"] for c in chunks)
        }
