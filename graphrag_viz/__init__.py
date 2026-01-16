"""
GraphRAG-Viz: A Glass Box Implementation of GraphRAG Pipeline
An Interpretability Framework for Global Query Summarization

This package provides a lightweight, transparent implementation of the GraphRAG
pipeline with full traceability and interpretability at every step.

Key Components:
- DocumentChunker: Transparent text segmentation
- EntityExtractor: LLM-based entity and relationship extraction
- GraphBuilder: Knowledge graph construction
- CommunityDetector: Graph clustering using Louvain algorithm
- CommunitySummarizer: LLM-based community summarization
- QueryEngine: Interpretable query processing
- GraphVisualizer: Interactive visualizations

Example Usage:
    >>> from graphrag_viz import GraphRAGPipeline
    >>> 
    >>> # Initialize pipeline
    >>> pipeline = GraphRAGPipeline()
    >>> 
    >>> # Process documents
    >>> documents = [{"id": "doc1", "text": "Your text here..."}]
    >>> results = pipeline.process_documents(documents)
    >>> 
    >>> # Query the knowledge graph
    >>> answer = pipeline.query("What are the main topics?")
    >>> print(answer["answer"])
"""

__version__ = "0.1.0"

from .pipeline import GraphRAGPipeline
from .config import PipelineConfig
from .core.chunker import DocumentChunker
from .core.extractor import EntityExtractor
from .core.graph_builder import GraphBuilder
from .core.community_detector import CommunityDetector
from .core.summarizer import CommunitySummarizer
from .core.query_engine import QueryEngine
from .visualization.visualizer import GraphVisualizer

__all__ = [
    "GraphRAGPipeline",
    "PipelineConfig",
    "DocumentChunker",
    "EntityExtractor",
    "GraphBuilder",
    "CommunityDetector",
    "CommunitySummarizer",
    "QueryEngine",
    "GraphVisualizer"
]
