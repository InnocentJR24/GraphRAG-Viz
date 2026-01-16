"""
Tests for the GraphRAG-Viz Glass Box pipeline.

Note: These tests require an OpenAI API key to be set in the environment.
For CI/CD, you may want to mock the LLM calls.
"""

import pytest
import os
from unittest.mock import Mock, patch

# Import components
from graphrag_viz.core.chunker import DocumentChunker
from graphrag_viz.core.graph_builder import GraphBuilder
from graphrag_viz.core.community_detector import CommunityDetector


class TestDocumentChunker:
    """Test the document chunking functionality."""
    
    def test_chunker_initialization(self):
        """Test that chunker initializes correctly."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
        assert chunker.chunk_size == 100
        assert chunker.chunk_overlap == 10
    
    def test_basic_chunking(self):
        """Test basic document chunking."""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        text = "This is a test document. " * 20  # Create longer text
        
        chunks = chunker.chunk_document(text, document_id="test_doc")
        
        assert len(chunks) > 0
        assert all("chunk_id" in c for c in chunks)
        assert all("text" in c for c in chunks)
        assert all("token_count" in c for c in chunks)
    
    def test_empty_document(self):
        """Test handling of empty documents."""
        chunker = DocumentChunker()
        chunks = chunker.chunk_document("", document_id="empty_doc")
        assert len(chunks) == 0
    
    def test_chunk_statistics(self):
        """Test chunk statistics generation."""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        text = "Test document. " * 30
        chunks = chunker.chunk_document(text, document_id="test")
        
        stats = chunker.get_chunk_statistics(chunks)
        
        assert "total_chunks" in stats
        assert "total_tokens" in stats
        assert "avg_tokens_per_chunk" in stats
        assert stats["total_chunks"] == len(chunks)


class TestGraphBuilder:
    """Test the graph building functionality."""
    
    def test_graph_initialization(self):
        """Test graph builder initialization."""
        builder = GraphBuilder()
        assert builder.graph is not None
        assert builder.graph.number_of_nodes() == 0
    
    def test_graph_construction(self):
        """Test building a graph from extraction results."""
        builder = GraphBuilder()
        
        # Mock extraction results
        extraction_results = [
            {
                "chunk_id": "chunk_0",
                "entities": [
                    {"name": "Alice", "type": "PERSON", "description": "A person", "entity_id": "e1"},
                    {"name": "Bob", "type": "PERSON", "description": "Another person", "entity_id": "e2"}
                ],
                "relationships": [
                    {"source": "Alice", "target": "Bob", "relationship": "knows", "description": "They know each other"}
                ]
            }
        ]
        
        graph = builder.build_graph(extraction_results)
        
        assert graph.number_of_nodes() == 2
        assert graph.number_of_edges() == 1
        assert "Alice" in graph.nodes()
        assert "Bob" in graph.nodes()
    
    def test_entity_merging(self):
        """Test that entities from different chunks are merged."""
        builder = GraphBuilder()
        
        extraction_results = [
            {
                "chunk_id": "chunk_0",
                "entities": [
                    {"name": "Alice", "type": "PERSON", "description": "A person", "entity_id": "e1"}
                ],
                "relationships": []
            },
            {
                "chunk_id": "chunk_1",
                "entities": [
                    {"name": "Alice", "type": "PERSON", "description": "Same person", "entity_id": "e2"}
                ],
                "relationships": []
            }
        ]
        
        graph = builder.build_graph(extraction_results)
        
        # Should have only one node for Alice despite appearing in two chunks
        assert graph.number_of_nodes() == 1
        assert "Alice" in graph.nodes()
        
        # Check that both source chunks are tracked
        source_chunks = graph.nodes["Alice"]["source_chunks"]
        assert len(source_chunks) == 2
    
    def test_graph_statistics(self):
        """Test graph statistics calculation."""
        builder = GraphBuilder()
        
        extraction_results = [
            {
                "chunk_id": "chunk_0",
                "entities": [
                    {"name": "A", "type": "PERSON", "description": "", "entity_id": "e1"},
                    {"name": "B", "type": "PERSON", "description": "", "entity_id": "e2"},
                    {"name": "C", "type": "PERSON", "description": "", "entity_id": "e3"}
                ],
                "relationships": [
                    {"source": "A", "target": "B", "relationship": "knows", "description": ""},
                    {"source": "B", "target": "C", "relationship": "knows", "description": ""}
                ]
            }
        ]
        
        graph = builder.build_graph(extraction_results)
        stats = builder.get_graph_statistics()
        
        assert stats["num_nodes"] == 3
        assert stats["num_edges"] == 2
        assert "avg_degree" in stats
        assert "density" in stats


class TestCommunityDetector:
    """Test community detection functionality."""
    
    def test_detector_initialization(self):
        """Test community detector initialization."""
        detector = CommunityDetector()
        assert detector.communities == {}
    
    def test_empty_graph(self):
        """Test detection on empty graph."""
        import networkx as nx
        detector = CommunityDetector()
        graph = nx.Graph()
        
        result = detector.detect_communities(graph)
        
        assert result["metadata"]["num_communities"] == 0
    
    def test_simple_community_detection(self):
        """Test community detection on a simple graph."""
        import networkx as nx
        detector = CommunityDetector()
        
        # Create a simple graph with two clear communities
        graph = nx.Graph()
        graph.add_edges_from([
            ("A", "B"), ("B", "C"), ("C", "A"),  # Community 1
            ("D", "E"), ("E", "F"), ("F", "D")   # Community 2
        ])
        
        # Add node attributes
        for node in graph.nodes():
            graph.nodes[node]["entity_type"] = "PERSON"
            graph.nodes[node]["description"] = f"Node {node}"
        
        result = detector.detect_communities(graph)
        
        assert result["metadata"]["num_communities"] > 0
        assert len(result["communities"]) > 0
        assert "modularity" in result["metadata"]


class TestConfiguration:
    """Test configuration handling."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        from graphrag_viz.config import PipelineConfig
        
        config = PipelineConfig()
        assert config.chunk_size > 0
        assert config.chunk_overlap >= 0
        assert config.chunk_overlap < config.chunk_size
    
    def test_config_validation(self):
        """Test configuration validation."""
        from graphrag_viz.config import PipelineConfig
        
        # Test invalid configuration
        with pytest.raises(ValueError):
            config = PipelineConfig(
                openai_api_key="",  # Empty API key
            )
            config.validate()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
