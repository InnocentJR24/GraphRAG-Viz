"""
Main GraphRAG pipeline orchestrator.
Provides the Glass Box implementation with full interpretability.
"""
import logging
import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from .config import PipelineConfig
from .core.chunker import DocumentChunker
from .core.extractor import EntityExtractor
from .core.graph_builder import GraphBuilder
from .core.community_detector import CommunityDetector
from .core.summarizer import CommunitySummarizer
from .core.query_engine import QueryEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GraphRAGPipeline:
    """
    Glass Box GraphRAG Pipeline with complete transparency and interpretability.
    
    This pipeline processes documents through a transparent workflow:
    1. Document chunking
    2. Entity and relationship extraction
    3. Knowledge graph construction
    4. Community detection
    5. Community summarization
    6. Query processing
    
    Every step maintains full provenance for interpretability.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the GraphRAG pipeline.
        
        Args:
            config: Pipeline configuration (uses default if not provided)
        """
        self.config = config or PipelineConfig()
        
        # Initialize components
        self.chunker = DocumentChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            model=self.config.openai_model
        )
        
        self.extractor = EntityExtractor(
            api_key=self.config.openai_api_key,
            model=self.config.openai_model
        )
        
        self.graph_builder = GraphBuilder()
        self.community_detector = CommunityDetector()
        self.summarizer = CommunitySummarizer(
            api_key=self.config.openai_api_key,
            model=self.config.openai_model
        )
        
        self.query_engine = QueryEngine(
            api_key=self.config.openai_api_key,
            model=self.config.openai_model
        )
        
        # Pipeline state
        self.chunks = []
        self.extraction_results = []
        self.graph = None
        self.communities = None
        self.partition = None
        self.summaries = None
        
        # Execution trace for interpretability
        self.execution_trace = {
            "pipeline_id": f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "steps": [],
            "started_at": None,
            "completed_at": None
        }
        
        logger.info(f"Initialized GraphRAG Pipeline: {self.execution_trace['pipeline_id']}")
    
    def process_documents(self, documents: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Process documents through the complete pipeline.
        
        Args:
            documents: List of documents with 'id' and 'text' keys
            
        Returns:
            Dictionary with pipeline results and trace information
        """
        self.execution_trace["started_at"] = datetime.now().isoformat()
        logger.info(f"Starting pipeline processing for {len(documents)} documents")
        
        try:
            # Step 1: Chunk documents
            self._step_chunk_documents(documents)
            
            # Step 2: Extract entities and relationships
            self._step_extract_entities()
            
            # Step 3: Build knowledge graph
            self._step_build_graph()
            
            # Step 4: Detect communities
            self._step_detect_communities()
            
            # Step 5: Summarize communities
            self._step_summarize_communities()
            
            # Step 6: Initialize query engine
            self._step_initialize_query_engine()
            
            self.execution_trace["completed_at"] = datetime.now().isoformat()
            logger.info("Pipeline processing completed successfully")
            
            # Save results if configured
            if self.config.save_intermediate_results:
                self._save_results()
            
            return self._compile_results()
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            self.execution_trace["error"] = str(e)
            self.execution_trace["completed_at"] = datetime.now().isoformat()
            raise
    
    def _step_chunk_documents(self, documents: List[Dict[str, str]]):
        """Step 1: Chunk documents."""
        logger.info("Step 1: Chunking documents")
        step_start = datetime.now()
        
        self.chunks = []
        for doc in documents:
            doc_chunks = self.chunker.chunk_document(
                text=doc["text"],
                document_id=doc.get("id", f"doc_{len(self.chunks)}")
            )
            self.chunks.extend(doc_chunks)
        
        stats = self.chunker.get_chunk_statistics(self.chunks)
        
        self.execution_trace["steps"].append({
            "step": 1,
            "name": "Document Chunking",
            "duration_seconds": (datetime.now() - step_start).total_seconds(),
            "input": f"{len(documents)} documents",
            "output": f"{len(self.chunks)} chunks",
            "statistics": stats
        })
        
        logger.info(f"Created {len(self.chunks)} chunks")
    
    def _step_extract_entities(self):
        """Step 2: Extract entities and relationships."""
        logger.info("Step 2: Extracting entities and relationships")
        step_start = datetime.now()
        
        self.extraction_results = self.extractor.extract_from_chunks(self.chunks)
        
        total_entities = sum(len(r["entities"]) for r in self.extraction_results)
        total_relationships = sum(len(r["relationships"]) for r in self.extraction_results)
        
        self.execution_trace["steps"].append({
            "step": 2,
            "name": "Entity & Relationship Extraction",
            "duration_seconds": (datetime.now() - step_start).total_seconds(),
            "input": f"{len(self.chunks)} chunks",
            "output": f"{total_entities} entities, {total_relationships} relationships",
            "statistics": {
                "total_entities": total_entities,
                "total_relationships": total_relationships,
                "avg_entities_per_chunk": total_entities / len(self.chunks) if self.chunks else 0
            }
        })
        
        logger.info(f"Extracted {total_entities} entities and {total_relationships} relationships")
    
    def _step_build_graph(self):
        """Step 3: Build knowledge graph."""
        logger.info("Step 3: Building knowledge graph")
        step_start = datetime.now()
        
        self.graph = self.graph_builder.build_graph(self.extraction_results)
        stats = self.graph_builder.get_graph_statistics()
        
        self.execution_trace["steps"].append({
            "step": 3,
            "name": "Knowledge Graph Construction",
            "duration_seconds": (datetime.now() - step_start).total_seconds(),
            "input": f"{len(self.extraction_results)} extraction results",
            "output": f"{self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges",
            "statistics": stats
        })
        
        logger.info(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def _step_detect_communities(self):
        """Step 4: Detect communities."""
        logger.info("Step 4: Detecting communities")
        step_start = datetime.now()
        
        community_result = self.community_detector.detect_communities(self.graph)
        self.communities = community_result["communities"]
        self.partition = community_result["partition"]
        
        self.execution_trace["steps"].append({
            "step": 4,
            "name": "Community Detection",
            "duration_seconds": (datetime.now() - step_start).total_seconds(),
            "input": f"{self.graph.number_of_nodes()} nodes",
            "output": f"{len(self.communities)} communities",
            "statistics": community_result["metadata"]
        })
        
        logger.info(f"Detected {len(self.communities)} communities")
    
    def _step_summarize_communities(self):
        """Step 5: Summarize communities."""
        logger.info("Step 5: Summarizing communities")
        step_start = datetime.now()
        
        self.summaries = self.summarizer.summarize_communities(self.graph, self.communities)
        
        total_tokens = sum(s["metadata"].get("total_tokens", 0) for s in self.summaries.values())
        
        self.execution_trace["steps"].append({
            "step": 5,
            "name": "Community Summarization",
            "duration_seconds": (datetime.now() - step_start).total_seconds(),
            "input": f"{len(self.communities)} communities",
            "output": f"{len(self.summaries)} summaries",
            "statistics": {
                "total_tokens_used": total_tokens,
                "avg_tokens_per_summary": total_tokens / len(self.summaries) if self.summaries else 0
            }
        })
        
        logger.info(f"Generated {len(self.summaries)} community summaries")
    
    def _step_initialize_query_engine(self):
        """Step 6: Initialize query engine."""
        logger.info("Step 6: Initializing query engine")
        
        self.query_engine.initialize(
            graph=self.graph,
            communities=self.communities,
            partition=self.partition,
            summaries=self.summaries
        )
        
        logger.info("Query engine initialized")
    
    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Query the knowledge graph with full interpretability.
        
        Args:
            question: User's question
            top_k: Number of top communities to consider
            
        Returns:
            Dictionary with answer and complete provenance
        """
        if not self.query_engine or not self.summaries:
            raise RuntimeError("Pipeline must be processed before querying")
        
        result = self.query_engine.query(question, top_k)
        
        # Add full provenance tracing
        result["full_provenance"] = self.query_engine.trace_answer_provenance(result)
        
        return result
    
    def _compile_results(self) -> Dict[str, Any]:
        """Compile comprehensive pipeline results."""
        return {
            "pipeline_id": self.execution_trace["pipeline_id"],
            "execution_trace": self.execution_trace,
            "graph_statistics": self.graph_builder.get_graph_statistics(),
            "community_structure": self.community_detector.visualize_community_structure(),
            "summaries": self.summaries,
            "ready_for_queries": True
        }
    
    def _save_results(self):
        """Save intermediate results for later analysis."""
        output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        pipeline_id = self.execution_trace["pipeline_id"]
        
        # Save execution trace
        with open(f"{output_dir}/{pipeline_id}_trace.json", "w") as f:
            json.dump(self.execution_trace, f, indent=2)
        
        # Save graph statistics
        with open(f"{output_dir}/{pipeline_id}_graph_stats.json", "w") as f:
            json.dump(self.graph_builder.get_graph_statistics(), f, indent=2)
        
        # Save summaries
        summaries_serializable = {}
        for comm_id, summary in self.summaries.items():
            summaries_serializable[str(comm_id)] = {
                k: v for k, v in summary.items() if k != "metadata"
            }
        
        with open(f"{output_dir}/{pipeline_id}_summaries.json", "w") as f:
            json.dump(summaries_serializable, f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
