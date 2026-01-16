# GraphRAG-Viz 🔍

**A Glass Box Implementation of GraphRAG Pipeline**

An Interpretability Framework for Global Query Summarization with complete transparency and traceability.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

GraphRAG-Viz is a lightweight, transparent implementation of the GraphRAG (Graph Retrieval-Augmented Generation) pipeline. Unlike traditional "black box" RAG systems, this implementation provides **complete interpretability** at every stage, allowing you to understand exactly how your knowledge graph is constructed and how queries are answered.

### What is GraphRAG?

GraphRAG enhances traditional RAG by constructing a knowledge graph from documents, detecting communities of related entities, and using hierarchical summarization to answer complex queries. This approach enables better handling of global questions that require understanding relationships across multiple documents.

### What Makes This a "Glass Box" Implementation?

- ✅ **Full Provenance Tracking**: Every entity and relationship traces back to its source text
- ✅ **Transparent Processing**: Complete execution traces showing all pipeline steps
- ✅ **Interpretable Results**: Query answers include full context and source attribution
- ✅ **Interactive Visualizations**: Explore your knowledge graph structure visually
- ✅ **Auditable Decisions**: Understand why communities were formed and how answers were generated

## 🌟 Key Features

- **📄 Document Processing**: Transparent text chunking with overlap tracking
- **🔍 Entity Extraction**: LLM-based extraction with full metadata
- **🕸️ Graph Construction**: Knowledge graph building with provenance
- **🏘️ Community Detection**: Louvain-based clustering with statistics
- **📝 Summarization**: LLM-powered community summaries
- **❓ Query Engine**: Interpretable query processing with trace information
- **📊 Visualization**: Interactive HTML visualizations and plots

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/InnocentJR24/GraphRAG-Viz.git
cd GraphRAG-Viz
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Basic Usage

```python
from graphrag_viz import GraphRAGPipeline

# Initialize the pipeline
pipeline = GraphRAGPipeline()

# Prepare your documents
documents = [
    {"id": "doc1", "text": "Your document text here..."},
    {"id": "doc2", "text": "Another document..."}
]

# Process documents through the pipeline
results = pipeline.process_documents(documents)

# Query the knowledge graph
answer = pipeline.query("What are the main topics discussed?")
print(answer["answer"])

# Access full provenance
print(answer["provenance"])
```

### Run the Example

```bash
python example.py
```

This will:
1. Process sample documents
2. Build a knowledge graph
3. Answer example queries
4. Generate interactive visualizations

## 📚 Pipeline Architecture

The Glass Box pipeline consists of six transparent stages:

```
Documents → Chunking → Entity Extraction → Graph Building → 
Community Detection → Summarization → Query Processing
```

### Stage 1: Document Chunking
- Splits documents into overlapping chunks
- Tracks token counts and overlap regions
- Maintains document-to-chunk provenance

### Stage 2: Entity & Relationship Extraction
- Uses LLM to extract entities and their relationships
- Tags entity types (PERSON, ORGANIZATION, LOCATION, etc.)
- Preserves source chunk references

### Stage 3: Knowledge Graph Construction
- Builds NetworkX graph from extracted entities
- Merges duplicate entities across chunks
- Tracks all source chunks for each node and edge

### Stage 4: Community Detection
- Applies Louvain algorithm for clustering
- Calculates community statistics (size, density, modularity)
- Identifies key nodes within each community

### Stage 5: Community Summarization
- Generates LLM-based summaries for each community
- Maintains entity and relationship metadata
- Tracks token usage for transparency

### Stage 6: Query Processing
- Finds relevant communities for each query
- Gathers context from community summaries
- Generates answers with full provenance tracing

## 🔧 Configuration

Configure the pipeline through `PipelineConfig` or environment variables:

```python
from graphrag_viz import PipelineConfig, GraphRAGPipeline

config = PipelineConfig(
    openai_api_key="your-key",
    openai_model="gpt-3.5-turbo",
    chunk_size=500,
    chunk_overlap=50,
    max_communities=10,
    enable_logging=True,
    save_intermediate_results=True,
    output_dir="output"
)

pipeline = GraphRAGPipeline(config)
```

### Environment Variables

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
CHUNK_SIZE=500
CHUNK_OVERLAP=50
MAX_COMMUNITIES=10
```

## 📊 Visualization

The package includes comprehensive visualization tools:

```python
from graphrag_viz import GraphVisualizer

visualizer = GraphVisualizer()

# Create interactive graph visualization
visualizer.visualize_graph(
    graph=pipeline.graph,
    partition=pipeline.partition,
    output_file="graph.html"
)

# Generate pipeline summary
visualizer.create_pipeline_summary_visualization(
    pipeline_results=results,
    output_file="summary.html"
)

# Plot distributions
visualizer.plot_community_distribution(
    communities=pipeline.communities,
    output_file="communities.png"
)

visualizer.plot_entity_type_distribution(
    graph=pipeline.graph,
    output_file="entity_types.png"
)
```

## 🧪 Testing

Run tests (if available):
```bash
pytest tests/
```

## 📖 Documentation

### Core Components

- **`DocumentChunker`**: Handles text segmentation with overlap
- **`EntityExtractor`**: Extracts entities and relationships using LLM
- **`GraphBuilder`**: Constructs knowledge graph with provenance
- **`CommunityDetector`**: Detects communities using Louvain algorithm
- **`CommunitySummarizer`**: Generates community summaries
- **`QueryEngine`**: Processes queries with interpretability
- **`GraphVisualizer`**: Creates interactive visualizations

### API Reference

See inline documentation in each module for detailed API information.

## 🎓 Use Cases

- **Research**: Analyze large document collections with transparency
- **Knowledge Management**: Build interpretable knowledge bases
- **Education**: Teach GraphRAG concepts with visible internals
- **Debugging**: Understand RAG pipeline behavior in detail
- **Compliance**: Audit AI decisions with full provenance

## 🤝 Contributing

Contributions are welcome! This is a lightweight implementation focused on interpretability. Please maintain the Glass Box philosophy:

1. All processing must be traceable
2. Maintain provenance information
3. Provide clear logging and statistics
4. Keep dependencies minimal

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

This implementation is inspired by:
- Microsoft's GraphRAG research and implementation
- The RAG community's work on interpretability
- NetworkX and Louvain community detection algorithms

## 🔗 Related Resources

- [GraphRAG Paper (arXiv)](https://arxiv.org/pdf/2404.16130)
- [Microsoft GraphRAG](https://github.com/microsoft/graphrag)
- [NetworkX Documentation](https://networkx.org/)

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with transparency in mind. Every decision is traceable. Every result is explainable.**
