# GraphRAG-Viz: An Interpretability Framework for Global Query Summarization

**A "Glass Box" GraphRAG implementation featuring Hybrid Search, Asynchronous Indexing, and Visual Provenance.**

This repository implements the **GraphRAG-Viz** framework, a reproduction and extension of the Microsoft Research paper *[From Local to Global](https://arxiv.org/abs/2404.16130)* (Edge et al., 2024).

Unlike standard RAG, GraphRAG builds a hierarchical **Knowledge Graph** to answer high-level "global" queries (e.g., "What are the recurring themes?"). **GraphRAG-Viz** extends this by replacing the expensive map-reduce process with an optimized **Hybrid Search** strategy and providing an interactive visualization dashboard to trace the LLM's reasoning path back to specific graph communities.

## Key Features

* **Asynchronous Indexing:** Utilizes `asyncio` to parallelize entity extraction (Llama 3.2), achieving a ~3.5x speedup in graph construction.
* **Hybrid Global Search:** Replaces exhaustive map-reduce with a two-stage retrieval (Vector Pre-filtering + LLM Reranking) to reduce query latency from >45s to <8s.
* **Entity Resolution:** Includes a post-processing step using `difflib` to merge duplicate nodes (e.g., "Bill Gates" vs. "Mr. Gates") for a cleaner topology.
* **Glass Box Visualization:** An interactive Streamlit dashboard that visualizes the "Reasoning Trace," allowing users to verify exactly which communities influenced the final answer.

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/InnocentJR24/GraphRAG-Viz.git
cd graphrag-viz

```

### 2. Install Dependencies

Install the required Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt

```

### 3. Set Up Ollama

Download and install [Ollama](https://ollama.com/), then pull the required LLM and embedding models:

```bash
# Pull the main LLM
ollama pull llama3.2

# Pull the embedding model for Hybrid Search
ollama pull nomic-embed-text

```

### 4. Prepare Your Data

The system is validated using *Alice's Adventures in Wonderland*. Place your source text file in `data/raw/book.txt` or update the path in `src/config.py`.

### 5. Run the Indexing Pipeline

Build the knowledge graph. This process includes async entity extraction, entity resolution, community detection, and summary generation.

```bash
python run_pipeline.py

```

*Note: This will also pre-compute vector embeddings for community summaries to enable Hybrid Search.*

### 6. Launch the Visualization & Search UI

Start the Streamlit application to query your knowledge graph and visualize the reasoning traces:

```bash
streamlit run app/main.py

```

### 7. Preview

![viz-demo](https://github.com/user-attachments/assets/2554f031-d485-4f40-8610-52f19545be99)

## Methodology

This project sits between standard Vector RAG and the original GraphRAG implementation.

### **Tech Stack**

* **LLM:** Llama 3.2 (via Ollama)
* **Embeddings:** `nomic-embed-text`
* **Orchestration:** LangChain & AsyncIO
* **Graph Logic:** NetworkX & CDLib (Leiden Algorithm)
* **Visualization:** Streamlit & PyVis

### **Pipeline Architecture**

1. **Asynchronous Graph Construction:**
* Text is chunked and processed in parallel using a semaphore-controlled async loop.
* Entities and relationships are extracted into structured objects.
* **Entity Resolution:** Nodes with high string similarity (>0.9) are merged to densify the graph.


2. **Hierarchical Summarization:**
* The **Leiden algorithm** detects communities within the graph.
* The LLM generates natural language summaries for every community.
* Embeddings for these summaries are pre-computed and stored.


3. **Hybrid Global Search (The Extension):**
* **Stage 1 (Broad):** Vector search retrieves the top- (e.g., 15) community summaries most similar to the user's query.
* **Stage 2 (Deep):** The LLM reranks these candidates, assigning relevance scores (0-10) and generating reasoning arguments.
* **Stage 3 (Synthesize):** High-scoring summaries are concatenated to generate the final answer.


4. **Interpretability:**
* The UI maps the generated answer back to the specific source communities, highlighting the subgraph that provided the context.



## Configuration

Edit `src/config.py` to customize:

* **LLM Settings:** Model names (`llama3.2`, `nomic-embed-text`) and concurrency limits.
* **Graph Settings:** Chunk size (default: 2000), overlap (200), and Leiden resolution.
* **Search Parameters:** The top- for vector retrieval and the relevance threshold for reranking.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

This implementation reproduces and extends the GraphRAG methodology by **Microsoft Research**.

**Reference:** Edge, D., et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.* arXiv:2404.16130.

**Data Source:** *Alice's Adventures in Wonderland* by Lewis Carroll, sourced from [Project Gutenberg](https://www.gutenberg.org/ebooks/11).
