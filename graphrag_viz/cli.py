#!/usr/bin/env python3
"""
Command-line interface for GraphRAG-Viz Glass Box Pipeline.

Usage:
    graphrag_viz process <documents_dir> [--output OUTPUT]
    graphrag_viz query <query> [--pipeline-dir PIPELINE_DIR]
    graphrag_viz visualize <pipeline_dir> [--output OUTPUT]
"""

import argparse
import json
import os
import sys
from pathlib import Path

from graphrag_viz import GraphRAGPipeline, GraphVisualizer, PipelineConfig


def load_documents_from_dir(documents_dir: str):
    """Load all text documents from a directory."""
    documents = []
    doc_dir = Path(documents_dir)
    
    if not doc_dir.exists():
        print(f"Error: Directory {documents_dir} does not exist")
        sys.exit(1)
    
    for file_path in doc_dir.glob("**/*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            documents.append({
                "id": file_path.stem,
                "text": text
            })
    
    if not documents:
        print(f"Warning: No .txt files found in {documents_dir}")
    
    return documents


def process_command(args):
    """Process documents through the pipeline."""
    print("=" * 70)
    print("GraphRAG-Viz: Processing Documents")
    print("=" * 70)
    print()
    
    # Load documents
    print(f"Loading documents from: {args.documents_dir}")
    documents = load_documents_from_dir(args.documents_dir)
    print(f"Loaded {len(documents)} documents\n")
    
    # Initialize pipeline
    config = PipelineConfig()
    config.output_dir = args.output
    pipeline = GraphRAGPipeline(config)
    
    # Process documents
    print("Processing documents through pipeline...")
    results = pipeline.process_documents(documents)
    
    print("\n✓ Processing complete!")
    print(f"\nResults saved to: {args.output}")
    print(f"Pipeline ID: {results['pipeline_id']}")
    print(f"Entities: {results['graph_statistics']['num_nodes']}")
    print(f"Relationships: {results['graph_statistics']['num_edges']}")
    print(f"Communities: {results['community_structure']['num_communities']}")
    print()
    
    # Save pipeline state for later querying
    state_file = os.path.join(args.output, "pipeline_state.json")
    with open(state_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Pipeline state saved to: {state_file}")


def query_command(args):
    """Query the processed knowledge graph."""
    print("=" * 70)
    print("GraphRAG-Viz: Query")
    print("=" * 70)
    print()
    
    # Load pipeline state
    state_file = os.path.join(args.pipeline_dir, "pipeline_state.json")
    
    if not os.path.exists(state_file):
        print(f"Error: Pipeline state not found at {state_file}")
        print("Please run 'process' command first.")
        sys.exit(1)
    
    print(f"Loading pipeline from: {args.pipeline_dir}")
    
    # For a full implementation, we would need to serialize/deserialize
    # the entire pipeline state including the graph and models
    print("\nNote: Full query functionality requires loading complete pipeline state.")
    print("For now, please use the Python API for querying.")
    print("\nExample:")
    print('  from graphrag_viz import GraphRAGPipeline')
    print('  pipeline = GraphRAGPipeline()')
    print('  # ... process documents ...')
    print('  answer = pipeline.query("Your question")')


def visualize_command(args):
    """Create visualizations from pipeline results."""
    print("=" * 70)
    print("GraphRAG-Viz: Visualize")
    print("=" * 70)
    print()
    
    state_file = os.path.join(args.pipeline_dir, "pipeline_state.json")
    
    if not os.path.exists(state_file):
        print(f"Error: Pipeline state not found at {state_file}")
        sys.exit(1)
    
    print(f"Loading results from: {args.pipeline_dir}")
    
    with open(state_file, "r") as f:
        results = json.load(f)
    
    # Create visualizations
    visualizer = GraphVisualizer()
    
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nCreating visualizations in: {output_dir}")
    
    # Create pipeline summary
    summary_file = visualizer.create_pipeline_summary_visualization(
        pipeline_results=results,
        output_file=os.path.join(output_dir, "pipeline_summary.html")
    )
    print(f"✓ Pipeline summary: {summary_file}")
    
    print("\nNote: Full graph visualization requires loading the complete graph object.")
    print("For interactive visualizations, please use the Python API.")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GraphRAG-Viz: Glass Box GraphRAG Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process documents
  graphrag_viz process ./documents --output ./output
  
  # Query the graph (Python API recommended)
  python -c "from graphrag_viz import GraphRAGPipeline; ..."
  
  # Create visualizations
  graphrag_viz visualize ./output --output ./visualizations
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process documents through pipeline")
    process_parser.add_argument("documents_dir", help="Directory containing .txt documents")
    process_parser.add_argument("--output", default="output", help="Output directory (default: output)")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the knowledge graph")
    query_parser.add_argument("query", help="Question to ask")
    query_parser.add_argument("--pipeline-dir", default="output", help="Pipeline directory (default: output)")
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Create visualizations")
    viz_parser.add_argument("pipeline_dir", help="Pipeline directory with results")
    viz_parser.add_argument("--output", default="visualizations", help="Output directory (default: visualizations)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    if args.command == "process":
        process_command(args)
    elif args.command == "query":
        query_command(args)
    elif args.command == "visualize":
        visualize_command(args)


if __name__ == "__main__":
    main()
