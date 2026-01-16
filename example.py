"""
Example usage of the GraphRAG-Viz Glass Box pipeline.

This example demonstrates how to:
1. Initialize the pipeline
2. Process documents
3. Query the knowledge graph
4. Visualize results
"""

from graphrag_viz import GraphRAGPipeline, GraphVisualizer
import os

# Sample documents for demonstration
SAMPLE_DOCUMENTS = [
    {
        "id": "doc1",
        "text": """
        Alice Johnson is a software engineer at TechCorp, a leading technology company based in San Francisco.
        She specializes in artificial intelligence and machine learning. Alice graduated from MIT with a degree
        in Computer Science. At TechCorp, she works on developing natural language processing systems and 
        collaborates closely with Bob Smith, who is the head of the AI research division.
        """
    },
    {
        "id": "doc2",
        "text": """
        TechCorp was founded in 2010 by Carol Williams and David Brown. The company is headquartered in 
        San Francisco, California, and has offices in New York, London, and Tokyo. TechCorp specializes in
        artificial intelligence solutions for enterprise clients. The company has grown to over 500 employees
        and is known for its innovative approach to machine learning and data analytics.
        """
    },
    {
        "id": "doc3",
        "text": """
        Bob Smith leads the AI research division at TechCorp. He has a PhD in Machine Learning from Stanford
        University and has published numerous papers on deep learning and neural networks. Bob's team focuses
        on developing cutting-edge AI technologies, including natural language processing, computer vision,
        and reinforcement learning. The team collaborates with universities and research institutions worldwide.
        """
    }
]


def main():
    """Run the example pipeline."""
    
    print("=" * 70)
    print("GraphRAG-Viz: Glass Box Pipeline Example")
    print("=" * 70)
    print()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set!")
        print("Please set your OpenAI API key in the .env file or environment variables.")
        print("Example: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Step 1: Initialize the pipeline
    print("Step 1: Initializing GraphRAG Pipeline...")
    pipeline = GraphRAGPipeline()
    print("✓ Pipeline initialized\n")
    
    # Step 2: Process documents
    print("Step 2: Processing documents through the pipeline...")
    print(f"   - Number of documents: {len(SAMPLE_DOCUMENTS)}")
    
    results = pipeline.process_documents(SAMPLE_DOCUMENTS)
    
    print("\n✓ Pipeline processing complete!")
    print(f"\n📊 Pipeline Results:")
    print(f"   - Pipeline ID: {results['pipeline_id']}")
    print(f"   - Total execution time: {sum(step['duration_seconds'] for step in results['execution_trace']['steps']):.2f}s")
    print(f"   - Entities discovered: {results['graph_statistics']['num_nodes']}")
    print(f"   - Relationships found: {results['graph_statistics']['num_edges']}")
    print(f"   - Communities detected: {results['community_structure']['num_communities']}")
    print()
    
    # Step 3: Display pipeline trace for transparency
    print("🔍 Pipeline Execution Trace (Glass Box Transparency):")
    for step in results['execution_trace']['steps']:
        print(f"   {step['step']}. {step['name']}")
        print(f"      Input: {step['input']}")
        print(f"      Output: {step['output']}")
        print(f"      Duration: {step['duration_seconds']:.2f}s")
    print()
    
    # Step 4: Query the knowledge graph
    print("Step 3: Querying the knowledge graph...")
    print()
    
    questions = [
        "Who works at TechCorp?",
        "What is Alice Johnson's role?",
        "Where is TechCorp located?"
    ]
    
    for question in questions:
        print(f"❓ Question: {question}")
        answer_result = pipeline.query(question, top_k=2)
        
        print(f"💡 Answer: {answer_result['answer']}")
        print(f"📍 Relevant Communities: {[c['community_id'] for c in answer_result['provenance']['relevant_communities']]}")
        print(f"🔗 Entities Referenced: {answer_result['provenance']['entities_referenced'][:5]}")
        print()
    
    # Step 5: Create visualizations
    print("Step 4: Creating visualizations...")
    
    visualizer = GraphVisualizer()
    
    # Create output directory
    os.makedirs("visualizations", exist_ok=True)
    
    # Visualize the full graph
    graph_file = visualizer.visualize_graph(
        graph=pipeline.graph,
        partition=pipeline.partition,
        output_file="visualizations/knowledge_graph.html"
    )
    print(f"✓ Knowledge graph visualization: {graph_file}")
    
    # Create pipeline summary
    summary_file = visualizer.create_pipeline_summary_visualization(
        pipeline_results=results,
        output_file="visualizations/pipeline_summary.html"
    )
    print(f"✓ Pipeline summary: {summary_file}")
    
    # Plot distributions
    visualizer.plot_community_distribution(
        communities=pipeline.communities,
        output_file="visualizations/community_distribution.png"
    )
    print(f"✓ Community distribution plot: visualizations/community_distribution.png")
    
    visualizer.plot_entity_type_distribution(
        graph=pipeline.graph,
        output_file="visualizations/entity_types.png"
    )
    print(f"✓ Entity type distribution: visualizations/entity_types.png")
    
    print()
    print("=" * 70)
    print("✅ Example completed successfully!")
    print("=" * 70)
    print()
    print("📁 Check the 'visualizations' folder for interactive visualizations")
    print("📁 Check the 'output' folder for detailed pipeline traces")
    print()
    print("🔍 Glass Box Transparency Features:")
    print("   - Full provenance tracking for all entities and relationships")
    print("   - Complete execution trace showing all pipeline steps")
    print("   - Interpretable query results with source attribution")
    print("   - Interactive visualizations showing graph structure")
    print()


if __name__ == "__main__":
    main()
