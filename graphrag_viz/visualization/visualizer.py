"""
Visualization utilities for Glass Box GraphRAG pipeline.
Provides interpretable visualizations of the knowledge graph and communities.
"""
import logging
from typing import Dict, Any, List, Optional
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

logger = logging.getLogger(__name__)


class GraphVisualizer:
    """
    Glass Box graph visualizer for interpretable knowledge graph exploration.
    """
    
    def __init__(self):
        self.graph = None
        self.partition = None
        logger.info("Initialized GraphVisualizer")
    
    def visualize_graph(
        self,
        graph: nx.Graph,
        partition: Optional[Dict[str, int]] = None,
        output_file: str = "graph.html",
        show_labels: bool = True,
        height: str = "750px",
        width: str = "100%"
    ) -> str:
        """
        Create interactive visualization of the knowledge graph.
        
        Args:
            graph: NetworkX graph
            partition: Community assignments (optional)
            output_file: Output HTML file path
            show_labels: Whether to show node labels
            height: Height of the visualization
            width: Width of the visualization
            
        Returns:
            Path to the generated HTML file
        """
        logger.info(f"Creating interactive graph visualization: {output_file}")
        
        # Create pyvis network
        net = Network(height=height, width=width, notebook=False, directed=False)
        
        # Set physics options for better layout
        net.set_options("""
        {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 100,
                    "springConstant": 0.08
                },
                "maxVelocity": 50,
                "solver": "forceAtlas2Based",
                "timestep": 0.35,
                "stabilization": {"iterations": 150}
            }
        }
        """)
        
        # Color palette for communities
        colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8",
            "#F7DC6F", "#BB8FCE", "#85C1E2", "#F8B88B", "#AAB7B8"
        ]
        
        # Add nodes
        for node in graph.nodes():
            node_data = graph.nodes[node]
            entity_type = node_data.get("entity_type", "UNKNOWN")
            description = node_data.get("description", "")
            degree = graph.degree(node)
            
            # Determine color based on community
            if partition and node in partition:
                comm_id = partition[node]
                color = colors[comm_id % len(colors)]
            else:
                color = "#97C2FC"
            
            # Node size based on degree
            size = 10 + (degree * 2)
            
            # Create hover tooltip
            title = f"""
            <b>{node}</b><br>
            Type: {entity_type}<br>
            Description: {description[:100]}...<br>
            Degree: {degree}<br>
            Community: {partition.get(node, 'N/A') if partition else 'N/A'}
            """
            
            net.add_node(
                node,
                label=node if show_labels else "",
                title=title,
                color=color,
                size=size
            )
        
        # Add edges
        for source, target in graph.edges():
            edge_data = graph.edges[source, target]
            relationship = edge_data.get("relationship", "related_to")
            weight = edge_data.get("weight", 1)
            
            title = f"{relationship} (weight: {weight})"
            
            net.add_edge(source, target, title=title, width=weight)
        
        # Save to file
        net.save_graph(output_file)
        logger.info(f"Graph visualization saved to {output_file}")
        
        return output_file
    
    def visualize_community(
        self,
        graph: nx.Graph,
        community_nodes: List[str],
        community_id: int,
        output_file: str = "community.html"
    ) -> str:
        """
        Visualize a specific community in detail.
        
        Args:
            graph: NetworkX graph
            community_nodes: List of nodes in the community
            community_id: Community ID
            output_file: Output HTML file path
            
        Returns:
            Path to the generated HTML file
        """
        logger.info(f"Creating community {community_id} visualization")
        
        # Extract subgraph
        subgraph = graph.subgraph(community_nodes)
        
        # Create pyvis network
        net = Network(height="750px", width="100%", notebook=False, directed=False)
        
        # Add title
        net.heading = f"Community {community_id} - {len(community_nodes)} entities"
        
        # Add nodes with detailed information
        for node in subgraph.nodes():
            node_data = graph.nodes[node]
            entity_type = node_data.get("entity_type", "UNKNOWN")
            description = node_data.get("description", "")
            degree = subgraph.degree(node)
            
            title = f"""
            <b>{node}</b><br>
            Type: {entity_type}<br>
            Description: {description}<br>
            Degree in community: {degree}
            """
            
            size = 15 + (degree * 3)
            
            net.add_node(node, label=node, title=title, size=size)
        
        # Add edges
        for source, target in subgraph.edges():
            edge_data = graph.edges[source, target]
            relationship = edge_data.get("relationship", "related_to")
            
            net.add_edge(source, target, title=relationship)
        
        net.save_graph(output_file)
        logger.info(f"Community visualization saved to {output_file}")
        
        return output_file
    
    def plot_community_distribution(
        self,
        communities: Dict[int, List[str]],
        output_file: str = "community_distribution.png"
    ):
        """
        Plot community size distribution.
        
        Args:
            communities: Dictionary mapping community IDs to node lists
            output_file: Output PNG file path
        """
        logger.info("Plotting community distribution")
        
        # Get community sizes
        sizes = [len(nodes) for nodes in communities.values()]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(sizes)), sorted(sizes, reverse=True))
        plt.xlabel("Community Rank")
        plt.ylabel("Number of Entities")
        plt.title("Community Size Distribution")
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        avg_size = sum(sizes) / len(sizes)
        plt.axhline(y=avg_size, color='r', linestyle='--', label=f'Average: {avg_size:.1f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        logger.info(f"Community distribution plot saved to {output_file}")
    
    def plot_entity_type_distribution(
        self,
        graph: nx.Graph,
        output_file: str = "entity_types.png"
    ):
        """
        Plot distribution of entity types.
        
        Args:
            graph: NetworkX graph
            output_file: Output PNG file path
        """
        logger.info("Plotting entity type distribution")
        
        # Count entity types
        type_counts = {}
        for node in graph.nodes():
            entity_type = graph.nodes[node].get("entity_type", "UNKNOWN")
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        
        # Create plot
        plt.figure(figsize=(10, 6))
        types = list(type_counts.keys())
        counts = list(type_counts.values())
        
        plt.bar(types, counts)
        plt.xlabel("Entity Type")
        plt.ylabel("Count")
        plt.title("Entity Type Distribution")
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        logger.info(f"Entity type distribution plot saved to {output_file}")
    
    def create_pipeline_summary_visualization(
        self,
        pipeline_results: Dict[str, Any],
        output_file: str = "pipeline_summary.html"
    ) -> str:
        """
        Create an HTML summary of the entire pipeline execution.
        
        Args:
            pipeline_results: Results from pipeline execution
            output_file: Output HTML file path
            
        Returns:
            Path to the generated HTML file
        """
        logger.info("Creating pipeline summary visualization")
        
        trace = pipeline_results.get("execution_trace", {})
        graph_stats = pipeline_results.get("graph_statistics", {})
        community_structure = pipeline_results.get("community_structure", {})
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>GraphRAG Pipeline Summary</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; }}
                h1 {{ color: #333; border-bottom: 2px solid #4ECDC4; padding-bottom: 10px; }}
                h2 {{ color: #555; margin-top: 30px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background-color: #f0f0f0; border-radius: 5px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #4ECDC4; }}
                .metric-label {{ font-size: 12px; color: #666; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #4ECDC4; color: white; }}
                .step {{ margin: 10px 0; padding: 10px; background-color: #f9f9f9; border-left: 3px solid #4ECDC4; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔍 GraphRAG Glass Box Pipeline Summary</h1>
                <p><strong>Pipeline ID:</strong> {trace.get('pipeline_id', 'N/A')}</p>
                <p><strong>Started:</strong> {trace.get('started_at', 'N/A')}</p>
                <p><strong>Completed:</strong> {trace.get('completed_at', 'N/A')}</p>
                
                <h2>📊 Key Metrics</h2>
                <div>
                    <div class="metric">
                        <div class="metric-value">{graph_stats.get('num_nodes', 0)}</div>
                        <div class="metric-label">Entities</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{graph_stats.get('num_edges', 0)}</div>
                        <div class="metric-label">Relationships</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{community_structure.get('num_communities', 0)}</div>
                        <div class="metric-label">Communities</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{graph_stats.get('num_components', 0)}</div>
                        <div class="metric-label">Components</div>
                    </div>
                </div>
                
                <h2>🔄 Pipeline Execution Steps</h2>
                {''.join([f'''
                <div class="step">
                    <strong>Step {step['step']}: {step['name']}</strong><br>
                    Input: {step['input']}<br>
                    Output: {step['output']}<br>
                    Duration: {step['duration_seconds']:.2f}s
                </div>
                ''' for step in trace.get('steps', [])])}
                
                <h2>📈 Graph Statistics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Average Degree</td><td>{graph_stats.get('avg_degree', 0):.2f}</td></tr>
                    <tr><td>Graph Density</td><td>{graph_stats.get('density', 0):.4f}</td></tr>
                    <tr><td>Connected</td><td>{'Yes' if graph_stats.get('is_connected', False) else 'No'}</td></tr>
                </table>
                
                <h2>🏘️ Community Structure</h2>
                <p>The graph has been partitioned into <strong>{community_structure.get('num_communities', 0)}</strong> communities.</p>
                
                <p style="margin-top: 40px; color: #666; font-size: 12px; text-align: center;">
                    Generated by GraphRAG-Viz Glass Box Pipeline
                </p>
            </div>
        </body>
        </html>
        """
        
        with open(output_file, "w") as f:
            f.write(html_content)
        
        logger.info(f"Pipeline summary saved to {output_file}")
        return output_file
