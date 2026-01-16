"""
Community detection module using Louvain algorithm.
Provides interpretable graph clustering for hierarchical summarization.
"""
import logging
from typing import Dict, Any, List, Set
import networkx as nx
import community as community_louvain

logger = logging.getLogger(__name__)


class CommunityDetector:
    """
    Glass Box community detector with transparent clustering process.
    
    Uses Louvain algorithm to partition the graph into interpretable communities.
    """
    
    def __init__(self):
        self.communities = {}
        self.community_metadata = {}
        logger.info("Initialized CommunityDetector")
    
    def detect_communities(self, graph: nx.Graph) -> Dict[str, Any]:
        """
        Detect communities in the graph with full traceability.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Dictionary with community assignments and metadata
        """
        if graph.number_of_nodes() == 0:
            logger.warning("Empty graph provided to community detection")
            return {"communities": {}, "metadata": {"num_communities": 0}}
        
        logger.info(f"Detecting communities in graph with {graph.number_of_nodes()} nodes")
        
        # Apply Louvain algorithm
        try:
            partition = community_louvain.best_partition(graph)
            
            # Organize nodes by community
            self.communities = {}
            for node, comm_id in partition.items():
                if comm_id not in self.communities:
                    self.communities[comm_id] = []
                self.communities[comm_id].append(node)
            
            # Calculate community metadata for interpretability
            self.community_metadata = self._calculate_community_metadata(graph, partition)
            
            logger.info(f"Detected {len(self.communities)} communities")
            
            return {
                "communities": self.communities,
                "partition": partition,
                "metadata": self.community_metadata
            }
            
        except Exception as e:
            logger.error(f"Error in community detection: {e}")
            # Fallback: each node is its own community
            partition = {node: i for i, node in enumerate(graph.nodes())}
            return {
                "communities": {i: [node] for i, node in enumerate(graph.nodes())},
                "partition": partition,
                "metadata": {"error": str(e), "num_communities": graph.number_of_nodes()}
            }
    
    def _calculate_community_metadata(self, graph: nx.Graph, partition: Dict) -> Dict[str, Any]:
        """
        Calculate comprehensive metadata for each community.
        
        Args:
            graph: NetworkX graph
            partition: Node to community mapping
            
        Returns:
            Dictionary with community metadata
        """
        metadata = {
            "num_communities": len(set(partition.values())),
            "modularity": community_louvain.modularity(partition, graph),
            "communities_detail": {}
        }
        
        # Organize by community
        comm_nodes = {}
        for node, comm_id in partition.items():
            if comm_id not in comm_nodes:
                comm_nodes[comm_id] = []
            comm_nodes[comm_id].append(node)
        
        # Calculate per-community metrics
        for comm_id, nodes in comm_nodes.items():
            subgraph = graph.subgraph(nodes)
            
            # Get entity types in this community
            entity_types = {}
            for node in nodes:
                entity_type = graph.nodes[node].get("entity_type", "UNKNOWN")
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            
            # Calculate internal vs external edges
            internal_edges = subgraph.number_of_edges()
            external_edges = 0
            for node in nodes:
                for neighbor in graph.neighbors(node):
                    if partition[neighbor] != comm_id:
                        external_edges += 1
            external_edges = external_edges // 2  # Each edge counted twice
            
            metadata["communities_detail"][comm_id] = {
                "size": len(nodes),
                "internal_edges": internal_edges,
                "external_edges": external_edges,
                "density": nx.density(subgraph) if len(nodes) > 1 else 0,
                "entity_types": entity_types,
                "key_nodes": self._get_key_nodes(graph, nodes, top_k=3)
            }
        
        return metadata
    
    def _get_key_nodes(self, graph: nx.Graph, nodes: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Identify key nodes in a community based on degree centrality.
        
        Args:
            graph: NetworkX graph
            nodes: List of nodes in the community
            top_k: Number of top nodes to return
            
        Returns:
            List of key node information
        """
        if not nodes:
            return []
        
        # Calculate degree within the community
        node_degrees = [(node, graph.degree(node)) for node in nodes]
        node_degrees.sort(key=lambda x: x[1], reverse=True)
        
        key_nodes = []
        for node, degree in node_degrees[:top_k]:
            key_nodes.append({
                "name": node,
                "degree": degree,
                "type": graph.nodes[node].get("entity_type", "UNKNOWN"),
                "description": graph.nodes[node].get("description", "")[:100]
            })
        
        return key_nodes
    
    def get_community_summary(self, comm_id: int) -> Dict[str, Any]:
        """
        Get interpretable summary of a specific community.
        
        Args:
            comm_id: Community ID
            
        Returns:
            Dictionary with community summary
        """
        if comm_id not in self.communities:
            return {"error": f"Community {comm_id} not found"}
        
        nodes = self.communities[comm_id]
        metadata = self.community_metadata.get("communities_detail", {}).get(comm_id, {})
        
        return {
            "community_id": comm_id,
            "size": len(nodes),
            "nodes": nodes,
            "key_nodes": metadata.get("key_nodes", []),
            "entity_types": metadata.get("entity_types", {}),
            "internal_edges": metadata.get("internal_edges", 0),
            "external_edges": metadata.get("external_edges", 0),
            "density": metadata.get("density", 0)
        }
    
    def get_node_community(self, node_name: str, partition: Dict) -> int:
        """
        Get the community ID for a specific node.
        
        Args:
            node_name: Name of the node
            partition: Node to community mapping
            
        Returns:
            Community ID
        """
        return partition.get(node_name, -1)
    
    def visualize_community_structure(self) -> Dict[str, Any]:
        """
        Generate data structure for visualizing community structure.
        
        Returns:
            Dictionary with visualization data
        """
        vis_data = {
            "num_communities": len(self.communities),
            "communities": []
        }
        
        for comm_id, nodes in self.communities.items():
            metadata = self.community_metadata.get("communities_detail", {}).get(comm_id, {})
            vis_data["communities"].append({
                "id": comm_id,
                "size": len(nodes),
                "key_nodes": metadata.get("key_nodes", []),
                "entity_distribution": metadata.get("entity_types", {})
            })
        
        return vis_data
