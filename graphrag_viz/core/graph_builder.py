"""
Knowledge graph construction module.
Provides transparent graph building with full traceability.
"""
import logging
from typing import List, Dict, Any, Set
import networkx as nx

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Glass Box graph builder that creates interpretable knowledge graphs.
    
    Maintains full provenance of all nodes and edges for transparency.
    """
    
    def __init__(self):
        self.graph = nx.Graph()
        self.entity_metadata = {}
        self.edge_metadata = {}
        logger.info("Initialized GraphBuilder")
    
    def build_graph(self, extraction_results: List[Dict[str, Any]]) -> nx.Graph:
        """
        Build knowledge graph from extraction results with full traceability.
        
        Args:
            extraction_results: List of entity extraction results
            
        Returns:
            NetworkX graph with interpretable attributes
        """
        logger.info(f"Building graph from {len(extraction_results)} extraction results")
        
        # Clear previous graph
        self.graph.clear()
        self.entity_metadata.clear()
        self.edge_metadata.clear()
        
        # Process all entities first
        for result in extraction_results:
            chunk_id = result["chunk_id"]
            
            for entity in result["entities"]:
                self._add_entity(entity, chunk_id)
        
        # Then process all relationships
        for result in extraction_results:
            chunk_id = result["chunk_id"]
            
            for relationship in result["relationships"]:
                self._add_relationship(relationship, chunk_id)
        
        # Add graph-level metadata for interpretability
        self.graph.graph["metadata"] = {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_source_chunks": len(extraction_results),
            "is_connected": nx.is_connected(self.graph) if self.graph.number_of_nodes() > 0 else False,
            "num_components": nx.number_connected_components(self.graph) if self.graph.number_of_nodes() > 0 else 0
        }
        
        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def _add_entity(self, entity: Dict[str, Any], chunk_id: str):
        """Add entity as a node with provenance information."""
        entity_name = entity["name"]
        
        # If entity already exists, merge the information
        if self.graph.has_node(entity_name):
            # Add this chunk as another source
            self.graph.nodes[entity_name]["source_chunks"].add(chunk_id)
            
            # Update description if new one is more detailed
            existing_desc = self.graph.nodes[entity_name].get("description", "")
            new_desc = entity.get("description", "")
            if len(new_desc) > len(existing_desc):
                self.graph.nodes[entity_name]["description"] = new_desc
        else:
            # Add new entity node
            self.graph.add_node(
                entity_name,
                entity_type=entity.get("type", "UNKNOWN"),
                description=entity.get("description", ""),
                source_chunks={chunk_id},
                entity_id=entity.get("entity_id", f"{chunk_id}_entity")
            )
        
        # Store in metadata for quick access
        if entity_name not in self.entity_metadata:
            self.entity_metadata[entity_name] = {
                "type": entity.get("type", "UNKNOWN"),
                "descriptions": [],
                "source_chunks": set()
            }
        
        self.entity_metadata[entity_name]["descriptions"].append(entity.get("description", ""))
        self.entity_metadata[entity_name]["source_chunks"].add(chunk_id)
    
    def _add_relationship(self, relationship: Dict[str, Any], chunk_id: str):
        """Add relationship as an edge with provenance information."""
        source = relationship["source"]
        target = relationship["target"]
        rel_type = relationship.get("relationship", "related_to")
        
        # Only add edge if both nodes exist
        if not self.graph.has_node(source) or not self.graph.has_node(target):
            logger.debug(f"Skipping edge {source}->{target}: one or both nodes don't exist")
            return
        
        # Add or update edge
        if self.graph.has_edge(source, target):
            # Edge exists, add provenance
            self.graph.edges[source, target]["source_chunks"].add(chunk_id)
            self.graph.edges[source, target]["weight"] = self.graph.edges[source, target].get("weight", 1) + 1
        else:
            # Create new edge
            self.graph.add_edge(
                source,
                target,
                relationship=rel_type,
                description=relationship.get("description", ""),
                source_chunks={chunk_id},
                relationship_id=relationship.get("relationship_id", f"{chunk_id}_rel"),
                weight=1
            )
        
        # Store in metadata
        edge_key = (source, target)
        if edge_key not in self.edge_metadata:
            self.edge_metadata[edge_key] = {
                "relationship": rel_type,
                "descriptions": [],
                "source_chunks": set()
            }
        
        self.edge_metadata[edge_key]["descriptions"].append(relationship.get("description", ""))
        self.edge_metadata[edge_key]["source_chunks"].add(chunk_id)
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Provide comprehensive graph statistics for transparency.
        
        Returns:
            Dictionary with graph statistics
        """
        if self.graph.number_of_nodes() == 0:
            return {
                "num_nodes": 0,
                "num_edges": 0,
                "num_components": 0,
                "density": 0,
                "avg_degree": 0
            }
        
        degrees = [self.graph.degree(n) for n in self.graph.nodes()]
        
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_components": nx.number_connected_components(self.graph),
            "is_connected": nx.is_connected(self.graph),
            "density": nx.density(self.graph),
            "avg_degree": sum(degrees) / len(degrees),
            "max_degree": max(degrees) if degrees else 0,
            "min_degree": min(degrees) if degrees else 0,
            "entity_types": self._count_entity_types()
        }
    
    def _count_entity_types(self) -> Dict[str, int]:
        """Count entities by type for interpretability."""
        type_counts = {}
        for node in self.graph.nodes():
            entity_type = self.graph.nodes[node].get("entity_type", "UNKNOWN")
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        return type_counts
    
    def get_node_provenance(self, node_name: str) -> Dict[str, Any]:
        """
        Get complete provenance information for a node.
        
        Args:
            node_name: Name of the node
            
        Returns:
            Dictionary with provenance information
        """
        if not self.graph.has_node(node_name):
            return {"error": "Node not found"}
        
        node_data = self.graph.nodes[node_name]
        
        return {
            "name": node_name,
            "type": node_data.get("entity_type", "UNKNOWN"),
            "description": node_data.get("description", ""),
            "source_chunks": list(node_data.get("source_chunks", [])),
            "degree": self.graph.degree(node_name),
            "neighbors": list(self.graph.neighbors(node_name))
        }
