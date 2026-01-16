"""
Community summarization module using LLM.
Generates interpretable summaries for each community in the knowledge graph.
"""
import logging
from typing import Dict, Any, List
import networkx as nx
from openai import OpenAI

logger = logging.getLogger(__name__)


class CommunitySummarizer:
    """
    Glass Box community summarizer with transparent summarization process.
    
    Generates summaries for graph communities while maintaining full provenance.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.summaries = {}
        logger.info(f"Initialized CommunitySummarizer with model: {model}")
    
    def summarize_communities(
        self, 
        graph: nx.Graph, 
        communities: Dict[int, List[str]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Generate summaries for all communities with full traceability.
        
        Args:
            graph: NetworkX graph
            communities: Dictionary mapping community IDs to node lists
            
        Returns:
            Dictionary mapping community IDs to summary information
        """
        logger.info(f"Summarizing {len(communities)} communities")
        
        self.summaries = {}
        
        for comm_id, nodes in communities.items():
            summary = self._summarize_community(graph, comm_id, nodes)
            self.summaries[comm_id] = summary
        
        logger.info(f"Completed summarization of {len(self.summaries)} communities")
        
        return self.summaries
    
    def _summarize_community(
        self, 
        graph: nx.Graph, 
        comm_id: int, 
        nodes: List[str]
    ) -> Dict[str, Any]:
        """
        Generate summary for a single community.
        
        Args:
            graph: NetworkX graph
            comm_id: Community ID
            nodes: List of nodes in the community
            
        Returns:
            Dictionary with summary and metadata
        """
        logger.info(f"Summarizing community {comm_id} with {len(nodes)} nodes")
        
        # Extract community information
        community_info = self._extract_community_info(graph, nodes)
        
        # Generate summary using LLM
        prompt = self._create_summary_prompt(community_info)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at creating concise, informative summaries of entity groups and their relationships."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            summary_text = response.choices[0].message.content
            
            return {
                "community_id": comm_id,
                "summary": summary_text,
                "num_entities": len(nodes),
                "entities": nodes,
                "key_entities": community_info["key_entities"],
                "entity_types": community_info["entity_types"],
                "num_relationships": community_info["num_relationships"],
                "metadata": {
                    "model": self.model,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Error summarizing community {comm_id}: {e}")
            return {
                "community_id": comm_id,
                "summary": f"Community with {len(nodes)} entities: {', '.join(nodes[:5])}{'...' if len(nodes) > 5 else ''}",
                "num_entities": len(nodes),
                "entities": nodes,
                "key_entities": nodes[:5],
                "entity_types": community_info["entity_types"],
                "num_relationships": community_info["num_relationships"],
                "metadata": {
                    "error": str(e)
                }
            }
    
    def _extract_community_info(self, graph: nx.Graph, nodes: List[str]) -> Dict[str, Any]:
        """
        Extract structured information about a community.
        
        Args:
            graph: NetworkX graph
            nodes: List of nodes in the community
            
        Returns:
            Dictionary with community information
        """
        # Get entity types
        entity_types = {}
        entity_descriptions = {}
        
        for node in nodes:
            entity_type = graph.nodes[node].get("entity_type", "UNKNOWN")
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            
            description = graph.nodes[node].get("description", "")
            if description:
                entity_descriptions[node] = description
        
        # Get relationships within community
        subgraph = graph.subgraph(nodes)
        relationships = []
        
        for u, v in subgraph.edges():
            rel_data = graph.edges[u, v]
            relationships.append({
                "source": u,
                "target": v,
                "relationship": rel_data.get("relationship", "related_to"),
                "description": rel_data.get("description", "")
            })
        
        # Identify key entities (highest degree)
        node_degrees = [(node, graph.degree(node)) for node in nodes]
        node_degrees.sort(key=lambda x: x[1], reverse=True)
        key_entities = [node for node, _ in node_degrees[:5]]
        
        return {
            "num_entities": len(nodes),
            "entity_types": entity_types,
            "entity_descriptions": entity_descriptions,
            "key_entities": key_entities,
            "num_relationships": len(relationships),
            "relationships": relationships[:10]  # Limit for prompt size
        }
    
    def _create_summary_prompt(self, community_info: Dict[str, Any]) -> str:
        """
        Create an interpretable prompt for community summarization.
        
        Args:
            community_info: Dictionary with community information
            
        Returns:
            Prompt string
        """
        entities_str = ", ".join(community_info["key_entities"][:10])
        
        relationships_str = ""
        for rel in community_info["relationships"][:5]:
            relationships_str += f"- {rel['source']} {rel['relationship']} {rel['target']}\n"
        
        entity_types_str = ", ".join([f"{k}: {v}" for k, v in community_info["entity_types"].items()])
        
        prompt = f"""Summarize the following community of related entities and relationships.

Community contains {community_info['num_entities']} entities with the following distribution:
{entity_types_str}

Key entities: {entities_str}

Sample relationships:
{relationships_str}

Provide a 2-3 sentence summary that explains:
1. The main theme or topic of this community
2. The key entities and their roles
3. The primary relationships between entities

Focus on being concise and informative."""

        return prompt
    
    def get_community_summary(self, comm_id: int) -> Dict[str, Any]:
        """
        Get the summary for a specific community.
        
        Args:
            comm_id: Community ID
            
        Returns:
            Dictionary with summary information
        """
        return self.summaries.get(comm_id, {"error": "Community not found"})
    
    def get_all_summaries(self) -> Dict[int, Dict[str, Any]]:
        """
        Get all community summaries.
        
        Returns:
            Dictionary of all summaries
        """
        return self.summaries
