"""
Query engine for interpretable information retrieval.
Provides transparent query processing with full provenance tracing.
"""
import logging
from typing import Dict, Any, List
import networkx as nx
from openai import OpenAI

logger = logging.getLogger(__name__)


class QueryEngine:
    """
    Glass Box query engine with interpretable query processing.
    
    Processes queries by finding relevant communities and generating
    answers with complete traceability.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.graph = None
        self.communities = None
        self.partition = None
        self.summaries = None
        logger.info(f"Initialized QueryEngine with model: {model}")
    
    def initialize(
        self,
        graph: nx.Graph,
        communities: Dict[int, List[str]],
        partition: Dict[str, int],
        summaries: Dict[int, Dict[str, Any]]
    ):
        """
        Initialize the query engine with graph and community data.
        
        Args:
            graph: NetworkX graph
            communities: Dictionary mapping community IDs to nodes
            partition: Dictionary mapping nodes to community IDs
            summaries: Dictionary of community summaries
        """
        self.graph = graph
        self.communities = communities
        self.partition = partition
        self.summaries = summaries
        logger.info(f"Query engine initialized with {len(communities)} communities")
    
    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Process a query with full interpretability.
        
        Args:
            question: User's question
            top_k: Number of top communities to consider
            
        Returns:
            Dictionary with answer and complete trace information
        """
        logger.info(f"Processing query: {question}")
        
        if not self.graph or not self.communities:
            return {"error": "Query engine not initialized"}
        
        # Step 1: Identify relevant communities
        relevant_communities = self._find_relevant_communities(question, top_k)
        
        # Step 2: Gather context from relevant communities
        context = self._gather_context(relevant_communities)
        
        # Step 3: Generate answer using LLM
        answer_result = self._generate_answer(question, context)
        
        # Step 4: Compile interpretable response
        response = {
            "question": question,
            "answer": answer_result["answer"],
            "provenance": {
                "relevant_communities": relevant_communities,
                "context_used": context,
                "entities_referenced": self._extract_entities_from_context(context),
                "model": self.model,
                "tokens_used": answer_result.get("tokens_used", 0)
            },
            "trace": {
                "step_1": "Identified relevant communities based on query",
                "step_2": "Gathered context from community summaries",
                "step_3": "Generated answer using gathered context",
                "interpretability": "All entities and relationships can be traced back to source chunks"
            }
        }
        
        logger.info(f"Query processed successfully. Found {len(relevant_communities)} relevant communities")
        
        return response
    
    def _find_relevant_communities(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Find communities most relevant to the query.
        
        Args:
            question: User's question
            top_k: Number of communities to return
            
        Returns:
            List of relevant community information
        """
        # Use LLM to score community relevance
        community_scores = []
        
        for comm_id, summary_data in self.summaries.items():
            score = self._score_community_relevance(question, summary_data)
            community_scores.append({
                "community_id": comm_id,
                "score": score,
                "summary": summary_data.get("summary", ""),
                "num_entities": summary_data.get("num_entities", 0),
                "key_entities": summary_data.get("key_entities", [])
            })
        
        # Sort by relevance score
        community_scores.sort(key=lambda x: x["score"], reverse=True)
        
        return community_scores[:top_k]
    
    def _score_community_relevance(self, question: str, summary_data: Dict[str, Any]) -> float:
        """
        Score community relevance to the query.
        
        Args:
            question: User's question
            summary_data: Community summary data
            
        Returns:
            Relevance score (0-1)
        """
        # Simple keyword-based scoring for lightweight implementation
        # In production, would use embeddings for better accuracy
        
        question_lower = question.lower()
        summary_lower = summary_data.get("summary", "").lower()
        
        # Count keyword matches
        question_words = set(question_lower.split())
        summary_words = set(summary_lower.split())
        
        # Remove common words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "and", "or"}
        question_words = question_words - stop_words
        summary_words = summary_words - stop_words
        
        # Calculate overlap
        if not question_words:
            return 0.0
        
        overlap = len(question_words & summary_words)
        score = overlap / len(question_words)
        
        # Boost score if entities are mentioned
        entities = summary_data.get("key_entities", [])
        for entity in entities:
            if entity.lower() in question_lower:
                score += 0.3
        
        return min(score, 1.0)
    
    def _gather_context(self, relevant_communities: List[Dict[str, Any]]) -> str:
        """
        Gather context from relevant communities.
        
        Args:
            relevant_communities: List of relevant community data
            
        Returns:
            Context string for answer generation
        """
        context_parts = []
        
        for comm_info in relevant_communities:
            comm_id = comm_info["community_id"]
            summary = comm_info["summary"]
            entities = ", ".join(comm_info["key_entities"][:5])
            
            context_parts.append(
                f"Community {comm_id} (Score: {comm_info['score']:.2f}):\n"
                f"Summary: {summary}\n"
                f"Key entities: {entities}\n"
            )
        
        return "\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> Dict[str, Any]:
        """
        Generate answer using LLM with gathered context.
        
        Args:
            question: User's question
            context: Context from relevant communities
            
        Returns:
            Dictionary with answer and metadata
        """
        prompt = f"""Based on the following context from a knowledge graph, answer the question.
Be specific and reference entities from the context when relevant.
If the context doesn't contain enough information, acknowledge this.

Context:
{context}

Question: {question}

Answer:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on structured knowledge graph data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "tokens_used": response.usage.total_tokens,
                "model": self.model
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "tokens_used": 0,
                "error": str(e)
            }
    
    def _extract_entities_from_context(self, context: str) -> List[str]:
        """
        Extract entity names mentioned in the context.
        
        Args:
            context: Context string
            
        Returns:
            List of entity names
        """
        entities = []
        
        # Extract entities mentioned in context
        for node in self.graph.nodes():
            if node.lower() in context.lower():
                entities.append(node)
        
        return entities[:10]  # Limit to top 10
    
    def trace_answer_provenance(self, answer_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trace the complete provenance of an answer back to source chunks.
        
        Args:
            answer_response: Response from query method
            
        Returns:
            Dictionary with complete provenance information
        """
        entities = answer_response["provenance"]["entities_referenced"]
        
        provenance = {
            "entities_used": [],
            "source_chunks": set(),
            "communities_involved": []
        }
        
        for entity in entities:
            if self.graph.has_node(entity):
                node_data = self.graph.nodes[entity]
                source_chunks = node_data.get("source_chunks", set())
                
                provenance["entities_used"].append({
                    "name": entity,
                    "type": node_data.get("entity_type", "UNKNOWN"),
                    "description": node_data.get("description", ""),
                    "source_chunks": list(source_chunks)
                })
                
                provenance["source_chunks"].update(source_chunks)
                
                # Get community
                if entity in self.partition:
                    comm_id = self.partition[entity]
                    if comm_id not in provenance["communities_involved"]:
                        provenance["communities_involved"].append(comm_id)
        
        provenance["source_chunks"] = list(provenance["source_chunks"])
        
        return provenance
