"""
Entity and relationship extraction module using LLM.
Provides transparent, traceable entity extraction with full provenance.
"""
import logging
import json
from typing import List, Dict, Any, Tuple
from openai import OpenAI

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Glass Box entity extractor using LLM with full traceability.
    
    Extracts entities and relationships from text chunks while maintaining
    complete provenance information for interpretability.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        logger.info(f"Initialized EntityExtractor with model: {model}")
    
    def extract_entities_and_relations(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract entities and relationships from a text chunk with full traceability.
        
        Args:
            chunk: Chunk dictionary with text and metadata
            
        Returns:
            Dictionary containing entities, relationships, and extraction metadata
        """
        chunk_id = chunk["chunk_id"]
        text = chunk["text"]
        
        logger.info(f"Extracting entities from chunk: {chunk_id}")
        
        # Create interpretable prompt
        prompt = self._create_extraction_prompt(text)
        
        try:
            # Call LLM for extraction
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting entities and relationships from text. Return results as valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Deterministic for reproducibility
                response_format={"type": "json_object"}
            )
            
            # Parse response
            result = json.loads(response.choices[0].message.content)
            
            # Add provenance and traceability metadata
            extraction_result = {
                "chunk_id": chunk_id,
                "entities": self._normalize_entities(result.get("entities", []), chunk_id),
                "relationships": self._normalize_relationships(result.get("relationships", []), chunk_id),
                "extraction_metadata": {
                    "model": self.model,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "source_chunk": chunk_id,
                    "source_text_preview": text[:100] + "..." if len(text) > 100 else text
                }
            }
            
            logger.info(f"Extracted {len(extraction_result['entities'])} entities and {len(extraction_result['relationships'])} relationships from {chunk_id}")
            
            return extraction_result
            
        except Exception as e:
            logger.error(f"Error extracting entities from {chunk_id}: {e}")
            return {
                "chunk_id": chunk_id,
                "entities": [],
                "relationships": [],
                "extraction_metadata": {
                    "error": str(e)
                }
            }
    
    def _create_extraction_prompt(self, text: str) -> str:
        """Create a clear, interpretable prompt for entity extraction."""
        return f"""Extract entities and relationships from the following text. 

For each entity, identify:
- name: The entity name
- type: Entity type (PERSON, ORGANIZATION, LOCATION, CONCEPT, EVENT, etc.)
- description: Brief description of the entity

For each relationship, identify:
- source: Source entity name
- target: Target entity name  
- relationship: Type of relationship (e.g., "works_for", "located_in", "related_to")
- description: Brief description of the relationship

Return the results as JSON with this structure:
{{
    "entities": [
        {{"name": "...", "type": "...", "description": "..."}},
        ...
    ],
    "relationships": [
        {{"source": "...", "target": "...", "relationship": "...", "description": "..."}},
        ...
    ]
}}

Text to analyze:
{text}
"""
    
    def _normalize_entities(self, entities: List[Dict], chunk_id: str) -> List[Dict[str, Any]]:
        """Add provenance information to entities."""
        normalized = []
        for idx, entity in enumerate(entities):
            normalized.append({
                "entity_id": f"{chunk_id}_entity_{idx}",
                "name": entity.get("name", "Unknown"),
                "type": entity.get("type", "UNKNOWN"),
                "description": entity.get("description", ""),
                "source_chunk": chunk_id
            })
        return normalized
    
    def _normalize_relationships(self, relationships: List[Dict], chunk_id: str) -> List[Dict[str, Any]]:
        """Add provenance information to relationships."""
        normalized = []
        for idx, rel in enumerate(relationships):
            normalized.append({
                "relationship_id": f"{chunk_id}_rel_{idx}",
                "source": rel.get("source", "Unknown"),
                "target": rel.get("target", "Unknown"),
                "relationship": rel.get("relationship", "related_to"),
                "description": rel.get("description", ""),
                "source_chunk": chunk_id
            })
        return normalized
    
    def extract_from_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract entities and relationships from multiple chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of extraction results with full provenance
        """
        logger.info(f"Starting entity extraction from {len(chunks)} chunks")
        results = []
        
        for chunk in chunks:
            result = self.extract_entities_and_relations(chunk)
            results.append(result)
        
        # Compute aggregate statistics for transparency
        total_entities = sum(len(r["entities"]) for r in results)
        total_relationships = sum(len(r["relationships"]) for r in results)
        
        logger.info(f"Extraction complete: {total_entities} entities, {total_relationships} relationships")
        
        return results
