import networkx as nx
import logging
import asyncio
import difflib
from typing import List, Optional, Tuple
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from .config import MODEL_NAME, GRAPH_FILE

logger = logging.getLogger(__name__)

class EntityRelation(BaseModel):
    source: str = Field(description="The source entity")
    target: str = Field(description="The target entity")
    relationship: str = Field(description="The relationship between them")
    description: Optional[str] = Field(description="Brief context")

class GraphExtraction(BaseModel):
    relations: List[EntityRelation]

class GraphBuilder:
    def __init__(self):
        self.llm = ChatOllama(model=MODEL_NAME, temperature=0)
        self.graph = nx.Graph()
        
        self.prompt = PromptTemplate(
            template="""
            Extract key entities (people, orgs, places) and relationships.
            Return a JSON list.
            
            Text: {text}
            """,
            input_variables=["text"],
        )
        self.chain = self.prompt | self.llm.with_structured_output(GraphExtraction)

    async def process_documents_async(self, text_chunks: List[str], concurrency=4):
        """Extracts entities in parallel using a semaphore."""
        logger.info(f"Starting async extraction on {len(text_chunks)} chunks (Concurrency: {concurrency})...")
        semaphore = asyncio.Semaphore(concurrency)

        async def _process_chunk(chunk_id, text):
            async with semaphore:
                try:
                    extraction = await self.chain.ainvoke({"text": text})
                    if extraction and extraction.relations:
                        return extraction.relations
                except Exception as e:
                    logger.error(f"Chunk {chunk_id} failed: {e}")
                return []

        tasks = [_process_chunk(i, chunk) for i, chunk in enumerate(text_chunks)]
        results = await asyncio.gather(*tasks)

        count = 0
        for batch in results:
            if batch:
                for item in batch:
                    self._update_graph(item)
                count += 1
        logger.info(f"Graph built. Processed {count}/{len(text_chunks)} chunks successfully.")

    def _update_graph(self, item: EntityRelation):
        src = item.source.strip()
        tgt = item.target.strip()
        
        if src == tgt: return 

        self.graph.add_node(src, type="entity")
        self.graph.add_node(tgt, type="entity")
        
        if self.graph.has_edge(src, tgt):
            self.graph[src][tgt]['weight'] += 1
            curr_desc = self.graph[src][tgt]['description']
            if item.description and item.description not in curr_desc:
                 self.graph[src][tgt]['description'] += f"; {item.description}"
        else:
            self.graph.add_edge(
                src, tgt, 
                weight=1, 
                relationship=item.relationship,
                description=item.description or ""
            )

    def resolve_entities(self, similarity_threshold=0.9):
        """Merges nodes that are textually similar (e.g., 'Bill Gates' & 'Mr. Gates')."""
        logger.info("Resolving entities...")
        nodes = list(self.graph.nodes())
        nodes.sort(key=len, reverse=True)
        
        mapping = {}
        processed = set()

        for i, node_a in enumerate(nodes):
            if node_a in processed: continue
            
            for node_b in nodes[i+1:]:
                if node_b in processed: continue

                ratio = difflib.SequenceMatcher(None, node_a.lower(), node_b.lower()).ratio()
                
                is_substring = (len(node_b) > 4 and node_b in node_a)
                
                if ratio > similarity_threshold or is_substring:
                    mapping[node_b] = node_a
                    processed.add(node_b)
        
        if mapping:
            logger.info(f"Merging {len(mapping)} duplicate entities...")
            self.graph = nx.relabel_nodes(self.graph, mapping)
            
            self.graph.remove_edges_from(nx.selfloop_edges(self.graph))
        else:
            logger.info("No duplicates found.")

    def save_graph(self):
        nx.write_gexf(self.graph, GRAPH_FILE)
        logger.info(f"Graph saved to {GRAPH_FILE} (Nodes: {self.graph.number_of_nodes()})")