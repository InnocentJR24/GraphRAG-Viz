import networkx as nx
import json
import logging
from cdlib import algorithms
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .config import MODEL_NAME, COMMUNITY_SUMMARIES, NODE_MAP

logger = logging.getLogger(__name__)

class CommunitySummarizer:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.llm = ChatOllama(model=MODEL_NAME, temperature=0)
        self.communities = {}
        self.summaries = {}     

        self.summary_prompt = PromptTemplate(
            template="""
            Analyze the following community of entities:
            Entities: {entities}
            Relationships: {relationships}
            
            Write a comprehensive summary identifying the common theme or narrative.
            """,
            input_variables=["entities", "relationships"]
        )
        self.chain = self.summary_prompt | self.llm | StrOutputParser()

    def detect_communities(self):
        try:
            coms = algorithms.leiden(self.graph)
            for i, nodes in enumerate(coms.communities):
                self.communities[i] = nodes
            logger.info(f"Detected {len(self.communities)} communities.")
            return self.communities
        except ImportError:
            logger.error("Leiden algorithm failed.")
            return {}

    def summarize_all(self):
        for com_id, nodes in self.communities.items():
            if len(nodes) < 3: continue
            
            logger.info(f"Summarizing Community {com_id}...")
            
            subgraph = self.graph.subgraph(nodes)
            
            edges = sorted(
                subgraph.edges(data=True), 
                key=lambda x: x[2].get('weight', 1), 
                reverse=True
            )
            
            relations = [
                f"{u} -> {v} ({d.get('description','')})" 
                for u, v, d in edges
            ]
            
            try:
                summary = self.chain.invoke({
                    "entities": ", ".join(nodes),
                    "relationships": "\n".join(relations[:30]) 
                })
                self.summaries[com_id] = summary
            except Exception as e:
                logger.error(f"Failed to summarize community {com_id}: {e}")

    def save_data(self):
        with open(COMMUNITY_SUMMARIES, "w") as f:
            json.dump(self.summaries, f, indent=2)
            
        node_map = {node: cid for cid, nodes in self.communities.items() for node in nodes}
        with open(NODE_MAP, "w") as f:
            json.dump(node_map, f, indent=2)