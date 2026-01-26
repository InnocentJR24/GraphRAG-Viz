import json
import asyncio
import logging
import numpy as np
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from .config import MODEL_NAME, COMMUNITY_SUMMARIES

logger = logging.getLogger(__name__)

class GlobalSearchEngine:
    def __init__(self):
        self.llm = ChatOllama(model=MODEL_NAME, temperature=0)
        self.embedder = OllamaEmbeddings(model="nomic-embed-text") 
        
        self.summaries = {}
        self.community_vectors = {}
        self._load_data()
        
        self.rerank_prompt = PromptTemplate(
            template="""
            Query: {query}
            Summary: {summary}
            
            Rate the relevance of this summary to the query on a scale of 0 to 10.
            Return ONLY a JSON object: {{"score": <int>, "reason": "<string>"}}
            """,
            input_variables=["query", "summary"]
        )
        self.reranker = self.rerank_prompt | self.llm | JsonOutputParser()

        self.reduce_chain = (
            PromptTemplate.from_template(
                """
                Query: {query}
                
                Context (from relevant communities):
                {context}
                
                Answer the query comprehensively using ONLY the provided context.
                """
            )
            | self.llm
        )

    def _load_data(self):
        try:
            with open(COMMUNITY_SUMMARIES, "r") as f:
                self.summaries = json.load(f)
            
            if self.summaries:
                logger.info("Pre-computing community embeddings...")
                texts = [f"Summary: {s}" for s in self.summaries.values()]
                self.ids = list(self.summaries.keys())
                
                if texts:
                    self.embeddings = self.embedder.embed_documents(texts)
                    self.community_vectors = dict(zip(self.ids, self.embeddings))
                else:
                    self.community_vectors = {}
                logger.info(f"Embeddings ready for {len(self.ids)} communities.")
                
        except FileNotFoundError:
            logger.warning("Summaries not found. Run pipeline first.")

    def _cosine_similarity(self, vec1, vec2):
        if not vec1 or not vec2: return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    async def _rerank_candidates(self, query, candidates):
        logger.info(f"Reranking {len(candidates)} candidates with LLM...")

        async def score_doc(cid, text):
            try:
                response = await self.reranker.ainvoke({"query": query, "summary": text})
                return {
                    "id": cid, 
                    "score": response.get("score", 0), 
                    "summary": text,
                    "reason": response.get("reason", "")
                }
            except Exception as e:
                logger.error(f"Rerank failed for {cid}: {e}")
                return {"id": cid, "score": 0, "summary": text}

        tasks = [score_doc(cid, self.summaries[cid]) for cid, _ in candidates]
        scored_docs = await asyncio.gather(*tasks)
        relevant_docs = [d for d in scored_docs if d['score'] > 4]
        return sorted(relevant_docs, key=lambda x: x['score'], reverse=True)

    async def global_search(self, query: str):
        logger.info(f"Searching: {query}")

        query_vec = self.embedder.embed_query(query)
        scores = []
        for com_id, com_vec in self.community_vectors.items():
            score = self._cosine_similarity(query_vec, com_vec)
            scores.append((com_id, score))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        broad_candidates = scores[:15]
        if not broad_candidates:
            return {"answer": "No information found.", "evidence": []}
            
        evidence = await self._rerank_candidates(query, broad_candidates)
        final_evidence = evidence[:5]
        if not final_evidence:
            return {"answer": "I found some potential topics, but they didn't match your specific question closely enough.", "evidence": []}

        context_str = "\n\n".join([f"ID {e['id']} (Relevance {e['score']}/10): {e['summary']}" for e in final_evidence])
        final_answer = await self.reduce_chain.ainvoke({"query": query, "context": context_str})
        return {"answer": final_answer.content, "evidence": final_evidence}