import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

INPUT_FILE = DATA_RAW / "book.txt"
GRAPH_FILE = DATA_PROCESSED / "knowledge_graph.gexf"
COMMUNITY_SUMMARIES = DATA_PROCESSED / "community_summaries.json"
NODE_MAP = DATA_PROCESSED / "node_community_map.json"

MODEL_NAME = "llama3.2" 
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

os.makedirs(DATA_RAW, exist_ok=True)
os.makedirs(DATA_PROCESSED, exist_ok=True)