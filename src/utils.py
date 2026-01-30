import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import CHUNK_SIZE, CHUNK_OVERLAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_text(filepath: str) -> str:
    """Reads a raw text file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"File not found at {filepath}")
        return ""

def chunk_text(text: str) -> list[str]:
    """Splits text into chunks for the LLM context window."""
    if not text:
        return []
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_text(text)
    logger.info(f"Split text into {len(chunks)} chunks.")
    return chunks