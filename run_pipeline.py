import asyncio
from src.utils import load_text, chunk_text
from src.pipeline import GraphBuilder
from src.communities import CommunitySummarizer
from src.config import INPUT_FILE

async def main():
    print(">>> 1. LOADING DATA")
    raw_text = load_text(INPUT_FILE)
    if not raw_text:
        print(f"Please create {INPUT_FILE} with your content.")
        return

    chunks = chunk_text(raw_text)

    print("\n>>> 2. BUILDING GRAPH (ASYNC)")
    builder = GraphBuilder()
    await builder.process_documents_async(chunks, concurrency=4)
    builder.resolve_entities()
    builder.save_graph()

    print("\n>>> 3. SUMMARIZING COMMUNITIES")
    summarizer = CommunitySummarizer(builder.graph)
    if summarizer.detect_communities():
        summarizer.summarize_all()
        summarizer.save_data()
        print("\n>>> PIPELINE COMPLETE.")
    else:
        print("No communities detected.")

if __name__ == "__main__":
    asyncio.run(main())