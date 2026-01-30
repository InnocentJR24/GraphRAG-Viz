import sys
import os
import asyncio
import streamlit as st
import streamlit.components.v1 as components

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag import GlobalSearchEngine
from app.visualizer import GraphVisualizer

st.set_page_config(
    layout="wide", 
    page_title="GraphRAG Local",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_engine():
    return GlobalSearchEngine()

@st.cache_resource
def load_visualizer():
    return GraphVisualizer()

engine = load_engine()
viz = load_visualizer()

@st.cache_data
def get_graph_html(active_communities: list[str] = None):
    return viz.generate_html(active_communities=active_communities)


st.sidebar.title("🕸️ GraphRAG Local")
st.sidebar.markdown("---")
mode = st.sidebar.radio("View Mode", ["Search & Trace", "Full Knowledge Graph"])

st.title("Knowledge Graph Explorer")

if mode == "Search & Trace":
    query = st.text_input("Ask a question about your data:", placeholder="e.g., What are the main themes regarding AI?")
    
    if query:
        with st.spinner("Searching knowledge graph (Vector Search)..."):
            result = asyncio.run(engine.global_search(query))
        
        st.markdown(f"### 💡 Answer")
        st.markdown(result['answer'])
        
        st.markdown("---")
        st.subheader("Reasoning Trace")
        
        if result['evidence']:
            evidence_ids = [str(item['id']) for item in result['evidence']]
            st.caption(f"Found {len(evidence_ids)} relevant communities.")
            
            html_path = get_graph_html(evidence_ids)
            with open(html_path, 'r', encoding='utf-8') as f:
                components.html(f.read(), height=600)
                
            with st.expander("🔍 Inspect Source Summaries"):
                for item in result['evidence']:
                    st.markdown(f"**Community {item['id']} (Relevance: {item['score']})**")
                    st.text(item['summary'])
                    st.divider()
        else:
            st.warning("No relevant communities found in the graph.")

elif mode == "Full Knowledge Graph":
    st.subheader("Global Network Structure")
    st.caption("Visualizing all entities and relationships.")
    
    html_path = get_graph_html(None)
    with open(html_path, 'r', encoding='utf-8') as f:
        components.html(f.read(), height=650)