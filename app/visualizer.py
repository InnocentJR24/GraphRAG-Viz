import networkx as nx
from pyvis.network import Network
import json
import os
import matplotlib.colors as mcolors
from src.config import GRAPH_FILE, NODE_MAP

class GraphVisualizer:
    def __init__(self):
        self.output_dir = os.path.dirname(os.path.abspath(__file__))
        self.G = nx.read_gexf(GRAPH_FILE)
        
        try:
            with open(NODE_MAP, "r") as f:
                self.node_map = json.load(f)
        except FileNotFoundError:
            self.node_map = {}
            
        self.palette = list(mcolors.TABLEAU_COLORS.values())

    def generate_html(self, active_communities: list = None, filename="graph.html"):
        net = Network(height="600px", width="100%", bgcolor="#1E1E1E", font_color="white")
        
        net.force_atlas_2based(
            gravity=-50, 
            central_gravity=0.01, 
            spring_length=100, 
            spring_strength=0.08, 
            damping=0.4, 
            overlap=0
        )

        for node in self.G.nodes():
            com_id = str(self.node_map.get(node, "0"))
            
            color = "#444444"
            size = 8
            label = None
            title = f"Node: {node}\nCommunity: {com_id}"
            
            is_active = active_communities and com_id in active_communities
            
            if is_active or not active_communities:
                color_idx = int(com_id) % len(self.palette) if com_id.isdigit() else 0
                color = self.palette[color_idx]
                size = 15
                label = node
            
            if is_active:
                size = 25
                title += " (MATCH)"

            net.add_node(node, label=label, title=title, color=color, size=size)

        for u, v, data in self.G.edges(data=True):
            src_c = str(self.node_map.get(u, "0"))
            tgt_c = str(self.node_map.get(v, "0"))
            
            width = 1
            color = "#333333"

            if active_communities:
                if src_c in active_communities and tgt_c in active_communities:
                    color = "#rgba(255, 255, 255, 0.6)"
                    width = 2
                else:
                    color = "#rgba(50, 50, 50, 0.2)"

            net.add_edge(u, v, title=data.get('description', ''), color=color, width=width)

        output_path = os.path.join(self.output_dir, filename)
        net.save_graph(output_path)
        return output_path