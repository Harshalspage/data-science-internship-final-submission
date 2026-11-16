
import os
import networkx as nx
import matplotlib.pyplot as plt

# Project paths
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

GRAPH_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "hackernews_interaction_graph.graphml"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "reports", "figures")
OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "hackernews_interaction_network.png"
)

# Load graph
print("Loading interaction graph...")

G = nx.read_graphml(GRAPH_PATH)

print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")

# Calculate node centrality
centrality = nx.degree_centrality(G)

# Node sizes based on centrality
node_sizes = [
    300 + (centrality[node] * 15000)
    for node in G.nodes()
]

# Create network layout
print("Generating network visualization...")

pos = nx.spring_layout(
    G,
    seed=42,
    k=0.8,
    iterations=100
)

# Draw graph
plt.figure(figsize=(16, 12))

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=node_sizes,
    node_color="skyblue",
    alpha=0.85
)

nx.draw_networkx_edges(
    G,
    pos,
    edge_color="gray",
    alpha=0.4,
    arrows=True,
    arrowsize=10
)

# Label only the top 10 users
top_users = sorted(
    centrality.items(),
    key=lambda item: item[1],
    reverse=True
)[:10]

top_labels = {
    user: user for user, score in top_users
}

nx.draw_networkx_labels(
    G,
    pos,
    labels=top_labels,
    font_size=8,
    font_weight="bold"
)

plt.title(
    "Hacker News User Interaction Network",
    fontsize=16
)

plt.axis("off")
plt.tight_layout()

# Save visualization
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nVisualization saved successfully!")
print(OUTPUT_PATH)
