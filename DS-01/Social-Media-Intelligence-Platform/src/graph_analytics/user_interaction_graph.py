
import json
import csv
from pathlib import Path
from collections import Counter

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "raw" / "hackernews_posts.json"
)

GRAPH_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "hackernews_interaction_graph.graphml"
)

NODE_REPORT = (
    PROJECT_ROOT / "reports" / "evaluation"
    / "user_centrality.csv"
)

EDGE_REPORT = (
    PROJECT_ROOT / "reports" / "evaluation"
    / "user_interactions.csv"
)


def build_interaction_graph():

    print("Loading Hacker News records...")

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        records = json.load(file)

    graph = nx.DiGraph()

    story_authors = set()
    comment_authors = set()

    for record in records:

        if record.get("item_type") == "story":

            author = record.get("author")

            if author and author != "unknown":
                story_authors.add(author)

                graph.add_node(
                    author,
                    role="story_author"
                )

        elif record.get("item_type") == "comment":

            commenter = record.get("author")
            story_author = record.get("story_author")

            if (
                not commenter
                or not story_author
                or commenter == "unknown"
                or story_author == "unknown"
            ):
                continue

            comment_authors.add(commenter)

            graph.add_node(
                commenter,
                role="commenter"
            )

            # A user commenting on their own story
            if commenter == story_author:
                continue

            if graph.has_edge(commenter, story_author):

                graph[commenter][story_author]["weight"] += 1

            else:

                graph.add_edge(
                    commenter,
                    story_author,
                    weight=1
                )

    print("\nGraph construction completed!")

    print("Number of nodes:", graph.number_of_nodes())
    print("Number of edges:", graph.number_of_edges())

    # Calculate centrality metrics
    degree_centrality = nx.degree_centrality(graph)

    betweenness_centrality = nx.betweenness_centrality(
        graph,
        weight=None
    )

    weighted_interactions = dict(graph.in_degree(weight="weight"))

    # Save graph
    GRAPH_FILE.parent.mkdir(parents=True, exist_ok=True)

    nx.write_graphml(graph, GRAPH_FILE)

    # Save user centrality report
    NODE_REPORT.parent.mkdir(parents=True, exist_ok=True)

    with NODE_REPORT.open(
        "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "username",
            "role",
            "degree_centrality",
            "betweenness_centrality",
            "weighted_interactions"
        ])

        for user in graph.nodes:

            writer.writerow([
                user,
                graph.nodes[user].get("role", "unknown"),
                round(degree_centrality.get(user, 0), 6),
                round(betweenness_centrality.get(user, 0), 6),
                weighted_interactions.get(user, 0)
            ])

    # Save interaction report
    with EDGE_REPORT.open(
        "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "commenter",
            "story_author",
            "interaction_count"
        ])

        for source, target, data in graph.edges(data=True):

            writer.writerow([
                source,
                target,
                data.get("weight", 1)
            ])

    # Display top users
    print("\nTOP USERS BY DEGREE CENTRALITY:")

    top_users = sorted(
        degree_centrality.items(),
        key=lambda item: item[1],
        reverse=True
    )[:10]

    for username, score in top_users:

        print(
            f"{username}: {score:.6f}"
        )

    print("\nReports saved:")
    print(GRAPH_FILE)
    print(NODE_REPORT)
    print(EDGE_REPORT)


if __name__ == "__main__":
    build_interaction_graph()
