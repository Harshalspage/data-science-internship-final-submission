
import json
import csv
from pathlib import Path
from collections import Counter, defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "cleaned_posts.json"
)

RESULT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "topic_model_results_unified.json"
)

KEYWORD_FILE = (
    PROJECT_ROOT / "reports" / "evaluation"
    / "unified_topic_keywords.csv"
)

PLATFORM_FILE = (
    PROJECT_ROOT / "reports" / "evaluation"
    / "platform_topic_distribution.csv"
)


def run_topic_modeling():

    print("Loading unified cleaned dataset...")

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        records = json.load(file)

    valid_records = [
        record for record in records
        if record.get("cleaned_text", "").strip()
    ]

    if len(valid_records) < 3:
        raise ValueError("At least three valid records are required.")

    documents = [
        record["cleaned_text"]
        for record in valid_records
    ]

    print("Records loaded:", len(documents))

    # Convert text into TF-IDF features
    print("\nCreating TF-IDF features...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=2000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    if tfidf_matrix.shape[1] < 3:
        raise ValueError("Not enough vocabulary for topic modeling.")

    # Train NMF model
    n_topics = min(5, tfidf_matrix.shape[0], tfidf_matrix.shape[1])

    print("Training NMF model...")
    print("Topics configured:", n_topics)

    nmf_model = NMF(
        n_components=n_topics,
        random_state=42,
        init="nndsvda",
        max_iter=500
    )

    document_topics = nmf_model.fit_transform(tfidf_matrix)

    feature_names = vectorizer.get_feature_names_out()

    # Extract topic keywords
    topic_keywords = []

    for topic_id, weights in enumerate(nmf_model.components_):

        top_indices = weights.argsort()[::-1][:10]

        keywords = [
            feature_names[index]
            for index in top_indices
            if weights[index] > 0
        ]

        topic_keywords.append({
            "topic_id": int(topic_id),
            "keywords": keywords
        })

    # Assign topics to records
    results = []

    platform_distribution = defaultdict(Counter)
    topic_distribution = Counter()

    for index, record in enumerate(valid_records):

        scores = document_topics[index]
        total_score = scores.sum()

        if total_score > 0:
            topic_id = int(scores.argmax())
            topic_strength = float(scores[topic_id] / total_score)
            keywords = topic_keywords[topic_id]["keywords"]
        else:
            # No meaningful topic contribution
            topic_id = -1
            topic_strength = 0.0
            keywords = "Unassigned - no topic contribution"

        result = {
            **record,
            "topic_id": topic_id,
            "topic_keywords": keywords,
            "topic_strength": round(topic_strength, 6)
        }

        results.append(result)

        platform = record.get("platform", "unknown")

        platform_distribution[platform][topic_id] += 1
        topic_distribution[topic_id] += 1

    # Save post-level results
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with RESULT_FILE.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    # Save topic keywords
    with KEYWORD_FILE.open(
        "w", encoding="utf-8", newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(["topic_id", "keywords", "record_count"])

        for topic in topic_keywords:
            topic_id = topic["topic_id"]

            writer.writerow([
                topic_id,
                ", ".join(topic["keywords"]),
                topic_distribution[topic_id]
            ])
        if topic_distribution[-1] > 0:
            writer.writerow([
                -1,
                "Unassigned - no topic contribution",
                topic_distribution[-1]
            ])

    # Save platform-wise topic distribution
    with PLATFORM_FILE.open(
        "w", encoding="utf-8", newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "platform",
            "topic_id",
            "record_count"
        ])

        for platform, counts in sorted(platform_distribution.items()):

            for topic_id, count in sorted(counts.items()):

                writer.writerow([
                    platform,
                    topic_id,
                    count
                ])

    # Display results
    print("\nTOPIC MODELING COMPLETED")
    print("------------------------")
    print("Records analyzed:", len(results))
    print("Topics discovered:", n_topics)

    print("\nDISCOVERED TOPICS:")

    for topic in topic_keywords:

        print(
            f"\nTopic {topic['topic_id']}: "
            f"{', '.join(topic['keywords'])}"
        )

        print(
            "Records assigned:",
            topic_distribution[topic["topic_id"]]
        )

    print("\nPLATFORM-WISE TOPIC DISTRIBUTION:")

    for platform, counts in platform_distribution.items():
        print(platform, dict(counts))

    print("\nGenerated reports:")
    print(RESULT_FILE)
    print(KEYWORD_FILE)
    print(PLATFORM_FILE)


if __name__ == "__main__":
    run_topic_modeling()
