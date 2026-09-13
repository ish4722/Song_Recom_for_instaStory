"""Offline recommender evaluation.

Expected CSV schema:
    image,expected_song_ids

`expected_song_ids` is a comma-separated list of relevant song identifiers.
Use artist::track as the identifier, for example: Taylor Swift::Cruel Summer.
The current CSV is intentionally a placeholder. Populate it before running.
"""

import csv
import math
import os

DATASET_PATH = os.path.join(os.path.dirname(__file__), "evaluation_dataset.csv")


def precision_at_k(predicted, relevant, k):
    return sum(x in relevant for x in predicted[:k]) / k


def recall_at_k(predicted, relevant, k):
    if not relevant:
        return 0.0
    return sum(x in relevant for x in predicted[:k]) / len(relevant)


def mrr(predicted, relevant):
    for rank, item in enumerate(predicted, 1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(predicted, relevant, k):
    dcg = sum((1.0 / math.log2(rank + 1)) for rank, item in enumerate(predicted[:k], 1) if item in relevant)
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def load_rows():
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row.get("image") and not row["image"].startswith("#")]
    return rows


def main():
    rows = load_rows()
    if not rows:
        print("Evaluation dataset is empty.")
        print(f"Fill {DATASET_PATH} with real image-to-song relevance labels, then rerun this script.")
        return

    # This module intentionally keeps metric computation independent of the web app.
    # Connect your image inference function here when the labeled dataset is ready.
    print(f"Loaded {len(rows)} evaluation examples.")
    print("Next step: connect the inference function to generate top-K predictions for each image.")


if __name__ == "__main__":
    main()
