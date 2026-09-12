import json
import re
from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

BRAND = "British_Airways"

INPUT_FILE = Path("data/raw/large_sample.csv")
OUTPUT_FILE = Path("data/processed/interactions.jsonl")


# ============================================================
# Load dataset
# ============================================================

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df):,} tweets")


# ============================================================
# Prepare data
# ============================================================

df["tweet_id"] = df["tweet_id"].astype(str)

df["timestamp"] = pd.to_datetime(
    df["created_at"],
    errors="coerce",
    utc=True
)

df = df.sort_values("timestamp").reset_index(drop=True)


# ============================================================
# Separate customer and support tweets
# ============================================================

customer_tweets = df[
    df["inbound"] == True
].copy()

support_tweets = df[
    (df["inbound"] == False) &
    (df["author_id"] == BRAND)
].copy()

print(f"Customer tweets: {len(customer_tweets):,}")
print(f"{BRAND} support tweets: {len(support_tweets):,}")


# ============================================================
# Find customer tweets mentioning our brand
# ============================================================

brand_pattern = re.escape(BRAND)

customer_tweets = customer_tweets[
    customer_tweets["text"]
    .str.contains(
        brand_pattern,
        case=False,
        regex=True,
        na=False
    )
].copy()

print(
    f"Customer tweets mentioning {BRAND}: "
    f"{len(customer_tweets):,}"
)


# ============================================================
# Resolution heuristic
# ============================================================

resolution_words = [
    "resolved",
    "fixed",
    "solved",
    "working",
    "sorted",
    "all set",
    "issue is fixed",
    "problem solved",
]


def looks_resolved(text):
    text = str(text).lower()

    return any(
        word in text
        for word in resolution_words
    )


support_tweets["resolved"] = (
    support_tweets["text"]
    .apply(looks_resolved)
)


# ============================================================
# Create interactions
# ============================================================

interactions = []

support_records = support_tweets.to_dict("records")


for _, customer in customer_tweets.iterrows():

    customer_time = customer["timestamp"]

    # Find support responses after the customer tweet.
    possible_responses = [
        support
        for support in support_records
        if support["timestamp"] >= customer_time
    ]

    if not possible_responses:
        continue

    # Select the earliest support response.
    support = min(
        possible_responses,
        key=lambda x: x["timestamp"]
    )

    interaction = {
        "conversation_id": str(customer["tweet_id"]),

        "customer_message": str(
            customer["text"]
        ),

        "support_response": str(
            support["text"]
        ),

        "conversation": [
            {
                "tweet_id": str(customer["tweet_id"]),
                "speaker": "customer",
                "text": str(customer["text"]),
            },
            {
                "tweet_id": str(support["tweet_id"]),
                "speaker": "support",
                "text": str(support["text"]),
            },
        ],

        "resolved": bool(
            support["resolved"]
        ),
    }

    interactions.append(interaction)


# ============================================================
# Create output directory
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Save JSONL
# ============================================================

print(
    f"Writing {len(interactions):,} interactions..."
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    for interaction in interactions:

        f.write(
            json.dumps(
                interaction,
                ensure_ascii=False
            ) + "\n"
        )


print()
print("Done!")
print(f"Output: {OUTPUT_FILE}")
print(f"Interactions: {len(interactions):,}")