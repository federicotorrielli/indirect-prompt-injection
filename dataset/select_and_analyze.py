import re
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud

warnings.filterwarnings("ignore")


def extract_review_texts(reviews_str):
    """Extract review texts from the reviews string"""
    reviews = eval(reviews_str)
    texts = []
    for review in reviews:
        if "review" in review and "main_review" in review["review"]:
            main_review = review["review"]["main_review"]
            if main_review and main_review.startswith("main_review: "):
                text = main_review.replace("main_review: ", "")
                texts.append(text)
    return texts


def calculate_sentiment(text):
    """Calculate sentiment using TextBlob"""
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity


def extract_review_features(reviews_str):
    """Extract detailed features from reviews"""
    reviews = eval(reviews_str)
    features = {
        "num_reviews": len(reviews),
        "review_scores": [],
        "review_confidences": [],
        "review_texts": [],
        "total_text_length": 0,
        "avg_text_length": 0,
        "sentiment_polarity": [],
        "sentiment_subjectivity": [],
    }

    for review in reviews:
        if "score" in review:
            features["review_scores"].append(review["score"])
        if "confidence" in review:
            features["review_confidences"].append(review["confidence"])
        if "review" in review and "main_review" in review["review"]:
            main_review = review["review"]["main_review"]
            if main_review and main_review.startswith("main_review: "):
                text = main_review.replace("main_review: ", "")
                features["review_texts"].append(text)
                features["total_text_length"] += len(text)

                # Calculate sentiment
                polarity, subjectivity = calculate_sentiment(text)
                features["sentiment_polarity"].append(polarity)
                features["sentiment_subjectivity"].append(subjectivity)

    if features["review_texts"]:
        features["avg_text_length"] = features["total_text_length"] / len(
            features["review_texts"]
        )

    return features


def main():
    print("Loading verbose reviews dataset...")
    df = pd.read_csv("openreview_verbose_reviews.csv")

    print(f"Original dataset shape: {df.shape}")
    print(f"Decision distribution: {df['decision'].value_counts()}")

    # Filter for papers with at least 3 reviews
    print("\nFiltering for papers with at least 3 reviews...")
    df_filtered = df[df["num_reviews"] >= 3].copy()
    print(f"After filtering: {df_filtered.shape}")

    # Calculate review statistics for each paper
    print("\nCalculating review statistics...")
    review_stats = []

    for idx, row in df_filtered.iterrows():
        try:
            features = extract_review_features(row["reviews"])
            if features["num_reviews"] >= 3 and features["review_texts"]:
                review_stats.append(
                    {
                        "paperhash": row["paperhash"],
                        "title": row["title"],
                        "decision": row["decision"],
                        "mean_score": row["mean_score"],
                        "num_reviews": features["num_reviews"],
                        "total_text_length": features["total_text_length"],
                        "avg_text_length": features["avg_text_length"],
                        "avg_sentiment_polarity": np.mean(
                            features["sentiment_polarity"]
                        )
                        if features["sentiment_polarity"]
                        else 0,
                        "avg_sentiment_subjectivity": np.mean(
                            features["sentiment_subjectivity"]
                        )
                        if features["sentiment_subjectivity"]
                        else 0,
                        "review_texts": features["review_texts"],
                        "review_scores": features["review_scores"],
                        "sentiment_polarities": features["sentiment_polarity"],
                        "sentiment_subjectivities": features["sentiment_subjectivity"],
                    }
                )
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue

    stats_df = pd.DataFrame(review_stats)
    print(f"Papers with valid review data: {len(stats_df)}")

    # Sort by total text length (most verbose first)
    stats_df = stats_df.sort_values("total_text_length", ascending=False)

    # Select top 100 most verbose papers with balanced accept/reject
    accepted_papers = stats_df[stats_df["decision"] == True].head(50)
    rejected_papers = stats_df[stats_df["decision"] == False].head(50)

    final_selection = pd.concat([accepted_papers, rejected_papers])
    print(f"\nFinal selection: {len(final_selection)} papers")
    print(f"Accepted: {len(accepted_papers)}, Rejected: {len(rejected_papers)}")

    # Text and sentiment analysis
    print("\n=== TEXT AND SENTIMENT ANALYSIS ===")

    # 1. Basic statistics
    print("\nBasic Statistics:")
    print(
        f"Average review length: {final_selection['avg_text_length'].mean():.0f} characters"
    )
    print(
        f"Average sentiment polarity: {final_selection['avg_sentiment_polarity'].mean():.3f}"
    )
    print(
        f"Average sentiment subjectivity: {final_selection['avg_sentiment_subjectivity'].mean():.3f}"
    )

    # 2. Sentiment comparison by decision
    print("\nSentiment by Decision:")
    accepted_sentiment = final_selection[final_selection["decision"] == True][
        "avg_sentiment_polarity"
    ].mean()
    rejected_sentiment = final_selection[final_selection["decision"] == False][
        "avg_sentiment_polarity"
    ].mean()
    print(f"Accepted papers avg sentiment: {accepted_sentiment:.3f}")
    print(f"Rejected papers avg sentiment: {rejected_sentiment:.3f}")
    print(f"Sentiment difference: {accepted_sentiment - rejected_sentiment:.3f}")

    # 3. Extract all review texts for word analysis
    all_review_texts = []
    accepted_texts = []
    rejected_texts = []

    for _, row in final_selection.iterrows():
        for text in row["review_texts"]:
            all_review_texts.append(text)
            if row["decision"]:
                accepted_texts.append(text)
            else:
                rejected_texts.append(text)

    print(f"\nTotal review texts: {len(all_review_texts)}")
    print(f"Accepted review texts: {len(accepted_texts)}")
    print(f"Rejected review texts: {len(rejected_texts)}")

    # 4. Word frequency analysis
    def extract_keywords(texts, top_n=20):
        # Clean and tokenize
        all_words = []
        for text in texts:
            # Remove common review phrases and clean
            text = re.sub(r"main_review:", "", text.lower())
            text = re.sub(r"[^\w\s]", " ", text)
            words = text.split()
            # Filter out common words and short words
            stop_words = {
                "the",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "a",
                "an",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "could",
                "should",
                "may",
                "might",
                "can",
                "this",
                "that",
                "these",
                "those",
                "i",
                "you",
                "he",
                "she",
                "it",
                "we",
                "they",
                "me",
                "him",
                "her",
                "us",
                "them",
                "my",
                "your",
                "his",
                "her",
                "its",
                "our",
                "their",
            }
            words = [w for w in words if len(w) > 3 and w not in stop_words]
            all_words.extend(words)
        return Counter(all_words).most_common(top_n)

    print("\nTop keywords in accepted reviews:")
    accepted_keywords = extract_keywords(accepted_texts)
    for word, count in accepted_keywords[:10]:
        print(f"  {word}: {count}")

    print("\nTop keywords in rejected reviews:")
    rejected_keywords = extract_keywords(rejected_texts)
    for word, count in rejected_keywords[:10]:
        print(f"  {word}: {count}")

    # 5. Create visualizations
    plt.figure(figsize=(15, 10))

    # Sentiment distribution
    plt.subplot(2, 3, 1)
    plt.hist(
        final_selection[final_selection["decision"] == True]["avg_sentiment_polarity"],
        alpha=0.7,
        label="Accepted",
        bins=20,
    )
    plt.hist(
        final_selection[final_selection["decision"] == False]["avg_sentiment_polarity"],
        alpha=0.7,
        label="Rejected",
        bins=20,
    )
    plt.xlabel("Sentiment Polarity")
    plt.ylabel("Frequency")
    plt.title("Sentiment Distribution by Decision")
    plt.legend()

    # Review length distribution
    plt.subplot(2, 3, 2)
    plt.hist(
        final_selection[final_selection["decision"] == True]["avg_text_length"],
        alpha=0.7,
        label="Accepted",
        bins=20,
    )
    plt.hist(
        final_selection[final_selection["decision"] == False]["avg_text_length"],
        alpha=0.7,
        label="Rejected",
        bins=20,
    )
    plt.xlabel("Average Review Length")
    plt.ylabel("Frequency")
    plt.title("Review Length Distribution by Decision")
    plt.legend()

    # Sentiment vs Score
    plt.subplot(2, 3, 3)
    plt.scatter(
        final_selection["avg_sentiment_polarity"],
        final_selection["mean_score"],
        c=["green" if x else "red" for x in final_selection["decision"]],
        alpha=0.6,
    )
    plt.xlabel("Sentiment Polarity")
    plt.ylabel("Mean Score")
    plt.title("Sentiment vs Review Score")

    # Subjectivity distribution
    plt.subplot(2, 3, 4)
    plt.hist(
        final_selection[final_selection["decision"] == True][
            "avg_sentiment_subjectivity"
        ],
        alpha=0.7,
        label="Accepted",
        bins=20,
    )
    plt.hist(
        final_selection[final_selection["decision"] == False][
            "avg_sentiment_subjectivity"
        ],
        alpha=0.7,
        label="Rejected",
        bins=20,
    )
    plt.xlabel("Sentiment Subjectivity")
    plt.ylabel("Frequency")
    plt.title("Subjectivity Distribution by Decision")
    plt.legend()

    # Number of reviews distribution
    plt.subplot(2, 3, 5)
    plt.hist(
        final_selection[final_selection["decision"] == True]["num_reviews"],
        alpha=0.7,
        label="Accepted",
        bins=range(3, 12),
    )
    plt.hist(
        final_selection[final_selection["decision"] == False]["num_reviews"],
        alpha=0.7,
        label="Rejected",
        bins=range(3, 12),
    )
    plt.xlabel("Number of Reviews")
    plt.ylabel("Frequency")
    plt.title("Number of Reviews Distribution")
    plt.legend()

    # Box plot of sentiment by decision
    plt.subplot(2, 3, 6)
    sentiment_data = [
        final_selection[final_selection["decision"] == True]["avg_sentiment_polarity"],
        final_selection[final_selection["decision"] == False]["avg_sentiment_polarity"],
    ]
    plt.boxplot(sentiment_data, labels=["Accepted", "Rejected"])
    plt.ylabel("Sentiment Polarity")
    plt.title("Sentiment Box Plot by Decision")

    plt.tight_layout()
    plt.savefig("sentiment_analysis_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 6. Word clouds
    if accepted_texts and rejected_texts:
        plt.figure(figsize=(15, 6))

        # Word cloud for accepted reviews
        plt.subplot(1, 2, 1)
        accepted_text = " ".join(accepted_texts)
        wordcloud = WordCloud(width=400, height=300, background_color="white").generate(
            accepted_text
        )
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud - Accepted Reviews")

        # Word cloud for rejected reviews
        plt.subplot(1, 2, 2)
        rejected_text = " ".join(rejected_texts)
        wordcloud = WordCloud(width=400, height=300, background_color="white").generate(
            rejected_text
        )
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud - Rejected Reviews")

        plt.tight_layout()
        plt.savefig("wordclouds.png", dpi=300, bbox_inches="tight")
        plt.show()

    # 7. Detailed sentiment analysis per review
    individual_sentiments = []
    for _, row in final_selection.iterrows():
        for i, (text, polarity, subjectivity) in enumerate(
            zip(
                row["review_texts"],
                row["sentiment_polarities"],
                row["sentiment_subjectivities"],
            )
        ):
            individual_sentiments.append(
                {
                    "paperhash": row["paperhash"],
                    "decision": row["decision"],
                    "review_index": i,
                    "text_length": len(text),
                    "sentiment_polarity": polarity,
                    "sentiment_subjectivity": subjectivity,
                    "text": text[:200] + "..." if len(text) > 200 else text,
                }
            )

    individual_df = pd.DataFrame(individual_sentiments)

    print("\n=== DETAILED SENTIMENT STATISTICS ===")
    print(f"Total individual reviews: {len(individual_df)}")
    print("\nSentiment statistics by decision:")
    print(
        individual_df.groupby("decision")[
            ["sentiment_polarity", "sentiment_subjectivity"]
        ].describe()
    )

    # Statistical significance test
    from scipy import stats

    accepted_sentiments = individual_df[individual_df["decision"] == True][
        "sentiment_polarity"
    ]
    rejected_sentiments = individual_df[individual_df["decision"] == False][
        "sentiment_polarity"
    ]

    t_stat, p_value = stats.ttest_ind(accepted_sentiments, rejected_sentiments)
    print("\nT-test for sentiment difference:")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Statistically significant: {'Yes' if p_value < 0.05 else 'No'}")

    # Save results
    final_selection.to_csv("selected_100_papers.csv", index=False)
    individual_df.to_csv("individual_review_sentiments.csv", index=False)

    print("\nFiles saved:")
    print(f"- selected_100_papers.csv: {len(final_selection)} papers")
    print(
        f"- individual_review_sentiments.csv: {len(individual_df)} individual reviews"
    )
    print("- sentiment_analysis_results.png: Visualization plots")
    print("- wordclouds.png: Word cloud comparisons")

    return final_selection, individual_df


if __name__ == "__main__":
    selection, individual_reviews = main()
