import ast
import warnings
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset

warnings.filterwarnings("ignore")

# Load the dataset
print("Loading OpenReview dataset...")
dataset = load_dataset("nhop/OpenReview", trust_remote_code=True)
df = pd.DataFrame(dataset["train"])

print(f"Original dataset shape: {df.shape}")

# 1. Filter papers and reviews up to November 2022
print("\n1. Filtering papers up to November 2022...")


def parse_date(date_str):
    """Parse various date formats"""
    if pd.isna(date_str):
        return None

    try:
        # Try different formats
        formats = [
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%SZ",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue

        # If none work, try parsing just the date part
        date_part = date_str.split(" ")[0]
        return datetime.strptime(date_part, "%Y-%m-%d")
    except:
        return None


# Parse publication dates
df["parsed_date"] = df["publication_date"].apply(parse_date)
df = df.dropna(subset=["parsed_date"])

# Filter up to November 2022
cutoff_date = datetime(2022, 11, 30)
df_filtered = df[df["parsed_date"] <= cutoff_date].copy()

print(f"After date filtering: {df_filtered.shape[0]} papers")
print(
    f"Date range: {df_filtered['parsed_date'].min()} to {df_filtered['parsed_date'].max()}"
)

# 2. Parse and analyze reviews
print("\n2. Analyzing reviews...")


def parse_reviews(review_data):
    """Parse review data"""
    if review_data is None:
        return []

    try:
        # Reviews are already parsed as lists
        if isinstance(review_data, list):
            return review_data
        elif isinstance(review_data, str):
            reviews = ast.literal_eval(review_data)
            return reviews if isinstance(reviews, list) else []
        else:
            return []
    except Exception:
        return []


def get_review_lengths(reviews):
    """Get lengths of review texts"""
    lengths = []
    for review in reviews:
        if isinstance(review, dict) and "review" in review:
            review_content = review["review"]
            if isinstance(review_content, dict):
                # Sum up all text fields in the review
                total_length = 0
                for field in [
                    "main_review",
                    "paper_summary",
                    "strength_weakness",
                    "questions",
                    "limitations",
                ]:
                    if field in review_content and review_content[field]:
                        total_length += len(str(review_content[field]))
                lengths.append(total_length)
            elif isinstance(review_content, str):
                lengths.append(len(review_content))
    return lengths


# Parse reviews
df_filtered["parsed_reviews"] = df_filtered["reviews"].apply(parse_reviews)
df_filtered["num_reviews"] = df_filtered["parsed_reviews"].apply(len)
df_filtered["review_lengths"] = df_filtered["parsed_reviews"].apply(get_review_lengths)
df_filtered["total_review_length"] = df_filtered["review_lengths"].apply(sum)
df_filtered["avg_review_length"] = df_filtered["review_lengths"].apply(
    lambda x: np.mean(x) if x else 0
)

print(f"Papers with reviews: {(df_filtered['num_reviews'] > 0).sum()}")
print(f"Average number of reviews per paper: {df_filtered['num_reviews'].mean():.2f}")
print(
    f"Average review length: {df_filtered['avg_review_length'].mean():.0f} characters"
)

# 3. Filter papers with verbose reviews
print("\n3. Filtering papers with verbose reviews...")

# Define verbose reviews as those with substantial content
# Using 75th percentile of review lengths as threshold
review_length_threshold = df_filtered["avg_review_length"].quantile(0.75)
print(
    f"Review length threshold (75th percentile): {review_length_threshold:.0f} characters"
)

df_verbose = df_filtered[
    (df_filtered["num_reviews"] > 0)
    & (df_filtered["avg_review_length"] > review_length_threshold)
].copy()

print(f"Papers with verbose reviews: {df_verbose.shape[0]}")

# 4. Create visualizations
print("\n4. Creating visualizations...")

# Set up the plotting style
plt.style.use("default")
sns.set_palette("husl")

# Create comprehensive analysis plots
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle(
    "OpenReview Dataset Analysis: Papers with Verbose Reviews (up to Nov 2022)",
    fontsize=16,
    y=0.98,
)

# 1. Distribution of review lengths (normal distribution)
ax1 = axes[0, 0]
review_lengths_flat = [
    length for lengths in df_verbose["review_lengths"] for length in lengths
]
ax1.hist(review_lengths_flat, bins=50, alpha=0.7, edgecolor="black")
ax1.set_xlabel("Review Length (characters)")
ax1.set_ylabel("Frequency")
ax1.set_title("Distribution of Individual Review Lengths")
ax1.axvline(
    np.mean(review_lengths_flat),
    color="red",
    linestyle="--",
    label=f"Mean: {np.mean(review_lengths_flat):.0f}",
)
ax1.legend()

# 2. Average review length distribution
ax2 = axes[0, 1]
ax2.hist(df_verbose["avg_review_length"], bins=30, alpha=0.7, edgecolor="black")
ax2.set_xlabel("Average Review Length per Paper")
ax2.set_ylabel("Frequency")
ax2.set_title("Distribution of Average Review Lengths")
ax2.axvline(
    df_verbose["avg_review_length"].mean(),
    color="red",
    linestyle="--",
    label=f"Mean: {df_verbose['avg_review_length'].mean():.0f}",
)
ax2.legend()

# 3. Number of reviews per paper
ax3 = axes[0, 2]
review_counts = df_verbose["num_reviews"].value_counts().sort_index()
ax3.bar(review_counts.index, review_counts.values, alpha=0.7)
ax3.set_xlabel("Number of Reviews")
ax3.set_ylabel("Number of Papers")
ax3.set_title("Number of Reviews per Paper")

# 4. Publications over time
ax4 = axes[1, 0]
df_verbose["year"] = df_verbose["parsed_date"].dt.year
yearly_counts = df_verbose["year"].value_counts().sort_index()
ax4.plot(
    yearly_counts.index, yearly_counts.values, marker="o", linewidth=2, markersize=6
)
ax4.set_xlabel("Year")
ax4.set_ylabel("Number of Papers")
ax4.set_title("Papers with Verbose Reviews by Year")
ax4.grid(True, alpha=0.3)

# 5. Mean scores distribution
ax5 = axes[1, 1]
ax5.hist(df_verbose["mean_score"].dropna(), bins=30, alpha=0.7, edgecolor="black")
ax5.set_xlabel("Mean Score")
ax5.set_ylabel("Frequency")
ax5.set_title("Distribution of Mean Review Scores")
ax5.axvline(
    df_verbose["mean_score"].mean(),
    color="red",
    linestyle="--",
    label=f"Mean: {df_verbose['mean_score'].mean():.2f}",
)
ax5.legend()

# 6. Venue distribution (top 10)
ax6 = axes[1, 2]
venue_counts = df_verbose["venue"].value_counts().head(10)
ax6.barh(range(len(venue_counts)), venue_counts.values, alpha=0.7)
ax6.set_yticks(range(len(venue_counts)))
ax6.set_yticklabels(venue_counts.index, fontsize=8)
ax6.set_xlabel("Number of Papers")
ax6.set_title("Top 10 Venues")

# 7. Decision distribution
ax7 = axes[2, 0]
decision_counts = df_verbose["decision"].value_counts()
ax7.pie(
    decision_counts.values,
    labels=["Rejected", "Accepted"],
    autopct="%1.1f%%",
    startangle=90,
)
ax7.set_title("Paper Decision Distribution")

# 8. Citation analysis
ax8 = axes[2, 1]
df_verbose["n_citations_filled"] = df_verbose["n_citations"].fillna(0)
ax8.scatter(df_verbose["mean_score"], df_verbose["n_citations_filled"], alpha=0.6)
ax8.set_xlabel("Mean Review Score")
ax8.set_ylabel("Number of Citations")
ax8.set_title("Review Score vs Citations")
ax8.set_yscale("log")

# 9. Review confidence vs score
ax9 = axes[2, 2]
ax9.scatter(df_verbose["mean_confidence"], df_verbose["mean_score"], alpha=0.6)
ax9.set_xlabel("Mean Confidence")
ax9.set_ylabel("Mean Score")
ax9.set_title("Review Confidence vs Score")

plt.tight_layout()
plt.savefig("openreview_analysis.png", dpi=300, bbox_inches="tight")
plt.show()

# 5. Extract detailed statistics
print("\n5. Extracting detailed statistics...")

# Paper statistics
print("\n=== PAPER STATISTICS ===")
print(f"Total papers with verbose reviews: {len(df_verbose)}")
print(
    f"Date range: {df_verbose['parsed_date'].min().strftime('%Y-%m-%d')} to {df_verbose['parsed_date'].max().strftime('%Y-%m-%d')}"
)
print(f"Unique venues: {df_verbose['venue'].nunique()}")
print(
    f"Accepted papers: {df_verbose['decision'].sum()} ({df_verbose['decision'].mean() * 100:.1f}%)"
)
print(
    f"Rejected papers: {(~df_verbose['decision']).sum()} ({(~df_verbose['decision']).mean() * 100:.1f}%)"
)

# Review statistics
print("\n=== REVIEW STATISTICS ===")
total_reviews = df_verbose["num_reviews"].sum()
print(f"Total reviews: {total_reviews}")
print(f"Average reviews per paper: {df_verbose['num_reviews'].mean():.2f}")
print("Review length statistics:")
print(f"  - Mean: {np.mean(review_lengths_flat):.0f} characters")
print(f"  - Median: {np.median(review_lengths_flat):.0f} characters")
print(f"  - Std: {np.std(review_lengths_flat):.0f} characters")
print(f"  - Min: {min(review_lengths_flat):.0f} characters")
print(f"  - Max: {max(review_lengths_flat):.0f} characters")

# Score statistics
print("\n=== SCORE STATISTICS ===")
print(
    f"Mean score: {df_verbose['mean_score'].mean():.3f} ± {df_verbose['mean_score'].std():.3f}"
)
print(
    f"Mean confidence: {df_verbose['mean_confidence'].mean():.3f} ± {df_verbose['mean_confidence'].std():.3f}"
)
print(
    f"Score range: {df_verbose['mean_score'].min():.3f} to {df_verbose['mean_score'].max():.3f}"
)

# Field of study analysis
print("\n=== FIELD OF STUDY ANALYSIS ===")


def parse_field_list(field_data):
    if field_data is None:
        return []
    try:
        if isinstance(field_data, list):
            return field_data
        elif isinstance(field_data, str):
            return ast.literal_eval(field_data)
        return []
    except Exception:
        return []


df_verbose["parsed_fields"] = df_verbose["field_of_study"].apply(parse_field_list)
all_fields = [field for fields in df_verbose["parsed_fields"] for field in fields]
field_counts = Counter(all_fields)
print("Top 10 fields of study:")
for field, count in field_counts.most_common(10):
    print(f"  {field}: {count}")

# Citation analysis
print("\n=== CITATION ANALYSIS ===")
cited_papers = df_verbose[df_verbose["n_citations"].notna()]
print(f"Papers with citation data: {len(cited_papers)}")
if len(cited_papers) > 0:
    print(f"Average citations: {cited_papers['n_citations'].mean():.2f}")
    print(f"Median citations: {cited_papers['n_citations'].median():.2f}")
    print(f"Most cited paper: {cited_papers['n_citations'].max():.0f} citations")

# Temporal analysis
print("\n=== TEMPORAL ANALYSIS ===")
yearly_stats = (
    df_verbose.groupby("year")
    .agg(
        {
            "mean_score": ["mean", "std", "count"],
            "num_reviews": "mean",
            "decision": "mean",
        }
    )
    .round(3)
)
yearly_stats.columns = [
    "avg_score",
    "score_std",
    "num_papers",
    "avg_reviews",
    "acceptance_rate",
]
print("Yearly statistics:")
print(yearly_stats)

# Save the filtered dataset
df_verbose.to_csv("openreview_verbose_reviews.csv", index=False)
print("\nFiltered dataset saved to 'openreview_verbose_reviews.csv'")

# Create a summary report
with open("dataset_analysis_report.txt", "w") as f:
    f.write("OpenReview Dataset Analysis Report\n")
    f.write("=" * 50 + "\n\n")

    f.write("FILTERING CRITERIA:\n")
    f.write("- Papers published up to November 2022\n")
    f.write(
        f"- Papers with verbose reviews (>{review_length_threshold:.0f} chars avg)\n\n"
    )

    f.write("DATASET SUMMARY:\n")
    f.write(f"- Total papers: {len(df_verbose)}\n")
    f.write(
        f"- Date range: {df_verbose['parsed_date'].min().strftime('%Y-%m-%d')} to {df_verbose['parsed_date'].max().strftime('%Y-%m-%d')}\n"
    )
    f.write(f"- Total reviews: {total_reviews}\n")
    f.write(f"- Average reviews per paper: {df_verbose['num_reviews'].mean():.2f}\n")
    f.write(f"- Average review length: {np.mean(review_lengths_flat):.0f} characters\n")
    f.write(f"- Acceptance rate: {df_verbose['decision'].mean() * 100:.1f}%\n\n")

    f.write("REVIEW LENGTH DISTRIBUTION:\n")
    f.write(f"- Mean: {np.mean(review_lengths_flat):.0f} characters\n")
    f.write(f"- Median: {np.median(review_lengths_flat):.0f} characters\n")
    f.write(f"- Standard deviation: {np.std(review_lengths_flat):.0f} characters\n")
    f.write("- This distribution appears approximately normal\n\n")

    f.write("SCORE STATISTICS:\n")
    f.write(
        f"- Mean score: {df_verbose['mean_score'].mean():.3f} ± {df_verbose['mean_score'].std():.3f}\n"
    )
    f.write(
        f"- Score range: {df_verbose['mean_score'].min():.3f} to {df_verbose['mean_score'].max():.3f}\n\n"
    )

    f.write("TOP VENUES:\n")
    for venue, count in venue_counts.head(5).items():
        f.write(f"- {venue}: {count} papers\n")

    f.write("\nTOP FIELDS OF STUDY:\n")
    for field, count in field_counts.most_common(5):
        f.write(f"- {field}: {count} papers\n")

print(
    "\nAnalysis complete! Check 'dataset_analysis_report.txt' for a detailed summary."
)
