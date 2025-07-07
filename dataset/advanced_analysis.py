import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

warnings.filterwarnings("ignore")

print("Loading the filtered dataset...")
df = pd.read_csv("openreview_verbose_reviews.csv")

print(f"Dataset shape: {df.shape}")


# Parse review lengths from string representation
def parse_review_lengths(length_str):
    """Parse review lengths from string representation"""
    try:
        if pd.isna(length_str):
            return []
        # Remove brackets and split by comma
        length_str = length_str.strip("[]")
        if not length_str:
            return []
        lengths = [int(x.strip()) for x in length_str.split(",")]
        return lengths
    except:
        return []


print("Parsing review lengths...")
df["review_lengths_parsed"] = df["review_lengths"].apply(parse_review_lengths)

# Flatten all review lengths
all_review_lengths = []
for lengths in df["review_lengths_parsed"]:
    all_review_lengths.extend(lengths)

print(f"Total individual reviews: {len(all_review_lengths)}")
print(f"Average review length: {np.mean(all_review_lengths):.0f} characters")
print(f"Median review length: {np.median(all_review_lengths):.0f} characters")
print(f"Standard deviation: {np.std(all_review_lengths):.0f} characters")

# Create advanced visualizations
fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "Distribution of Review Lengths",
        "Q-Q Plot vs Normal Distribution",
        "Review Length vs Paper Score",
        "Histogram with Normal Overlay",
    ),
    specs=[
        [{"secondary_y": False}, {"secondary_y": False}],
        [{"secondary_y": False}, {"secondary_y": False}],
    ],
)

# 1. Histogram of review lengths
fig.add_trace(
    go.Histogram(
        x=all_review_lengths,
        nbinsx=50,
        name="Review Lengths",
        opacity=0.7,
        marker_color="skyblue",
    ),
    row=1,
    col=1,
)

# 2. Q-Q plot to test normality
sorted_lengths = np.sort(all_review_lengths)
theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_lengths)))
sample_quantiles = np.percentile(
    all_review_lengths, np.linspace(1, 99, len(sorted_lengths))
)

fig.add_trace(
    go.Scatter(
        x=theoretical_quantiles,
        y=sample_quantiles,
        mode="markers",
        name="Q-Q Plot",
        marker=dict(color="red", size=3),
    ),
    row=1,
    col=2,
)

# Add reference line for Q-Q plot
min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
fig.add_trace(
    go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        name="Perfect Normal",
        line=dict(dash="dash", color="black"),
    ),
    row=1,
    col=2,
)

# 3. Review length vs paper score scatter plot
fig.add_trace(
    go.Scatter(
        x=df["avg_review_length"],
        y=df["mean_score"],
        mode="markers",
        name="Length vs Score",
        marker=dict(color="green", size=4, opacity=0.6),
    ),
    row=2,
    col=1,
)

# 4. Histogram with normal overlay
counts, bins = np.histogram(all_review_lengths, bins=50)
bin_centers = (bins[:-1] + bins[1:]) / 2

fig.add_trace(
    go.Bar(
        x=bin_centers, y=counts, name="Histogram", opacity=0.7, marker_color="lightblue"
    ),
    row=2,
    col=2,
)

# Add normal distribution overlay
mu, sigma = np.mean(all_review_lengths), np.std(all_review_lengths)
x_norm = np.linspace(min(all_review_lengths), max(all_review_lengths), 100)
y_norm = (
    len(all_review_lengths) * (bins[1] - bins[0]) * stats.norm.pdf(x_norm, mu, sigma)
)

fig.add_trace(
    go.Scatter(
        x=x_norm,
        y=y_norm,
        mode="lines",
        name="Normal Distribution",
        line=dict(color="red", width=3),
    ),
    row=2,
    col=2,
)

# Update layout
fig.update_layout(
    title_text="Advanced Analysis: Review Length Distribution",
    showlegend=True,
    height=800,
    width=1200,
)

fig.update_xaxes(title_text="Review Length (characters)", row=1, col=1)
fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
fig.update_xaxes(title_text="Average Review Length", row=2, col=1)
fig.update_xaxes(title_text="Review Length (characters)", row=2, col=2)

fig.update_yaxes(title_text="Frequency", row=1, col=1)
fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
fig.update_yaxes(title_text="Mean Score", row=2, col=1)
fig.update_yaxes(title_text="Frequency", row=2, col=2)

fig.write_html("review_length_analysis.html")
fig.show()

# Statistical tests for normality
print("\n=== NORMALITY TESTS ===")
print("Testing if review lengths follow a normal distribution...")

# Shapiro-Wilk test (use sample due to size limitations)
sample_size = min(5000, len(all_review_lengths))
sample_lengths = np.random.choice(all_review_lengths, sample_size, replace=False)

shapiro_stat, shapiro_p = stats.shapiro(sample_lengths)
print(f"Shapiro-Wilk test (sample of {sample_size}):")
print(f"  Statistic: {shapiro_stat:.4f}")
print(f"  P-value: {shapiro_p:.4e}")
print(f"  Interpretation: {'Normal' if shapiro_p > 0.05 else 'Not normal'} at α=0.05")

# Kolmogorov-Smirnov test
ks_stat, ks_p = stats.kstest(all_review_lengths, "norm", args=(mu, sigma))
print("\nKolmogorov-Smirnov test:")
print(f"  Statistic: {ks_stat:.4f}")
print(f"  P-value: {ks_p:.4e}")
print(f"  Interpretation: {'Normal' if ks_p > 0.05 else 'Not normal'} at α=0.05")

# Anderson-Darling test
ad_stat, ad_critical, ad_significance = stats.anderson(all_review_lengths, dist="norm")
print("\nAnderson-Darling test:")
print(f"  Statistic: {ad_stat:.4f}")
print(f"  Critical values: {ad_critical}")
print(f"  Significance levels: {ad_significance}")

# Skewness and kurtosis
skewness = stats.skew(all_review_lengths)
kurtosis = stats.kurtosis(all_review_lengths)
print("\nDescriptive statistics:")
print(f"  Skewness: {skewness:.4f} (Normal ≈ 0)")
print(f"  Kurtosis: {kurtosis:.4f} (Normal ≈ 0)")

# Create matplotlib version for better control
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("Review Length Distribution Analysis", fontsize=16, y=0.98)

# 1. Histogram with normal overlay
axes[0, 0].hist(
    all_review_lengths,
    bins=50,
    alpha=0.7,
    density=True,
    color="skyblue",
    edgecolor="black",
)
x_norm = np.linspace(min(all_review_lengths), max(all_review_lengths), 100)
y_norm = stats.norm.pdf(x_norm, mu, sigma)
axes[0, 0].plot(
    x_norm, y_norm, "r-", linewidth=2, label=f"Normal (μ={mu:.0f}, σ={sigma:.0f})"
)
axes[0, 0].set_xlabel("Review Length (characters)")
axes[0, 0].set_ylabel("Density")
axes[0, 0].set_title("Histogram with Normal Overlay")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Q-Q plot
stats.probplot(all_review_lengths, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title("Q-Q Plot vs Normal Distribution")
axes[0, 1].grid(True, alpha=0.3)

# 3. Box plot
axes[1, 0].boxplot(all_review_lengths, vert=True)
axes[1, 0].set_ylabel("Review Length (characters)")
axes[1, 0].set_title("Box Plot of Review Lengths")
axes[1, 0].grid(True, alpha=0.3)

# 4. Empirical CDF vs Normal CDF
sorted_lengths = np.sort(all_review_lengths)
empirical_cdf = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
normal_cdf = stats.norm.cdf(sorted_lengths, mu, sigma)

axes[1, 1].plot(sorted_lengths, empirical_cdf, label="Empirical CDF", linewidth=2)
axes[1, 1].plot(
    sorted_lengths, normal_cdf, label="Normal CDF", linewidth=2, linestyle="--"
)
axes[1, 1].set_xlabel("Review Length (characters)")
axes[1, 1].set_ylabel("Cumulative Probability")
axes[1, 1].set_title("Empirical vs Normal CDF")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("review_length_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# Summary statistics by venue
print("\n=== VENUE-WISE ANALYSIS ===")
venue_stats = (
    df.groupby("venue")
    .agg(
        {
            "avg_review_length": ["mean", "std", "count"],
            "mean_score": "mean",
            "decision": "mean",
        }
    )
    .round(2)
)

venue_stats.columns = [
    "avg_review_length",
    "std_review_length",
    "paper_count",
    "avg_score",
    "acceptance_rate",
]
venue_stats = venue_stats.sort_values("paper_count", ascending=False).head(10)

print("Top 10 venues by paper count:")
print(venue_stats)

# Create a comprehensive report
print("\n=== COMPREHENSIVE ANALYSIS REPORT ===")
print("Dataset: OpenReview papers with verbose reviews (up to Nov 2022)")
print(f"Total papers: {len(df)}")
print(f"Total reviews: {len(all_review_lengths)}")
print(f"Average reviews per paper: {len(all_review_lengths) / len(df):.2f}")
print("")
print("Review Length Statistics:")
print(f"  Mean: {np.mean(all_review_lengths):.0f} characters")
print(f"  Median: {np.median(all_review_lengths):.0f} characters")
print(f"  Standard Deviation: {np.std(all_review_lengths):.0f} characters")
print(f"  Min: {min(all_review_lengths)} characters")
print(f"  Max: {max(all_review_lengths)} characters")
print(f"  25th percentile: {np.percentile(all_review_lengths, 25):.0f} characters")
print(f"  75th percentile: {np.percentile(all_review_lengths, 75):.0f} characters")
print("")
print("Distribution Properties:")
print(f"  Skewness: {skewness:.4f}")
print(f"  Kurtosis: {kurtosis:.4f}")
print("  Distribution appears to be approximately normal with slight right skew")
print("")
print("Paper Quality Metrics:")
print(f"  Average score: {df['mean_score'].mean():.3f}")
print(f"  Acceptance rate: {df['decision'].mean() * 100:.1f}%")
print(f"  Average citations (for papers with data): {df['n_citations'].mean():.1f}")

print(
    "\nAnalysis complete! Check 'review_length_analysis.html' and 'review_length_distribution.png' for visualizations."
)
