"""Netflix content-based recommender using TF-IDF over descriptions.

Same idea as the Spotify audio-similarity project but for text. Vectorise each
title's description + genre list, compute cosine similarity, test on a handful
of sample queries. Story: description similarity alone isn't enough; the title
type (Movie / TV Show) has to be respected, and a genre boost helps recall.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _palette import NETFLIX as P, apply_to_mpl  # noqa: E402

sns.set_style("whitegrid")
apply_to_mpl(P)
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "font.size": 11})


def _cmap_native():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("project", [P.bg, P.cover_subtitle, P.accent, P.header_bg])


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--figures", required=True)
    ap.add_argument("--outputs", required=True)
    return ap.parse_args()


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["description"] = df["description"].fillna("")
    df["listed_in"] = df["listed_in"].fillna("")
    df["director"] = df["director"].fillna("")
    df["cast"] = df["cast"].fillna("")
    df["country"] = df["country"].fillna("")
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    return df.reset_index(drop=True)


def content_tokens(row: pd.Series) -> str:
    genres = row["listed_in"].replace(",", " ").lower()
    return f"{row['description']} {genres} {genres}"  # genre tokens twice for slight weight


def main() -> None:
    args = parse_args()
    fig_dir, out_dir = Path(args.figures), Path(args.outputs)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load(args.data)
    print(f"rows: {len(df):,}")

    # Type mix by release year
    mix = df.assign(release_year=df["release_year"].astype("Int64")).groupby(["release_year", "type"]).size().unstack(fill_value=0).sort_index()
    mix = mix.loc[mix.index >= 2000]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.fill_between(mix.index, 0, mix.get("Movie", 0), color=P.accent, alpha=0.75, label="Movie")
    ax.fill_between(mix.index, mix.get("Movie", 0), mix.get("Movie", 0) + mix.get("TV Show", 0), color=P.header_bg, alpha=0.85, label="TV Show")
    ax.set_xlabel("Release year")
    ax.set_ylabel("Titles added")
    ax.set_title("Netflix catalog by release year and content type (since 2000)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "type-mix-by-year.png")
    plt.close(fig)

    # Top genres
    genre_series = df["listed_in"].str.split(", ").explode().value_counts().head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    genre_series.iloc[::-1].plot(kind="barh", ax=ax, color=P.accent)
    ax.set_xlabel("Title count")
    ax.set_title("Top 20 genre tags on Netflix")
    fig.tight_layout()
    fig.savefig(fig_dir / "top-genres.png")
    plt.close(fig)

    # TF-IDF over description + genres
    vec = TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1, 2), min_df=3)
    corpus = df.apply(content_tokens, axis=1).tolist()
    X = vec.fit_transform(corpus)
    print(f"TF-IDF matrix: {X.shape}")

    # Recommender
    def recommend(idx: int, k: int = 10, type_weight: float = 0.0) -> pd.DataFrame:
        q = X[idx]
        # cosine similarity = 1 - distance = dot product on l2-normed; sklearn TfidfVec L2-normalises
        sims = (X @ q.T).toarray().flatten()
        if type_weight > 0:
            q_type = df["type"].iloc[idx]
            sims = sims - (df["type"] != q_type).astype(float).values * type_weight
        sims[idx] = -np.inf
        top = np.argpartition(-sims, k)[:k]
        top = top[np.argsort(-sims[top])]
        out = df.iloc[top][["title", "type", "listed_in", "release_year"]].copy()
        out["similarity"] = sims[top]
        return out.reset_index(drop=True)

    # Demo queries: three well-known titles
    def find(title_substr: str) -> int:
        mask = df["title"].str.contains(title_substr, case=False, regex=False, na=False)
        return int(df.loc[mask].index[0]) if mask.any() else 0

    demo_titles = ["Stranger Things", "The Crown", "Squid Game", "Black Mirror", "The Witcher"]
    demo_results = {}
    for title in demo_titles:
        idx = find(title)
        meta = df.iloc[idx]
        print(f"\nQuery: {meta['title']} ({meta['type']}, genres={meta['listed_in']})")
        recs = recommend(idx, k=10, type_weight=0.0)
        recs_typed = recommend(idx, k=10, type_weight=0.15)
        demo_results[title] = {
            "query": meta.to_dict(),
            "text_only": recs.to_dict(orient="records"),
            "with_type_blend": recs_typed.to_dict(orient="records"),
        }
        recs.to_csv(out_dir / f"recs-{title.lower().replace(' ', '-')}-text-only.csv", index=False)
        recs_typed.to_csv(out_dir / f"recs-{title.lower().replace(' ', '-')}-type-blend.csv", index=False)
        print(recs.head(5).to_string(index=False))

    # Evaluation: 500-query sample of same-type rate
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(df), size=500, replace=False)
    rates_text = []
    rates_blend = []
    for idx in sample_idx:
        q_type = df["type"].iloc[idx]
        recs_t = recommend(int(idx), k=10, type_weight=0.0)
        recs_b = recommend(int(idx), k=10, type_weight=0.15)
        rates_text.append((recs_t["type"] == q_type).mean())
        rates_blend.append((recs_b["type"] == q_type).mean())
    rates_text = np.array(rates_text)
    rates_blend = np.array(rates_blend)
    print(f"\nSame-type rate text-only: {rates_text.mean():.3f}")
    print(f"Same-type rate with type blend: {rates_blend.mean():.3f}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 1.001, 12)
    ax.hist(rates_text, bins=bins, alpha=0.65, label="Text only", color=P.muted)
    ax.hist(rates_blend, bins=bins, alpha=0.8, label="Type blend (w=0.15)", color=P.accent)
    ax.set_xlabel("Share of top-10 that match the query's type (Movie vs TV Show)")
    ax.set_ylabel("Queries")
    ax.set_title("Same-type rate across 500 random queries")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "type-retention.png")
    plt.close(fig)

    # Similarity-matrix heatmap for a small neighborhood
    sample = rng.choice(len(df), size=20, replace=False)
    sim_sample = (X[sample] @ X[sample].T).toarray()
    labels = [t[:25] + ("…" if len(t) > 25 else "") for t in df["title"].iloc[sample]]
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(sim_sample, xticklabels=labels, yticklabels=labels, cmap=_cmap_native(), cbar_kws={"label": "cosine similarity"}, ax=ax)
    ax.set_title("Similarity among 20 sampled titles (cosine over TF-IDF)")
    plt.xticks(rotation=60, ha="right")
    fig.tight_layout()
    fig.savefig(fig_dir / "similarity-matrix-sample.png")
    plt.close(fig)

    # Story animation: for Stranger Things (TV show) AND Pulp Fiction (movie), sweep type weight
    # side-by-side panel so you can see one query stays put and the other pivots.
    story_queries = [("Stranger Things", "TV Show"), ("Pulp Fiction", "Movie")]
    weights = np.linspace(0, 0.3, 16)
    data = {}
    for title, qtype in story_queries:
        idx = find(title)
        mc, sc, tops = [], [], []
        for w in weights:
            r = recommend(idx, k=10, type_weight=float(w))
            mc.append(int((r["type"] == "Movie").sum()))
            sc.append(int((r["type"] == "TV Show").sum()))
            tops.append(r["title"].head(3).tolist())
        data[title] = dict(qtype=qtype, movies=mc, shows=sc, tops=tops)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    fig.subplots_adjust(top=0.82, bottom=0.28, wspace=0.30)
    sup = fig.suptitle("", fontsize=13, fontweight="bold", y=0.96)

    panel_artists = {}
    for ax_, (title, d) in zip(axes, data.items()):
        ax_.set_xlim(-0.5, 1.5); ax_.set_ylim(0, 12)
        ax_.set_xticks([0, 1]); ax_.set_xticklabels(["Movie", "TV Show"], fontsize=11)
        ax_.set_ylabel("Titles in top-10", fontsize=11)
        ax_.set_title(f"{title}  ({d['qtype']})", fontsize=12)
        bars = ax_.bar([0, 1], [0, 0], width=0.6, color=[P.accent, P.header_bg], edgecolor=P.bg, linewidth=1.5)
        count_a = ax_.text(0, 0, "", ha="center", va="bottom", fontsize=12, fontweight="bold")
        count_b = ax_.text(1, 0, "", ha="center", va="bottom", fontsize=12, fontweight="bold")
        tops_txt = fig.text(ax_.get_position().x0 + ax_.get_position().width / 2, 0.09, "", ha="center", fontsize=9, style="italic", color=P.message_text, wrap=True)
        panel_artists[title] = (bars, count_a, count_b, tops_txt)

    def animate(i):
        w = weights[i]
        sup.set_text(f"Type-penalty sweep     weight = {w:.2f}")
        arts = []
        for title, (bars, count_a, count_b, tops_txt) in panel_artists.items():
            d = data[title]
            m, s = d["movies"][i], d["shows"][i]
            bars[0].set_height(m); bars[1].set_height(s)
            count_a.set_position((0, m + 0.3)); count_a.set_text(str(m))
            count_b.set_position((1, s + 0.3)); count_b.set_text(str(s))
            top3 = ", ".join(t[:34] + ("…" if len(t) > 34 else "") for t in d["tops"][i])
            tops_txt.set_text(f"top 3: {top3}")
            arts.extend([*bars, count_a, count_b, tops_txt])
        arts.append(sup)
        return arts

    anim = animation.FuncAnimation(fig, animate, frames=len(weights), interval=500, blit=False)
    anim.save(str(fig_dir / "type-weight-animation.gif"), writer="pillow", fps=2)
    plt.close(fig)

    summary = {
        "rows": int(len(df)),
        "vocab_size": int(len(vec.vocabulary_)),
        "same_type_text_only": float(rates_text.mean()),
        "same_type_blend": float(rates_blend.mean()),
        "demos": demo_results,
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    md = [
        "# Netflix content recommender summary",
        "",
        f"Rows: {len(df):,}. TF-IDF vocabulary: {len(vec.vocabulary_):,}.",
        "",
        "## Type-retention evaluation over 500 random queries",
        "",
        f"Text-only cosine similarity: same-type rate = {rates_text.mean():.3f} (mean).",
        f"Type blend (weight 0.15): same-type rate = {rates_blend.mean():.3f} (mean).",
    ]
    (out_dir / "analysis_summary.md").write_text("\n".join(md))
    print("Done")


if __name__ == "__main__":
    main()
