# When the Description Field Does the Work: A Netflix Content Recommender

A TF-IDF recommender over 8,807 Netflix titles — description text plus genre tags — that produces top-10 lists reading like a curator picked them. Stranger Things returns Nightflyers, Helix, and Chilling Adventures of Sabrina. Squid Game returns Kakegurui and Til Death Do Us Part. Black Mirror returns a cluster of British TV dramas, which is the interesting failure mode.

## Key results

| Metric | Value |
| --- | ---: |
| Catalog rows | 8,807 |
| Movies | 6,131 |
| TV Shows | 2,676 |
| TF-IDF vocabulary | 10,371 |
| Same-type top-10 rate (text-only) | 99.3% |
| Same-type top-10 rate (type blend w=0.15) | 99.9% |

Unlike the Spotify audio-similarity case, where the naive recommender leaks across genres, Netflix's description text carries enough Movie-vs-Show signal that the recommender respects the type boundary almost automatically. The real failure mode is genre-tag shadowing — titles with a dominant tag like "British TV Shows" get pulled into a neighborhood of other British shows even when the description vibe points elsewhere.

## What is in this repo

`src/run_analysis.py` is the end-to-end pipeline: load, vectorise, fit the recommender, evaluate on 500 random queries, generate the static figures and a type-weight animation. `notebooks/netflix-content-recommender-analysis.ipynb` is the narrative walk-through. `src/_palette.py` is the project palette (merlot + popcorn yellow). `figures/` holds the catalog-by-year chart, top-genres bar, similarity heatmap, type-retention histogram, type-weight animation, and the hero card. `outputs/` holds the top-10 demo lists per query title as CSV.

`REPORT.md` is the long-form written analysis.

## How to reproduce

The dataset is the Kaggle `netflix_titles.csv` from Shivam Bansal's Netflix movies and TV shows dataset. Download from <https://www.kaggle.com/datasets/shivamb/netflix-shows> and place it at `data/netflix_titles.csv`.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_analysis.py --data data/netflix_titles.csv --figures figures --outputs outputs
```

Runs in under thirty seconds on a modern CPU.

## Further reading

Full write-up with narrative and figures: <https://ndjstn.github.io/posts/netflix-content-recommender/>.

## License

MIT. See [LICENSE](LICENSE).
