from src.analysis import (plot_rating_distribution, plot_genre_treemap,
                          plot_wordcloud, plot_activity_over_time,
                          plot_popularity_vs_quality, plot_genre_violin,
                          plot_movies_by_year, plot_wordcloud_by_year)
from src.recommenders.deep_learning import DeepLearningRecommender
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.collaborative import CollaborativeRecommender
from src.recommenders.popularity import PopularityRecommender
from src.preprocess import preprocess_data
from src.data_loader import load_data, load_tags_data
from src.utils import (ensure_directories, PROCESSED_MOVIES_PATH,
                       PROCESSED_RATINGS_PATH, save_fig)
import sys
import os

# Fix path import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    print("🚀 START TRAINING PIPELINE...")
    ensure_directories()

    # --- 1. Load & Process Data ---
    print("\n[1/4] Loading and preprocessing data...")
    # By default we limit ratings rows for memory safety. To load the full
    # MovieLens-25M ratings set, set environment variable USE_FULL_RATINGS=1
    use_full = os.environ.get('USE_FULL_RATINGS', '0') == '1'
    ratings_nrows = None if use_full else 500000
    if use_full:
        print("⚠️ Loading full ratings file (MovieLens-25M). Ensure you have enough RAM.")
    movies, ratings = load_data(ratings_nrows)
    tags = load_tags_data()

    # Decide whether to sample/process all users.
    # KEEP_ALL_USERS=1 will preserve all users/movies (recommended for correctness with MovieLens-25M)
    # SAMPLE_SIZE can limit the number of rating rows (for memory safety) — if omitted the full filtered dataset is used.
    keep_all = os.environ.get('KEEP_ALL_USERS', '1') == '1'
    sample_size_env = os.environ.get('SAMPLE_SIZE', None)
    try:
        sample_size = int(
            sample_size_env) if sample_size_env is not None else None
    except ValueError:
        sample_size = None

    movies_proc, ratings_proc, tfidf, mappings = preprocess_data(
        movies, ratings, sample_size=sample_size, keep_all_users=keep_all)

    # Save cleaned data for the app to use
    movies_proc.to_pickle(PROCESSED_MOVIES_PATH)
    ratings_proc.to_pickle(PROCESSED_RATINGS_PATH)
    print("✅ Processed data saved.")

    # --- 2. Train and Save Models ---
    print("\n[2/4] Training and saving models...")
    # Collaborative Filtering (SVD)
    cf = CollaborativeRecommender(ratings_proc)
    cf.train()
    cf.save()
    print("✅ Collaborative Filtering (SVD) model saved.")

    # Content-Based
    cb = ContentBasedRecommender(movies_proc, tfidf)
    cb.save()
    print("✅ Content-Based model saved.")

    # Popularity
    pop = PopularityRecommender(movies_proc, ratings_proc)
    pop.train()
    pop.save()
    print("✅ Popularity model saved.")

    # Deep Learning
    dl = DeepLearningRecommender(
        mappings['num_users'], mappings['num_movies'], mappings)
    dl.train(ratings_proc, epochs=5)  # Use more epochs for better results
    dl.save()
    print("✅ Deep Learning model saved.")

    # --- 3. Generate and Save Visualizations ---
    print("\n[3/4] Generating and saving all 6 visualizations...")
    save_fig(plot_rating_distribution(ratings_proc), "rating_distribution")
    save_fig(plot_activity_over_time(ratings_proc), "activity_over_time")
    # Save the pre-generated 'movie count by year' chart so the app can load
    # the interactive 'movies_by_year.html' instead of rerunning the plotting.
    save_fig(plot_movies_by_year(movies_proc), "movies_by_year")
    save_fig(plot_popularity_vs_quality(ratings_proc), "popularity_vs_quality")
    save_fig(plot_genre_treemap(movies_proc), "genre_treemap")
    save_fig(plot_genre_violin(movies_proc, ratings_proc), "genre_violin")
    save_fig(plot_wordcloud(tags), "tags_wordcloud")

    # --- Save per-year wordclouds for tags (if tags and years available) ---
    try:
        years = sorted(movies_proc['year'].dropna().astype(int).unique())
        # Limit the number of years to avoid excessive asset generation
        for yr in years:
            # generate and save wordcloud for the year
            wc_fig = plot_wordcloud_by_year(tags, movies_proc, yr)
            save_fig(wc_fig, f"tags_wordcloud_{yr}")
    except Exception:
        # don't fail the pipeline if per-year wordcloud generation fails
        print("⚠️ Failed to generate per-year wordclouds — skipping.")

    print("\n[4/4] ✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("\n🎉 Now run the app with: streamlit run streamlit_app/app.py")


if __name__ == "__main__":
    main()
