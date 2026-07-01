# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Music Artist Recommendation System — Complete Script
# Item-based CF, User-based CF, Content-based (tag) filtering
# Built on the Last.fm HetRec 2011 dataset
# ─────────────────────────────────────────────────────────────────────────────

# Install dependencies (uncomment if running for the first time)
# !pip install pandas numpy scikit-learn matplotlib seaborn

import os
import pandas as pd
import numpy as np

# ── Dataset path ──────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

def data_path(filename):
    return os.path.join(DATA_DIR, filename)

# ── Load and explore user_artists.dat ────────────────────────────────────────
user_artists_df = pd.read_csv(data_path('user_artists.dat'), sep='\t')
print("--- user_artists_df Info ---")
user_artists_df.info()
print("\n--- user_artists_df Head ---")
print(user_artists_df.head())
print(f"\nUnique users: {user_artists_df['userID'].nunique()}")
print(f"Unique artists: {user_artists_df['artistID'].nunique()}")
print("\nMissing values in user_artists_df:")
print(user_artists_df.isnull().sum())

# ── Load and explore artists.dat ─────────────────────────────────────────────
artists_df = pd.read_csv(data_path('artists.dat'), sep='\t')
print("\n\n--- artists_df Info ---")
artists_df.info()
print("\n--- artists_df Head ---")
print(artists_df.head())
print(f"\nUnique artists in artists_df: {artists_df['id'].nunique()}")
print("\nMissing values in artists_df:")
print(artists_df.isnull().sum())

# ── Load and explore tags.dat ─────────────────────────────────────────────────
tags_df = pd.read_csv(data_path('tags.dat'), sep='\t', encoding='latin1')
print("\n\n--- tags_df Info ---")
tags_df.info()
print("\n--- tags_df Head ---")
print(tags_df.head())
print(f"\nUnique tags: {tags_df['tagID'].nunique()}")
print("\nMissing values in tags_df:")
print(tags_df.isnull().sum())

# ── Load and explore user_taggedartists.dat ───────────────────────────────────
user_taggedartists_df = pd.read_csv(data_path('user_taggedartists.dat'), sep='\t', encoding='latin1')
print("\n\n--- user_taggedartists_df Info ---")
user_taggedartists_df.info()
print("\n--- user_taggedartists_df Head ---")
print(user_taggedartists_df.head())
print(f"\nUnique users in user_taggedartists_df: {user_taggedartists_df['userID'].nunique()}")
print(f"Unique artists in user_taggedartists_df: {user_taggedartists_df['artistID'].nunique()}")
print(f"Unique tags in user_taggedartists_df: {user_taggedartists_df['tagID'].nunique()}")
print("\nMissing values in user_taggedartists_df:")
print(user_taggedartists_df.isnull().sum())

# ── Load and explore user_friends.dat ────────────────────────────────────────
user_friends_df = pd.read_csv(data_path('user_friends.dat'), sep='\t', encoding='latin1')
print("\n\n--- user_friends_df Info ---")
user_friends_df.info()
print("\n--- user_friends_df Head ---")
print(user_friends_df.head())
print(f"\nUnique users in user_friends_df: {user_friends_df['userID'].nunique()}")
print(f"Unique friends in user_friends_df: {user_friends_df['friendID'].nunique()}")
print("\nMissing values in user_friends_df:")
print(user_friends_df.isnull().sum())

# ── Load and explore user_taggedartists-timestamps.dat ───────────────────────
user_taggedartists_timestamps_df = pd.read_csv(data_path('user_taggedartists-timestamps.dat'), sep='\t', encoding='latin1')
print("\n\n--- user_taggedartists-timestamps_df Info ---")
user_taggedartists_timestamps_df.info()
print("\n--- user_taggedartists-timestamps_df Head ---")
print(user_taggedartists_timestamps_df.head())
print("\nMissing values in user_taggedartists-timestamps_df:")
print(user_taggedartists_timestamps_df.isnull().sum())


# ── 1. Rename 'id' column in artists_df to 'artistID' for consistent merging ─
artists_df.rename(columns={'id': 'artistID'}, inplace=True)
print("--- artists_df after renaming 'id' to 'artistID' ---")
print(artists_df.head())

# ── 2. Merge user_artists_df with artists_df to add artist names ──────────────
df_interactions = pd.merge(user_artists_df, artists_df[['artistID', 'name']],
                           on='artistID', how='left')
print("\n--- df_interactions (merged) Head ---")
print(df_interactions.head())
print("\n--- df_interactions Info ---")
df_interactions.info()
print("\nMissing artist names after merge (should be 0 or very low):", df_interactions['name'].isnull().sum())

# ── 3. Consolidate duplicate user-artist interactions by summing play counts ──
duplicate_interactions_count = df_interactions.duplicated(subset=['userID', 'artistID']).sum()
print(f"\nDuplicate user-artist interactions before consolidation: {duplicate_interactions_count}")

if duplicate_interactions_count > 0:
    print("Consolidating duplicates by summing 'weight'...")
    df_interactions_clean = df_interactions.groupby(['userID', 'artistID', 'name'], as_index=False)['weight'].sum()
    print(f"Unique user-artist interactions after consolidation: {len(df_interactions_clean)}")
else:
    print("No duplicates found.")
    df_interactions_clean = df_interactions.copy()

# Keep only artists with at least 20 users — reduces matrix size significantly
min_users = 20
popular_artists = user_artists_df.groupby('artistID')['userID'].nunique()
popular_artist_ids = popular_artists[popular_artists >= min_users].index
df_interactions_clean = df_interactions_clean[df_interactions_clean['artistID'].isin(popular_artist_ids)]


# ── Enrich tag data with artist names ────────────────────────────────────────
df_user_artist_tags = pd.merge(user_taggedartists_df, tags_df, on='tagID', how='left')
print("--- df_user_artist_tags (merged with tags) Head ---")
print(df_user_artist_tags.head())
print("\nMissing tag values after merge:", df_user_artist_tags['tagValue'].isnull().sum())

df_user_artist_tags_with_names = pd.merge(df_user_artist_tags, artists_df[['artistID', 'name']],
                                          on='artistID', how='left')
print("\n--- df_user_artist_tags_with_names Head ---")
print(df_user_artist_tags_with_names.head())
print("\nMissing artist names in tagged data:", df_user_artist_tags_with_names['name'].isnull().sum())

# ── Parse timestamps (milliseconds → datetime) ────────────────────────────────
user_taggedartists_timestamps_df['datetime'] = pd.to_datetime(
    user_taggedartists_timestamps_df['timestamp'], unit='ms'
)
print("\n--- user_taggedartists-timestamps_df with datetime column ---")
print(user_taggedartists_timestamps_df.head())


# ── Clean friend relationships (remove exact duplicates) ─────────────────────
print("--- user_friends_df Head ---")
print(user_friends_df.head())
user_friends_df.info()
print("\nMissing values:", user_friends_df.isnull().sum().to_dict())

duplicate_friends = user_friends_df.duplicated().sum()
print(f"\nDuplicate friend relationships: {duplicate_friends}")

if duplicate_friends > 0:
    user_friends_df_clean = user_friends_df.drop_duplicates().reset_index(drop=True)
    print(f"Friend relationships after deduplication: {len(user_friends_df_clean)}")
else:
    user_friends_df_clean = user_friends_df.copy()


# ── Train/test split ───────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split

X = df_interactions_clean[['userID', 'artistID']]
y = df_interactions_clean['weight']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("--- Data Split Summary ---")
print(f"Total interactions : {len(df_interactions_clean)}")
print(f"Training (75%)     : {len(X_train)}")
print(f"Testing  (25%)     : {len(X_test)}")


# ── Item-based Collaborative Filtering ────────────────────────────────────────
user_item_matrix = df_interactions_clean.pivot_table(
    index='userID', columns='artistID', values='weight'
).fillna(0)

print("--- User-Item Matrix (first 5×5) ---")
print(user_item_matrix.iloc[:5, :5])
print(f"\nShape: {user_item_matrix.shape}  (users × artists)")

artist_ids = user_item_matrix.columns

from sklearn.metrics.pairwise import cosine_similarity

item_similarity_matrix = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(
    item_similarity_matrix,
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)

print("--- Item-Item Cosine Similarity Matrix (first 5×5) ---")
print(item_similarity_df.iloc[:5, :5])
print(f"\nShape: {item_similarity_df.shape}")


def get_item_based_recommendations(user_id, num_recommendations=10):
    """Return top-N artist recommendations for a user using item-based CF."""
    user_listened_artists = df_interactions_clean[df_interactions_clean['userID'] == user_id]

    if user_listened_artists.empty:
        print(f"User {user_id} is a cold-start user (no listening history).")
        return []

    user_artist_weights = dict(zip(user_listened_artists['artistID'], user_listened_artists['weight']))
    recommendation_scores = {}

    for listened_artist_id, listened_weight in user_artist_weights.items():
        if listened_artist_id not in item_similarity_df.columns:
            continue

        similar_artists_scores = item_similarity_df[listened_artist_id]

        for similar_artist_id, similarity_score in similar_artists_scores.items():
            if similar_artist_id == listened_artist_id:
                continue
            if similar_artist_id in user_artist_weights:
                continue

            score = similarity_score * listened_weight
            recommendation_scores[similar_artist_id] = recommendation_scores.get(similar_artist_id, 0) + score

    top_recommendations = sorted(
        [(score, aid) for aid, score in recommendation_scores.items() if score > 0],
        reverse=True
    )[:num_recommendations]

    artist_id_to_name = dict(zip(artists_df['artistID'], artists_df['name']))
    return [(artist_id_to_name.get(aid, f"Artist ID {aid}"), score) for score, aid in top_recommendations]


print("--- Item-Based Recommendations ---")
for user_id in [2, 5, 170, 3000]:
    print(f"\nRecommendations for User {user_id}:")
    recs = get_item_based_recommendations(user_id, num_recommendations=5)
    if recs:
        for artist_name, score in recs:
            print(f"  - {artist_name} (Score: {score:.2f})")
    else:
        print("  No recommendations available.")


# ── User-based Collaborative Filtering ────────────────────────────────────────
user_item_matrix_f32 = user_item_matrix.astype('float32')

user_similarity_matrix = cosine_similarity(user_item_matrix_f32)
user_similarity_df = pd.DataFrame(
    user_similarity_matrix,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)

print("--- User-User Cosine Similarity Matrix (first 5×5) ---")
print(user_similarity_df.iloc[:5, :5])
print(f"\nShape: {user_similarity_df.shape}")


def get_user_based_recommendations(user_id, num_recommendations=10, k_neighbors=50):
    """Return top-N artist recommendations for a user using user-based CF."""
    user_listened_artists_ids = set(
        df_interactions_clean[df_interactions_clean['userID'] == user_id]['artistID']
    )

    if user_id not in user_similarity_df.index:
        print(f"User {user_id} not in similarity matrix (cold-start).")
        return []

    user_similarities = user_similarity_df[user_id].drop(user_id)
    top_k_similar_users = user_similarities[user_similarities > 0].nlargest(k_neighbors).index.tolist()

    if not top_k_similar_users:
        print(f"No similar users found for User {user_id}.")
        return []

    recommendation_scores = {}
    artist_id_to_name = dict(zip(artists_df['artistID'], artists_df['name']))

    for similar_user_id in top_k_similar_users:
        similarity_score = user_similarity_df.loc[user_id, similar_user_id]
        similar_user_artists = df_interactions_clean[df_interactions_clean['userID'] == similar_user_id]

        for _, row in similar_user_artists.iterrows():
            artist_id = row['artistID']
            if artist_id in user_listened_artists_ids:
                continue
            score = similarity_score * row['weight']
            recommendation_scores[artist_id] = recommendation_scores.get(artist_id, 0) + score

    top_recommendations = sorted(
        [(score, aid) for aid, score in recommendation_scores.items() if score > 0],
        reverse=True
    )[:num_recommendations]

    return [(artist_id_to_name.get(aid, f"Artist ID {aid}"), score) for score, aid in top_recommendations]


print("--- User-Based Recommendations ---")
for user_id in [2, 5, 170]:
    print(f"\nRecommendations for User {user_id}:")
    recs = get_user_based_recommendations(user_id, num_recommendations=5, k_neighbors=50)
    if recs:
        for artist_name, score in recs:
            print(f"  - {artist_name} (Score: {score:.2f})")
    else:
        print("  No recommendations available.")


# ── Evaluation — Predicting Play Counts ───────────────────────────────────────

def predict_item_based(user_id, artist_id):
    """Predict play-count weight for a user-artist pair using item-based CF."""
    if artist_id not in item_similarity_df.columns or user_id not in user_item_matrix.index:
        return 0

    user_listened_artists = df_interactions_clean[df_interactions_clean['userID'] == user_id]
    if user_listened_artists.empty:
        return 0

    user_artist_weights = dict(zip(user_listened_artists['artistID'], user_listened_artists['weight']))
    target_artist_similarities = item_similarity_df[artist_id]

    numerator = 0
    denominator = 0

    for listened_artist_id, listened_weight in user_artist_weights.items():
        if listened_artist_id not in target_artist_similarities.index:
            continue
        sim = target_artist_similarities[listened_artist_id]
        if sim > 0:
            numerator   += sim * listened_weight
            denominator += sim

    return numerator / denominator if denominator > 0 else 0


def predict_user_based(user_id, artist_id, k_neighbors=50):
    """Predict play-count weight for a user-artist pair using user-based CF."""
    if user_id not in user_similarity_df.index:
        return 0

    user_similarities = user_similarity_df[user_id].drop(user_id)
    top_k = user_similarities[user_similarities > 0].nlargest(k_neighbors)

    numerator = 0
    denominator = 0

    for neighbor_id, sim in top_k.items():
        neighbor_data = df_interactions_clean[
            (df_interactions_clean['userID'] == neighbor_id) &
            (df_interactions_clean['artistID'] == artist_id)
        ]
        if not neighbor_data.empty:
            weight = neighbor_data.iloc[0]['weight']
            numerator   += sim * weight
            denominator += sim

    return numerator / denominator if denominator > 0 else 0


from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

EVAL_SAMPLE = 500
test_sample = pd.concat([X_test, y_test], axis=1).sample(n=min(EVAL_SAMPLE, len(X_test)), random_state=42)

item_preds = [
    predict_item_based(row['userID'], row['artistID'])
    for _, row in test_sample.iterrows()
]
item_mae  = mean_absolute_error(test_sample['weight'], item_preds)
item_rmse = root_mean_squared_error(test_sample['weight'], item_preds)

user_preds = [
    predict_user_based(row['userID'], row['artistID'])
    for _, row in test_sample.iterrows()
]
user_mae  = mean_absolute_error(test_sample['weight'], user_preds)
user_rmse = root_mean_squared_error(test_sample['weight'], user_preds)

print("--- Evaluation Results (sample of", EVAL_SAMPLE, "interactions) ---")
print(f"Item-based CF  |  MAE: {item_mae:.2f}  |  RMSE: {item_rmse:.2f}")
print(f"User-based CF  |  MAE: {user_mae:.2f}  |  RMSE: {user_rmse:.2f}")


# ★★★ NEW — Visualization 1: Evaluation Comparison Chart ★★★
import matplotlib.pyplot as plt

methods = ['Item-based CF', 'User-based CF']
mae_scores = [item_mae, user_mae]
rmse_scores = [item_rmse, user_rmse]

x = np.arange(len(methods))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(x - width/2, mae_scores, width, label='MAE')
ax.bar(x + width/2, rmse_scores, width, label='RMSE')
ax.set_ylabel('Error')
ax.set_title('Prediction Error by Method')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
plt.tight_layout()
plt.savefig('evaluation_comparison.png', dpi=150)
plt.show()


# ── Explainability ─────────────────────────────────────────────────────────────

def explain_item_based_recommendation(user_id, recommended_artist_id, num_reasons=3):
    """Explain an item-based recommendation by showing which listened artists drove it."""
    recommended_artist_name = artists_df[artists_df['artistID'] == recommended_artist_id]['name'].iloc[0]
    print(f"\n--- Why was '{recommended_artist_name}' recommended to User {user_id}? ---")

    user_listened = df_interactions_clean[df_interactions_clean['userID'] == user_id]
    if user_listened.empty:
        print("No listening history found.")
        return

    if recommended_artist_id not in item_similarity_df.columns:
        print("Recommended artist not in similarity matrix.")
        return

    similarities_to_recommended = item_similarity_df[recommended_artist_id]
    artist_id_to_name = dict(zip(artists_df['artistID'], artists_df['name']))

    driving_artists = []
    for _, row in user_listened.iterrows():
        listened_artist_id = row['artistID']
        if listened_artist_id in similarities_to_recommended.index:
            sim = similarities_to_recommended[listened_artist_id]
            if sim > 0:
                driving_artists.append((sim, listened_artist_id, row['weight']))

    driving_artists.sort(reverse=True)
    print(f"Top reasons (artists you've played that are similar to '{recommended_artist_name}'):")
    for sim, aid, weight in driving_artists[:num_reasons]:
        name = artist_id_to_name.get(aid, f"Artist {aid}")
        print(f"  - '{name}'  (play count: {weight},  similarity: {sim:.2f})")


for artist_name in ['Sufjan Stevens', 'Kreator']:
    row = artists_df[artists_df['name'] == artist_name]
    if not row.empty:
        explain_item_based_recommendation(user_id=5, recommended_artist_id=row['artistID'].iloc[0])


def explain_user_based_recommendation(user_id, recommended_artist_id, num_reasons=3, k_neighbors=5):
    """Explain a user-based recommendation by showing which similar users listened to the artist."""
    recommended_artist_name = artists_df[artists_df['artistID'] == recommended_artist_id]['name'].iloc[0]
    print(f"\n--- Why was '{recommended_artist_name}' recommended to User {user_id}? ---")

    if user_id not in user_similarity_df.index:
        print("User not in similarity matrix.")
        return

    user_listened_ids = set(df_interactions_clean[df_interactions_clean['userID'] == user_id]['artistID'])
    user_similarities = user_similarity_df[user_id].drop(user_id)
    top_neighbors = user_similarities[user_similarities > 0].nlargest(k_neighbors).index.tolist()
    artist_id_to_name = dict(zip(df_interactions_clean['artistID'], df_interactions_clean['name']))

    print(f"Recommendation driven by users similar to you who listened to '{recommended_artist_name}':")
    reasons_shown = 0

    for neighbor_id in top_neighbors:
        sim = user_similarity_df.loc[user_id, neighbor_id]
        neighbor_artists = df_interactions_clean[df_interactions_clean['userID'] == neighbor_id]
        neighbor_artist_ids = set(neighbor_artists['artistID'])

        if recommended_artist_id in neighbor_artist_ids:
            common = user_listened_ids.intersection(neighbor_artist_ids)
            common_names = [artist_id_to_name.get(aid, str(aid)) for aid in list(common)[:3]]
            print(f"  - User {neighbor_id} (similarity: {sim:.2f}) — you both enjoy: {', '.join(common_names)}")
            reasons_shown += 1
            if reasons_shown >= num_reasons:
                break


for artist_name in ['U2', 'Coldplay', 'Pink Floyd']:
    row = artists_df[artists_df['name'] == artist_name]
    if not row.empty:
        explain_user_based_recommendation(user_id=2, recommended_artist_id=row['artistID'].iloc[0])


# ── Bonus 1 — Smart Playlist Generator (Item-to-Item) ────────────────────────

def generate_smart_playlist_item_to_item(seed_artist_name, num_songs=7):
    """Generate a playlist of artists similar to the seed artist (item-to-item CF)."""
    print(f"\n--- Playlist based on '{seed_artist_name}' (Item-to-Item CF) ---")

    seed_row = artists_df[artists_df['name'].str.lower() == seed_artist_name.lower()]
    if seed_row.empty:
        print(f"Artist '{seed_artist_name}' not found.")
        return []

    seed_id = seed_row['artistID'].iloc[0]
    if seed_id not in item_similarity_df.columns:
        print(f"'{seed_artist_name}' not in similarity matrix (too few interactions?).")
        return []

    scores = item_similarity_df[seed_id].drop(seed_id)
    top = scores[scores > 0].nlargest(num_songs)

    artist_id_to_name = dict(zip(artists_df['artistID'], artists_df['name']))
    return [f"{artist_id_to_name.get(aid, aid)} (similarity: {s:.2f})" for aid, s in top.items()]


for seed in ['Duran Duran', 'Radiohead', 'Metallica', 'Beirut', 'Britney Spears']:
    playlist = generate_smart_playlist_item_to_item(seed, num_songs=7)
    if playlist:
        print('\n'.join(f'  {t}' for t in playlist))


# ── Bonus 2 — Content-Based Filtering (Tag Similarity) ────────────────────────

artist_tag_counts = df_user_artist_tags_with_names.groupby(
    ['artistID', 'tagValue']
).size().reset_index(name='tag_count')

artist_tag_matrix = artist_tag_counts.pivot_table(
    index='artistID', columns='tagValue', values='tag_count'
).fillna(0)

print("--- Artist-Tag Matrix (first 5×5) ---")
print(artist_tag_matrix.iloc[:5, :5])
print(f"\nShape: {artist_tag_matrix.shape}  (artists × unique tags)")

tag_sim_matrix = cosine_similarity(artist_tag_matrix)
tag_based_artist_similarity_df = pd.DataFrame(
    tag_sim_matrix,
    index=artist_tag_matrix.index,
    columns=artist_tag_matrix.index
)

print("\n--- Tag-Based Artist Similarity (first 5×5) ---")
print(tag_based_artist_similarity_df.iloc[:5, :5])


def generate_tag_based_playlist(seed_artist_name, num_songs=7):
    """Generate a playlist of artists sharing similar tags to the seed artist."""
    print(f"\n--- Tag-Based Playlist: '{seed_artist_name}' ---")

    seed_row = artists_df[artists_df['name'].str.lower() == seed_artist_name.lower()]
    if seed_row.empty:
        print(f"Artist '{seed_artist_name}' not found.")
        return []

    seed_id = seed_row['artistID'].iloc[0]
    if seed_id not in tag_based_artist_similarity_df.columns:
        print(f"'{seed_artist_name}' not in tag similarity matrix (no tags?).")
        return []

    scores = tag_based_artist_similarity_df[seed_id].drop(seed_id)
    top = scores[scores > 0].nlargest(num_songs)

    artist_id_to_name = dict(zip(artists_df['artistID'], artists_df['name']))
    return [f"{artist_id_to_name.get(aid, aid)} (tag similarity: {s:.2f})" for aid, s in top.items()]


for seed in ['Metallica', 'Radiohead', 'Britney Spears', 'Beirut']:
    playlist = generate_tag_based_playlist(seed, num_songs=5)
    if playlist:
        print('\n'.join(f'  {t}' for t in playlist))


# ★★★ NEW — Visualization 2: Artist Similarity Map (Tag-Based, t-SNE) ★★★
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Use the top 300 most-tagged artists for a clear, fast-to-compute plot
top_artists = artist_tag_counts.groupby('artistID')['tag_count'].sum().nlargest(300).index
subset_matrix = artist_tag_matrix.loc[artist_tag_matrix.index.isin(top_artists)]

# PCA first: reduce from ~9,749 tag dimensions down to 50 before t-SNE
# This is the standard approach for high-dimensional data — same visual output, fraction of the time
n_components_pca = min(50, subset_matrix.shape[0], subset_matrix.shape[1])
pca = PCA(n_components=n_components_pca)
subset_reduced = pca.fit_transform(subset_matrix.values)

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
coords = tsne.fit_transform(subset_reduced)

artist_id_to_name = dict(zip(artists_df['artistID'], artists_df['name']))
names = [artist_id_to_name.get(aid, str(aid)) for aid in subset_matrix.index]

plt.figure(figsize=(10, 8))
plt.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=20)

# Label a few well-known artists for orientation/context
highlight = ['Metallica', 'Radiohead', 'Britney Spears', 'Beirut', 'Coldplay', 'Megadeth', 'Madonna']
for i, name in enumerate(names):
    if name in highlight:
        plt.annotate(name, (coords[i, 0], coords[i, 1]), fontsize=9, fontweight='bold')

plt.title('Artist Similarity Map (Tag-Based, t-SNE Projection)')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.tight_layout()
plt.savefig('artist_similarity_map.png', dpi=150)
plt.show()

