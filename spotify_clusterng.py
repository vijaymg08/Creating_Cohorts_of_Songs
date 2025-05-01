import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv('rolling_stones_spotify.csv', parse_dates=['release_date'])
# Data cleaning
def clean_data(df):
    df['duration_min'] = df['duration_ms'] / 60000
    df['release_year'] = df['release_date'].dt.year
    df['is_live'] = df['name'].str.contains('- Live', case=False).astype(int)

    audio_features = [
        'danceability', 'energy', 'loudness', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence',
        'tempo', 'duration_min'
    ]

    return df.dropna(subset=audio_features), audio_features


# Perform EDA with data dictionary
def perform_analysis(df, features):
    plt.figure(figsize=(14, 6))
    df.groupby('release_year')[features].mean().plot()
    plt.title('Audio Feature Evolution Over Time')
    plt.ylabel('Feature Value (Standardized)')
    plt.show()
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='is_live', y='energy', data=df)
    plt.title('Energy Distribution: Live vs Studio Recordings')
    plt.xticks([0, 1], ['Studio', 'Live'])
    plt.show()


# Clustering with audio
def cluster_songs(df, features):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    pca = PCA(n_components=0.9)
    X_pca = pca.fit_transform(X_scaled)
    wcss = []
    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X_pca)
        wcss.append(kmeans.inertia_)

    plt.plot(range(2, 11), wcss, marker='o')
    plt.title('Elbow Method for Optimal Clusters')
    plt.show()
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_pca)

    print(f"Silhouette Score: {silhouette_score(X_pca, df['cluster']):.2f}")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['cluster'], palette='viridis')
    plt.title('Song Clusters in PCA Space')
    plt.show()

    return df

if __name__ == "__main__":
    df_clean, features = clean_data(df)
    perform_analysis(df_clean, features)
    df_clustered = cluster_songs(df_clean, features)

    output_cols = ['name', 'album', 'release_year', 'cluster', 'popularity'] + features
    df_clustered[output_cols].to_csv('rolling_stones_clusters.csv', index=False)
