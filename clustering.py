import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('Global_Music_Streaming_Listener_Preferences.csv')

# ---------------- Data Modification ----------------
df.rename(columns={
    'User ID': 'user_id',
    'Age': 'age',
    'Country': 'country',
    'Streaming Platform': 'streaming_platform',
    'Top Genre': 'top_genre',
    'Minutes Streamed Per Day': 'minutes_streamed_per_day',
    'Most Played Artist': 'most_played_artist',
    'Number of Songs Liked': 'number_of_songs_liked',
    'Subscription Type': 'subscription_type',
    'Listening Time (Morning/Afternoon/Night)': 'listening_time',
    'Discover Weekly Engagement (%)': 'discover_weekly_engagement',
    'Repeat Song Rate (%)': 'repeat_song_rate'
}, inplace=True)

developed_countries = {'Japan', 'Germany', 'Australia', 'South Korea', 'UK', 'Canada', 'USA', 'France'}
developing_countries = {'India', 'Brazil'}

# Add new column
df['country_dev_status'] = df['country'].apply(lambda x: 'developed' if x in developed_countries else 'developing')

# All features
num_all_features = [
    'age', 'minutes_streamed_per_day', 'number_of_songs_liked',
    'discover_weekly_engagement','repeat_song_rate'
]

cat_all_features = [
    'country', 'streaming_platform', 'top_genre', 'most_played_artist',
    'subscription_type', 'listening_time', 'country_dev_status'
]
# ---------------- Select Features ----------------
features = [
    'minutes_streamed_per_day', 
    'number_of_songs_liked',
    'discover_weekly_engagement'
] 
X = df[features].copy()

# ---------------- Encoding and Scaling ----------------
numeric_features = [
    'minutes_streamed_per_day', 
    'number_of_songs_liked',
    'discover_weekly_engagement'
]
categorical_features = [

]
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# ----------------- Pipeline ----------------
clustering_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('kmeans', KMeans(n_clusters=4, random_state=42))
])

# ----------------- Fit the Model ----------------
clustering_pipeline.fit(X)

# Get cluster labels
cluster_labels = clustering_pipeline.named_steps['kmeans'].labels_

# ----------------- Validation ----------------
# Silhouette Score
X_processed = preprocessor.fit_transform(X)
if hasattr(X_processed, "toarray"):
    X_processed = X_processed.toarray()
sil_score = silhouette_score(X_processed, clustering_pipeline.named_steps['kmeans'].labels_)
print(f'Silhouette Score: {sil_score:.3f}')

# Try different number of clusters
scores = []
range_n = range(2, 11)
for k in range_n:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_processed)
    scores.append(silhouette_score(X_processed, labels))

plt.figure(figsize=(10, 6))
plt.plot(range_n, scores, marker='o')
plt.title('Silhouette Score vs Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Assign cluster labels to DataFrame
df['cluster'] = cluster_labels

# ----------------- Interpret Clusters ----------------
# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)

df['pca1'] = X_pca[:, 0]
df['pca2'] = X_pca[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster', palette='Set2')
plt.title('Clusters Visualization via PCA')
plt.show()

# ----------------- Cluster Profiling ----------------
# Handling numeric features
cluster_profiles = df.groupby('cluster')[num_all_features].mean()
print("Cluster Feature Averages:\n", cluster_profiles)

# Handling categorical features
def top_n_modes(series, n=2):
    return series.value_counts().nlargest(n).index.tolist()

# Apply to categorical columns
top_n_summary = df.groupby('cluster')[cat_all_features].agg(lambda x: top_n_modes(x, n=2))

# Prettify as strings for easy viewing
top_n_summary = top_n_summary.applymap(lambda x: ', '.join(x))
print(top_n_summary)

# Add count to understand size
cluster_profiles['count'] = df['cluster'].value_counts().sort_index()
print("\nCluster Sizes:\n", cluster_profiles['count'])

# ---------------- Plotting Features -----------------
cluster_profiles_no_count = cluster_profiles.drop(columns=['count'])
normalized_profiles = cluster_profiles_no_count.copy()
normalized_profiles = (normalized_profiles - normalized_profiles.min()) / (normalized_profiles.max()- normalized_profiles.min())

normalized_profiles.plot(kind='bar', figsize=(12, 6))
plt.title('Normalized Numeric Features per Cluster')
plt.ylabel('Normalized Value')
plt.xlabel('Cluster')
plt.xticks(rotation=0)
plt.legend(loc='best')
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
from pandas.plotting import table

# Split top_n_summary into two parts
half = len(top_n_summary.columns) // 2
first_half = top_n_summary.iloc[:, :half]
second_half = top_n_summary.iloc[:, half:]

# First table (part 1)
plt.figure(figsize=(14, 4))
ax1 = plt.subplot(111, frame_on=False)
ax1.xaxis.set_visible(False)
ax1.yaxis.set_visible(False)
tbl1 = table(ax1, first_half, loc='center', colWidths=[0.18]*first_half.shape[1])
tbl1.auto_set_font_size(False)
tbl1.set_fontsize(9)
tbl1.scale(1.3, 1.3)
plt.title("Top Categorical Values per Cluster (Part 1)")
plt.tight_layout()
plt.show()

# Second table (part 2)
plt.figure(figsize=(14, 4))
ax2 = plt.subplot(111, frame_on=False)
ax2.xaxis.set_visible(False)
ax2.yaxis.set_visible(False)
tbl2 = table(ax2, second_half, loc='center', colWidths=[0.18]*second_half.shape[1])
tbl2.auto_set_font_size(False)
tbl2.set_fontsize(9)
tbl2.scale(1.3, 1.3)
plt.title("Top Categorical Values per Cluster (Part 2)")
plt.tight_layout()
plt.show()

# ----------------- Boxplots for Visual Interpretation ----------------
# for feature in num_all_features:
#     plt.figure(figsize=(8, 5))
#     sns.boxplot(x='cluster', y=feature, data=df)
#     plt.title(f'{feature} by Cluster')
#     plt.show()

# ----------------- Focused Lift Analysis ----------------
# Compute overall proportions
overall_premium_pct = (df['subscription_type'] == 'Premium').mean()
overall_developed_pct = (df['country_dev_status'] == 'developed').mean()

# Compute per-cluster proportions
premium_lift = df.groupby('cluster')['subscription_type'].apply(lambda x: (x == 'Premium').mean()) / overall_premium_pct
developed_lift = df.groupby('cluster')['country_dev_status'].apply(lambda x: (x == 'developed').mean()) / overall_developed_pct

# Combine into a DataFrame for readability
focused_lift_df = pd.DataFrame({
    'Premium Lift': premium_lift,
    'Developed Country Lift': developed_lift
})

print("\nFocused Lift Values (per cluster):")
print(focused_lift_df)

# Export full data with cluster column
df.to_csv('clustered_listener_data.csv', index=False)