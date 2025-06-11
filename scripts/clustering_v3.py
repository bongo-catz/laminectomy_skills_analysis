import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
import logging as log
import click 
from datetime import datetime
from scipy.stats import zscore
from umap import UMAP

# Configure logging and plotting style
log.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=log.INFO)
sns.set(style="whitegrid", palette="muted", font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)

def load_and_preprocess_data(filepath):
    """Load data and perform initial preprocessing"""
    log.info(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    
    # Convert time to datetime if it's not already
    if 'Stroke End Time' in df.columns:
        df = df.rename(columns={'Stroke End Time': 'Time_Seconds'})
        
    df['Time_Quartile'] = pd.qcut(df['Time_Seconds'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    
    # Basic data validation
    if df.isnull().sum().sum() > 0:
        log.warning("Data contains missing values. Imputing with median.")
        df = df.fillna(df.median())
    
    return df

def merge_participant_characteristics(df, json_path):
    """Merge stroke data with participant characteristics from JSON"""
    import json

    # Load participant characteristics JSON
    with open(json_path, 'r') as f:
        participant_data = json.load(f)
    participant_df = pd.DataFrame(participant_data)

    # Extract participant ID from 'Source' column
    df['participant'] = df['Source'].apply(lambda x: x.split('_')[0])
    
    # Merge stroke data with participant metadata
    df = df.merge(participant_df[['participant', 'yrs_prev_exp']], on='participant', how='left')
    
    missing = df['yrs_prev_exp'].isnull().sum()
    if missing > 0:
        log.warning(f"{missing} strokes have no matching 'yrs_prev_exp' in participant data.")
    
    return df

def feature_engineering(df, feature_cols):
    """Create additional meaningful features"""
    
    # Create composite features
    # if all(col in df.columns for col in ['Velocity', 'Acceleration']):
    #    df['Velocity_Acceleration_Ratio'] = df['Velocity'] / (df['Acceleration'] + 1e-6)
    
    # Efficiency
    #if all(col in df.columns for col in ['Length', 'Voxels Removed']):
    #    df['Efficiency'] = df['Voxels Removed'] / (df['Length'] + 1e-6)
    
    # Time-based features
    if 'Time_Seconds' in df.columns and 'Source' in df.columns:
        df = df.sort_values(by=['Source', 'Time_Seconds']).reset_index(drop=True)
        # Compute Stroke Duration: difference in Time_Seconds within each Source
        df['Stroke_Duration'] = df.groupby('Source')['Time_Seconds'].diff().fillna(df['Time_Seconds'])
                                
    # Update feature columns with new features
    new_features = [f for f in ['Velocity_Acceleration_Ratio', 'Efficiency', 'Stroke_Duration', 'Jerk_Magnitude'] if f in df.columns]
    return df, feature_cols + new_features

def remove_outliers(df, feature_cols):
    """Remove outliers using IQR method"""
    df_clean = df.copy()
    for col in feature_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        df_clean = df_clean[~((df_clean[col] < (Q1 - 1.5 * IQR)) | 
                             (df_clean[col] > (Q3 + 1.5 * IQR)))]
    log.info(f"Removed {len(df)-len(df_clean)} outliers")
    return df_clean

def normalize_features(df, feature_cols, method='standard'):
    """Normalize features using different methods"""
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'power':
        scaler = PowerTransformer(method='yeo-johnson')
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    scaled_data = scaler.fit_transform(df[feature_cols])
    return pd.DataFrame(scaled_data, columns=feature_cols), scaler

def determine_optimal_clusters(data, max_k=15):
    """Determine optimal number of clusters using multiple methods"""
    log.info("Determining optimal number of clusters...")
    
    # Elbow method
    visualizer = KElbowVisualizer(KMeans(random_state=42, n_init='auto'), k=(2, max_k), metric='distortion')
    visualizer.fit(data)
    elbow_k = visualizer.elbow_value_
    visualizer.show(outpath="elbow_method.png")
    plt.close()
    
    # Silhouette score
    visualizer = KElbowVisualizer(KMeans(random_state=42, n_init='auto'), k=(2, max_k), metric='silhouette')
    visualizer.fit(data)
    silhouette_k = visualizer.elbow_value_
    visualizer.show(outpath="silhouette_method.png")
    plt.close()
    
    log.info(f"Suggested clusters - Elbow: {elbow_k}, Silhouette: {silhouette_k}")
    suggested_k = min(elbow_k, silhouette_k)
    return suggested_k

def perform_clustering(data, n_clusters, method='kmeans'):
    """Perform clustering using different algorithms"""
    log.info(f"Performing {method} clustering with {n_clusters} clusters")
    
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    elif method == 'gmm':
        model = GaussianMixture(n_components=n_clusters, random_state=42)
    elif method == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif method == 'dbscan':
        model = DBSCAN(eps=0.5, min_samples=5)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    if method == 'gmm':
        labels = model.fit_predict(data)
    else:
        model.fit(data)
        labels = model.labels_
    
    # Calculate metrics
    if len(np.unique(labels)) > 1:  # Some metrics require at least 2 clusters
        silhouette = silhouette_score(data, labels)
        db_score = davies_bouldin_score(data, labels)
        ch_score = calinski_harabasz_score(data, labels)
        log.info(f"Clustering metrics - Silhouette: {silhouette:.3f}, Davies-Bouldin: {db_score:.3f}, Calinski-Harabasz: {ch_score:.3f}")
    else:
        log.warning("Only one cluster found - cannot compute metrics")
    
    return model, labels

def visualize_clusters(data, labels, feature_cols, output_dir):
    """Create various visualizations of the clusters"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Use a discrete color palette
    unique_labels = sorted(np.unique(labels))
    num_clusters = len(unique_labels)
    palette = sns.color_palette("Set2", n_colors=num_clusters)
    color_map = {label: palette[i] for i, label in enumerate(unique_labels)}
    colors = [color_map[label] for label in labels]
    
    # PCA for dimensionality reduction
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(data)
    
    # t-SNE for visualization (more expensive but often better for visualization)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(data)
    
    # UMAP for visualization
    umap = UMAP(n_components=2, random_state=42)
    umap_result = umap.fit_transform(data)

    # PCA Plot
    plt.figure(figsize=(10, 8))
    for label in unique_labels:
        idx = np.array(labels) == label
        plt.scatter(pca_result[idx, 0], pca_result[idx, 1], 
                    c=[color_map[label]], label=f'Phenotype {label}', alpha=0.6)
    plt.title('PCA Cluster Visualization')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title="Phenotype")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_cluster_visualization.png'))
    plt.close()
    
    # t-SNE Plot
    plt.figure(figsize=(10, 8))
    for label in unique_labels:
        idx = np.array(labels) == label
        plt.scatter(tsne_result[idx, 0], tsne_result[idx, 1], 
                    c=[color_map[label]], label=f'Phenotype {label}', alpha=0.6)
    plt.title('t-SNE Cluster Visualization')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title="Phenotype")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tsne_cluster_visualization.png'))
    plt.close()

    # UMAP Plot
    plt.figure(figsize=(10, 8))
    for label in unique_labels:
        idx = np.array(labels) == label
        plt.scatter(umap_result[idx, 0], umap_result[idx, 1], 
                    c=[color_map[label]], label=f'Phenotype {label}', alpha=0.6)
    plt.title('UMAP Cluster Visualization')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(title="Phenotype")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'umap_cluster_visualization.png'))
    plt.close()
    
    '''
    # Pairplot of features with clusters
    plot_data = pd.DataFrame(data, columns=feature_cols)
    plot_data['Cluster'] = labels
    
    # Sample if too many points for pairplot
    if len(plot_data) > 1000:
        plot_data = plot_data.sample(1000, random_state=42)
    
    sns.pairplot(plot_data, hue='Cluster', palette='viridis', corner=True)
    plt.savefig(os.path.join(output_dir, 'feature_pairplot.png'))
    '''
    plt.close()

def plot_cluster_bars(df, labels, feature_cols, output_dir):
    """Create bar plots of feature means per cluster with clean layout and discrete colors."""
    df = df.copy()
    df['Phenotype'] = labels
    cluster_means = df.groupby('Phenotype')[feature_cols].mean().reset_index()

    # Set up aesthetics
    num_features = len(feature_cols)
    num_cols = 2
    num_rows = (num_features + 1) // num_cols
    palette = sns.color_palette("Set2", n_colors=len(cluster_means['Phenotype'].unique()))
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, num_rows * 4))
    axes = axes.flatten()

    for i, feature in enumerate(feature_cols):
        sns.barplot(
            data=cluster_means,
            x='Phenotype', y=feature,
            palette=palette,
            ax=axes[i]
        )
        axes[i].set_title(f'{feature} by Stroke Phenotype', fontsize=12)
        axes[i].set_xlabel('Phenotype')
        axes[i].set_ylabel(feature)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Add a global legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Phenotype {i}',
                          markerfacecolor=palette[i], markersize=10)
               for i in range(len(palette))]
    fig.legend(handles=handles, loc='upper center', ncol=len(palette), title="Phenotypes")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'cluster_feature_bars.png'))
    plt.close()
    
def plot_quartile_distribution(df, cluster_col, quartile_col, output_dir, filename='quartile_cluster_distribution.png'):
    """Plot distribution of clusters across time quartiles with discrete colors and cleaner formatting."""
    # Prepare data
    grouped = df.groupby([quartile_col, cluster_col]).size().unstack(fill_value=0)
    percentages = grouped.div(grouped.sum(axis=1), axis=0) * 100

    # Plot
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("Set2", n_colors=percentages.columns.nunique())
    percentages.plot(kind='bar', stacked=True, color=palette, edgecolor='black')

    # Formatting
    plt.title('Stroke Phenotype Distribution across Time Quartiles')
    plt.xlabel('Time Quartile')
    plt.ylabel('Percentage of Strokes (%)')
    plt.legend(title='Phenotype', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()

def analyze_cluster_profiles(df, labels, feature_cols, output_dir):
    """Analyze and visualize cluster characteristics"""
    df['Cluster'] = labels
    cluster_stats = df.groupby('Cluster')[feature_cols].agg(['mean', 'median', 'std'])
    
    # Flatten multi-level column headers
    cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]
    
    # Save cluster statistics
    cluster_stats.to_csv(os.path.join(output_dir, 'cluster_statistics.csv'))
    
    # Plot cluster profiles
    plt.figure(figsize=(15, 8))
    sns.heatmap(cluster_stats.filter(like='_mean').T, annot=True, cmap='viridis', fmt='.2f')
    plt.title('Cluster Feature Profiles (Mean Values)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_profiles.png'))
    plt.close()
    
    # Plot time distribution if available
    if 'Time_Seconds' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Cluster', y='Time_Seconds', data=df)
        plt.title('Time Distribution by Cluster')
        plt.ylabel('Time (seconds)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'time_distribution.png'))
        plt.close()
    
    return cluster_stats

def plot_stroke_distribution_by_experience(df, output_dir):
    """Plot stacked percentage of strokes by cluster for each year of prior experience"""
    df_exp = df.dropna(subset=['yrs_prev_exp', 'Cluster']).copy()

    # Group and normalize to percentage
    grouped = df_exp.groupby(['yrs_prev_exp', 'Cluster']).size().unstack(fill_value=0)
    percentages = grouped.div(grouped.sum(axis=1), axis=0) * 100

    # Sort index to make sure x-axis is ordered
    percentages = percentages.sort_index()

    # Plot
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("Set2", n_colors=percentages.shape[1])
    percentages.plot(kind='bar', stacked=True, color=palette, edgecolor='black')

    plt.title('Stroke Phenotype Composition by Years of Prior Experience')
    plt.xlabel('Years of Prior Experience')
    plt.ylabel('Percentage of Strokes (%)')
    plt.legend(title='Phenotype', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.ylim(0, 100)
    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'stroke_distribution_by_experience.png'))
    plt.close()

def save_cluster_results(df, output_dir):
    """Save the clustered data and summary statistics"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'clustered_results_{timestamp}.csv')
    df.to_csv(output_file, index=False)
    
    log.info(f"Saved clustered results to: {output_file}")

@click.command()
@click.option('--filepath', required=True, type=click.Path(exists=True),
              help='Path to the CSV file containing stroke metrics.')
@click.option('--participant-json', required=True, type=click.Path(exists=True),
              help='Path to JSON file containing participant characteristics.')
@click.option('--output-dir', required=True, type=click.Path(),
              help='Directory to save clustering results.')
@click.option('--num-clusters', default=None, type=int,
              help='Number of clusters to use. If None, will determine automatically.')
@click.option('--clustering-method', default='kmeans',
              type=click.Choice(['kmeans', 'gmm', 'agglomerative', 'dbscan']),
              help='Clustering algorithm to use.')
@click.option('--feature-cols', default='Length,Velocity,Acceleration,Jerk,Voxels Removed',
              help='Comma-separated list of features to use for clustering.')
@click.option('--outliers', is_flag=True, default=False,
              help='Whether to remove outliers from the feature columns before clustering.')
def main(filepath, participant_json, output_dir, num_clusters, clustering_method, feature_cols, outliers):
    """Main function to perform stroke data clustering analysis"""
    # Parse feature columns
    feature_cols = [col.strip() for col in feature_cols.split(',')]
    
    # Load and preprocess data
    df = load_and_preprocess_data(filepath)
    df = merge_participant_characteristics(df, participant_json)

    df, feature_cols = feature_engineering(df, feature_cols)
    
    # Optionally remove outliers
    if outliers:
        log.info("Removing outliers from feature columns")
        df = remove_outliers(df, feature_cols) 
    
    # Normalize features
    scaled_data, scaler = normalize_features(df, feature_cols, method='power')
    
    # PCA (dimenstionality reduction) before clustering
    pca = PCA(n_components=0.95, random_state=42)  # Keep 95% variance
    reduced_data = pca.fit_transform(scaled_data)
    log.info(f"Reduced dimensions from {scaled_data.shape[1]} to {reduced_data.shape[1]}")

    # Determine optimal clusters if not specified
    if num_clusters is None:
        suggested_k = determine_optimal_clusters(scaled_data)
        num_clusters = suggested_k
        log.info(f"Using automatically determined number of clusters: {num_clusters}")
    
    # Perform clustering
    model, labels = perform_clustering(scaled_data, num_clusters, method=clustering_method)
    df['Cluster'] = labels
        
    # Visualize and analyze results
    visualize_clusters(scaled_data, labels, feature_cols, output_dir)
    plot_cluster_bars(df, labels, feature_cols, output_dir)
    plot_quartile_distribution(df, cluster_col='Cluster', quartile_col='Time_Quartile', output_dir=output_dir)
    cluster_stats = analyze_cluster_profiles(df, labels, feature_cols, output_dir)
    
    # Visualize stroke distribution by years of experience
    plot_stroke_distribution_by_experience(df, output_dir)
    
    # Add clusters to original data and save
    save_cluster_results(df, output_dir)
    
    log.info("Clustering analysis completed successfully!")

if __name__ == "__main__":
    main()