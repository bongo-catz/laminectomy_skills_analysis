import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import logging as log
import click 

# example usage: python3 scripts/clustering_v2.py 
#                  --filepath data/Laminectomy_Data/merged_extracted_metrics/combined_extracted_stroke_metrics.csv
#                  --output-dir data/Laminectomy_Data/clustering_v2
#                  --num-clusters 5

def load_data(filepath):
    print(f"Loading data from: {filepath}")
    return pd.read_csv(filepath)

def standardize_features(df, feature_cols):
    scaler = StandardScaler()
    return scaler.fit_transform(df[feature_cols])

def perform_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(data)
    return kmeans, labels

def assign_time_quartiles(df, time_col='Stroke End Time'):
    df['Time_Quartile'] = pd.qcut(df[time_col], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    return df

def plot_variable_vs_time(df, variables, cluster_col, save_dir, cmap='viridis'):
    os.makedirs(save_dir, exist_ok=True)
    for var in variables:
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df['Stroke End Time'], df[var], c=df[cluster_col], cmap=cmap, s=15, alpha=0.7)
        cbar = fig.colorbar(scatter, ticks=np.arange(df[cluster_col].nunique()))
        cbar.set_label('Stroke Phenotype')
        ax.set_title(f'{var} over Time')
        ax.set_xlabel('Time')
        ax.set_ylabel(var)
        plt.savefig(os.path.join(save_dir, f'{var}_vs_Time_clusters.jpg'), bbox_inches='tight')
        plt.close()

def plot_quartile_distribution(df, cluster_col, quartile_col, save_path):
    grouped = df.groupby([quartile_col, cluster_col]).size().unstack(fill_value=0)
    percentages = grouped.div(grouped.sum(axis=1), axis=0) * 100
    percentages.plot(kind='bar', stacked=True, cmap='viridis', figsize=(10, 6))
    plt.title('Stroke Phenotype Distribution across Time Quartiles')
    plt.xlabel('Time Quartile')
    plt.ylabel('Percentage of Strokes (%)')
    plt.legend(title='Phenotype', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_metric_bars(cluster_means, save_dir, cmap='viridis'):
    os.makedirs(save_dir, exist_ok=True)
    colors = cm.get_cmap(cmap, len(cluster_means))
    for metric in cluster_means.columns:
        cluster_means[metric].plot(kind='bar', color=[colors(i) for i in range(len(cluster_means))])
        plt.title(f'{metric} by Stroke Phenotype')
        plt.xlabel('Phenotype')
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{metric}_by_phenotype.jpg'))
        plt.close()

def plot_metric_grid(cluster_means, save_path, cmap='viridis'):
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    axs = axs.flatten()
    fig.delaxes(axs[-1])  # Remove extra subplot if metrics are <6

    colors = cm.get_cmap(cmap, len(cluster_means))

    for i, metric in enumerate(cluster_means.columns):
        cluster_means[metric].plot(kind='bar', ax=axs[i], color=[colors(j) for j in range(len(cluster_means))])
        axs[i].set_title(f'{metric} by Stroke Phenotype')
        axs[i].set_xlabel('Phenotype')
        axs[i].set_ylabel(metric)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors(i)) for i in range(len(cluster_means))]
    labels = [f'Phenotype {i}' for i in range(len(cluster_means))]
    fig.legend(handles, labels, title="Phenotypes", loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=len(labels))
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path)
    plt.close()

def plot_elbow_curve(scaled_data, save_path, max_k=10):
    inertia = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(scaled_data)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), inertia, 'bo-')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


@click.command()
@click.option('--filepath', required=True, type=click.Path(exists=True),
              help='Path to the CSV file containing experiment directories.')
@click.option('--output-dir', required=True, type=click.Path(),
              help='Path to save the combined CSV output.')
@click.option('--num-clusters', default=5, show_default=True,
              help='Number of clusters to classify strokes into.')
def main(filepath, output_dir, num_clusters):
    # Configs
    log.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=log.INFO)
    
    log.info(f"Reading data from: {filepath}")
    log.info(f"Saving cluster results to: {output_dir}")
    
    n_clusters = num_clusters
    feature_cols = ['Length', 'Velocity', 'Acceleration', 'Jerk', 'Voxels Removed']

    # Load and preprocess
    df = load_data(filepath)
    df = assign_time_quartiles(df)
    scaled = standardize_features(df, feature_cols)

    # Clustering
    log.info(f"Performing clusters with {n_clusters} clusters.")
    kmeans, labels = perform_kmeans(scaled, n_clusters)
    df['Cluster'] = labels

    # Cluster averages
    cluster_means = df.groupby('Cluster')[feature_cols].mean()

    # Plots
    plot_variable_vs_time(df, feature_cols, 'Cluster', output_dir)
    plot_quartile_distribution(df, 'Cluster', 'Time_Quartile', os.path.join(output_dir, 'quartile_cluster_distribution.jpg'))
    plot_metric_bars(cluster_means, output_dir)
    plot_metric_grid(cluster_means, os.path.join(output_dir, 'cluster_metrics_summary.jpg'))
    log.info("Plotting elbow curve.")
    plot_elbow_curve(scaled, os.path.join(output_dir, 'elbow_method.jpg'))

    log.info("All visualizations and clustering completed.")


if __name__ == "__main__":
    main()