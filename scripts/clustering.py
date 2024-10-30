import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np


# Load the data
file_path = '/Users/orenw/Documents/tb_skills_analysis/data/SDF_UserStudy_Data/extracted_metrics/combined_extracted_stroke_metrics.csv'
data = pd.read_csv(file_path)

# Select the variables for clustering (excluding 'Time')
clustering_data = data[['Stroke Length', 'Stroke Velocity', 'Stroke Acceleration',
                        'Stroke Jerk', 'Voxels Removed per Stroke']]

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(clustering_data)

# Set the number of clusters
optimal_clusters = 5  # Modify this based on elbow method

# Fit k-means
kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)
kmeans.fit(scaled_data)

# Get the cluster labels for each data point
data['Cluster'] = kmeans.labels_

# Divide the data into quartiles based on 'Time'
data['Time_Quartile'] = pd.qcut(data['Time'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# Calculate the average values of each metric for each cluster
cluster_means = data.groupby('Cluster').mean()[clustering_data.columns]

# Define a consistent color map for the clusters
colors = cm.get_cmap('viridis', optimal_clusters)  # Use the 'viridis' colormap with 6 distinct colors

# Function to plot bar graph for stroke phenotype assignments across quartiles
def plot_stroke_phenotype_distribution_by_quartile(data, save_path=None):
    # Group the data by 'Time_Quartile' and 'Stroke Phenotype' and count the occurrences
    quartile_phenotype_counts = data.groupby(['Time_Quartile', 'Cluster']).size().unstack(fill_value=0)
    
    # Convert counts to percentages
    quartile_phenotype_percentages = quartile_phenotype_counts.div(quartile_phenotype_counts.sum(axis=1), axis=0) * 100
    
    # Plot the bar graph for each quartile with percentage on y-axis
    quartile_phenotype_percentages.plot(kind='bar', stacked=True, figsize=(10, 6), cmap='viridis')
    
    # Add labels and title
    plt.title('Stroke Phenotype Distribution across Time Quartiles')
    plt.xlabel('Time Quartile')
    plt.ylabel('Percentage of Strokes (%)')
    plt.legend(title='Stroke Phenotype', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved bar graph at: {save_path}")
    
    # Show the plot
    plt.tight_layout()
    plt.show()

# Function to plot each variable over time, colored by stroke phenotype assignment, and save the figure
def plot_variable_over_time(data, variable_name, stroke_phenotype_labels, n_clusters, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a scatter plot of the variable vs Time, with color by stroke phenotype
    scatter = ax.scatter(data['Time'], data[variable_name], c=stroke_phenotype_labels, cmap='viridis', s=15, alpha=0.7)

    # Add labels and title
    ax.set_title(f'{variable_name} over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel(variable_name)
    
    # Create a colorbar to represent stroke phenotypes
    cbar = fig.colorbar(scatter, ticks=np.arange(n_clusters))
    cbar.set_label('Stroke Phenotype')

    # Add a legend showing the stroke phenotype colors
    legend1 = ax.legend(*scatter.legend_elements(), title="Stroke Phenotypes")
    ax.add_artist(legend1)

    # Save the figure
    fig.savefig(save_path, bbox_inches='tight')
    print(f"Saved figure for {variable_name} at: {save_path}")
    
    # Close the plot to free memory if you generate many plots
    plt.close(fig)

# Function to plot individual bar charts for each metric with consistent stroke phenotype colors
def plot_metric_averages_by_stroke_phenotype(stroke_phenotype_means, save_path_prefix=None):
    # Iterate through each metric and create a bar plot
    for metric in stroke_phenotype_means.columns:
        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot the average value of the metric across stroke phenotypes with color based on the stroke phenotype
        stroke_phenotype_means[metric].plot(kind='bar', ax=ax, width=0.8, color=[colors(i) for i in range(optimal_clusters)])

        # Add labels and title
        plt.title(f'Average {metric} by Stroke Phenotype')
        plt.xlabel('Stroke Phenotype')
        plt.ylabel(f'Average {metric}')
        
        # Add legend with stroke phenotype colors
        handles = [plt.Rectangle((0,0),1,1, color=colors(i)) for i in range(optimal_clusters)]
        labels = [f'Stroke Phenotype {i}' for i in range(optimal_clusters)]
        ax.legend(handles, labels, title="Stroke Phenotype", bbox_to_anchor=(1.05, 1), loc='upper left')

        # Save the plot if a save path is provided
        if save_path_prefix:
            save_path = f"{save_path_prefix}/stroke_phenotype_{metric}_averages.jpg"
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved plot for {metric} at: {save_path}")
        
        # Show the plot
        plt.tight_layout()
        plt.close()

# Function to plot all metrics as subplots in one figure with a shared legend
def plot_metrics_in_one_figure(stroke_phenotype_means, save_path_prefix=None):
    # Create a figure with 5 subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))

    # Remove the last empty subplot (bottom-right)
    fig.delaxes(axs[2, 1])

    # Define color mapping for stroke phenotypes
    optimal_clusters = len(stroke_phenotype_means.index)
    colors = cm.get_cmap('viridis', optimal_clusters)

    # Iterate through the metrics and plot each in its corresponding subplot
    for i, metric in enumerate(stroke_phenotype_means.columns):
        ax = axs[i // 2, i % 2]  # Determine the position in the grid
        stroke_phenotype_means[metric].plot(kind='bar', ax=ax, width=0.8, color=[colors(j) for j in range(optimal_clusters)])

        # Add labels and title
        ax.set_title(f'Average {metric} by Stroke Phenotype')
        ax.set_xlabel('Stroke Phenotype')
        ax.set_ylabel(f'Average {metric}')

    # Create shared legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors(i)) for i in range(optimal_clusters)]
    labels = [f'Stroke Phenotype {i}' for i in range(optimal_clusters)]
    fig.legend(handles, labels, title="Legend", bbox_to_anchor=(0.5, 0.97), loc='upper center', ncol=optimal_clusters)

    # Adjust layout to make sure subplots and the legend fit well
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # Save the plot if a save path is provided
    if save_path_prefix:
        save_path = f"{save_path_prefix}/cluster_all_metrics_averages.jpg"
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved combined plot at: {save_path}")

    # Show the plot
    plt.show()

# Variables to plot over time
variables_to_plot = ['Stroke Length', 'Stroke Velocity', 'Stroke Acceleration', 
                     'Stroke Jerk', 'Voxels Removed per Stroke']

# Directory where the figures will be saved
base_dir = '/Users/orenw/Documents/tb_skills_analysis/data/SDF_UserStudy_Data/extracted_metrics'

# Loop through each variable and create a separate graph with a legend for clusters, then save each plot
for variable in variables_to_plot:
    save_path = f"{base_dir}/kmeans_multi_{optimal_clusters}n_{variable}_vs._Time.jpg"
    plot_variable_over_time(data, variable, data['Cluster'], optimal_clusters, save_path)

# Set the save path for the bar graph
cluster_distribution_by_quartiles_save_path = f'{base_dir}/cluster_distribution_by_quartile.jpg'

# Plot the bar graph and save it
plot_stroke_phenotype_distribution_by_quartile(data, cluster_distribution_by_quartiles_save_path)

# Plot and save the metric averages by cluster
plot_metric_averages_by_stroke_phenotype(cluster_means, base_dir)

# Plot and save the metric averages as subplots
plot_metrics_in_one_figure(cluster_means, base_dir)




"""
# DIVIDING DATA INTO QUARTILES THEN RUNNING KMEANS

# Divide the data into quartiles based on the 'Time' variable
data['Time_Quartile'] = pd.qcut(data['Time'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# List to hold quartiles for processing
quartiles = ['Q1', 'Q2', 'Q3', 'Q4']

# Function to calculate and plot the elbow method
def calculate_and_plot_elbow(scaled_data, ax, quartile_label):
    inertia = []
    k_range = range(1, 11)  # Testing for k between 1 and 10
    
    # Calculate inertia for each k value
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)
    
    # Plot the elbow curve
    ax.plot(k_range, inertia, 'bo-', markersize=8)
    ax.set_title(f'Elbow Method for {quartile_label}')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia')
    ax.grid(True)

# Set up the figure for elbow plots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))  # 2x2 grid for 4 elbow plots
axes = axes.ravel()  # Flatten the 2x2 grid for easier indexing

# Loop through each quartile and plot elbow curves
for i, quartile in enumerate(quartiles):
    # Filter data for the current quartile
    quartile_data = data[data['Time_Quartile'] == quartile]
    
    # Select the variables for clustering
    clustering_data = quartile_data[['Stroke Length', 'Stroke Velocity', 'Stroke Acceleration',
                                     'Stroke Jerk', 'Stroke Curvature', 'Voxels Removed per Stroke']]
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)
    
    # Plot the elbow method for the current quartile
    calculate_and_plot_elbow(scaled_data, axes[i], quartile)

# Adjust layout and show elbow plots
plt.tight_layout()
plt.show()

# Now prompt for k-values for each quartile
k_values = {}  # Dictionary to store k values for each quartile
for quartile in quartiles:
    k_values[quartile] = int(input(f"Enter the number of clusters (k) for {quartile}: "))

# Set up the figure and subplots for clustering results
fig, axes = plt.subplots(2, 2, figsize=(14, 12))  # 2x2 grid for 4 clustering plots
axes = axes.ravel()

# Function to perform k-means clustering and plot the 2D results
def process_and_plot_quartile(quartile_label, ax, k_value):
    # Filter data for the current quartile
    quartile_data = data[data['Time_Quartile'] == quartile_label]
    
    # Select the variables for clustering
    clustering_data = quartile_data[['Stroke Length', 'Stroke Velocity', 'Stroke Acceleration',
                                     'Stroke Jerk', 'Stroke Curvature', 'Voxels Removed per Stroke']]
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)
    
    # Fit k-means with the chosen k value
    kmeans = KMeans(n_clusters=k_value, random_state=0)
    kmeans.fit(scaled_data)
    
    # Get cluster labels and centroids
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    # Perform PCA to reduce the dimensions to 2 for 2D visualization
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(scaled_data)
    
    # Reduce the centroids using the same PCA transformation
    reduced_centroids = pca.transform(centroids)
    
    # Plot the clusters in the provided subplot
    scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    
    # Plot the centroids as black circles
    ax.scatter(reduced_centroids[:, 0], reduced_centroids[:, 1], c='black', s=200, marker='o', label='Centroids')

    # Add titles and labels
    ax.set_title(f'{quartile_label}: K-Means Clusters (k={k_value})')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')

# Process and plot clusters for each quartile based on the chosen k-values (2D plots)
for i, quartile in enumerate(quartiles):
    process_and_plot_quartile(quartile, axes[i], k_values[quartile])

# Add a common legend for the centroids
plt.tight_layout()
plt.savefig(f'/Users/orenw/Documents/tb_skills_analysis/data/SDF_UserStudy_Data/extracted_metrics/kmeans_all_quartiles_PCA_2D.jpg')
plt.show()

# --- 3D Plotting Section ---

# Function to plot clusters in 3D using PCA with 3 components
def plot_3D_quartile_clusters(quartile_label, k_value):
    # Filter data for the current quartile
    quartile_data = data[data['Time_Quartile'] == quartile_label]
    
    # Select the variables for clustering
    clustering_data = quartile_data[['Stroke Length', 'Stroke Velocity', 'Stroke Acceleration',
                                     'Stroke Jerk', 'Stroke Curvature', 'Voxels Removed per Stroke']]
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)
    
    # Fit k-means with the chosen k value
    kmeans = KMeans(n_clusters=k_value, random_state=0)
    kmeans.fit(scaled_data)
    
    # Get cluster labels and centroids
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    # Perform PCA to reduce the dimensions to 3 for 3D visualization
    pca_3d = PCA(n_components=3)
    reduced_data_3d = pca_3d.fit_transform(scaled_data)
    reduced_centroids_3d = pca_3d.transform(centroids)
    
    # Plot the clusters in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(reduced_data_3d[:, 0], reduced_data_3d[:, 1], reduced_data_3d[:, 2], 
                    c=labels, cmap='viridis', s=50, alpha=0.7)
    
    # Plot the centroids as black circles
    ax.scatter(reduced_centroids_3d[:, 0], reduced_centroids_3d[:, 1], reduced_centroids_3d[:, 2], 
               c='black', s=200, marker='o', label='Centroids')

    # Add labels and title
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title(f'3D PCA Clusters for {quartile_label} (k={k_value})')
    
    # Show the plot
    plt.show()

# Loop through each quartile and plot 3D clusters
for quartile in quartiles:
    plot_3D_quartile_clusters(quartile, k_values[quartile])
"""



"""
# Elbow method to find optimal k
inertia = []
k_range = range(1, 11)  # Adjust the range if needed

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plotting the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, 'bo-', markersize=8)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia (Sum of Squared Distances)')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.savefig(f'/Users/orenw/Documents/tb_skills_analysis/data/SDF_UserStudy_Data/extracted_metrics/kmeans_elbow.jpg')
plt.show()
"""




"""
# FOR PLOTTING ONE VARIABLE VS. TIME 2D GRAPHS WITH CLUSTERS

# Create a new folder named after the x_var and y_var
folder_name = f'/Users/orenw/Documents/tb_skills_analysis/data/SDF_UserStudy_Data/extracted_metrics/Figures/{y_var.replace(" ", "_")} vs. {x_var.replace(" ", "_")}'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Create a subplot grid for the combined cluster plot
fig_clusters, axs_clusters = plt.subplots(2, 3, figsize=(18, 12))  # 2 rows, 3 columns of subplots for clusters
axs_clusters = axs_clusters.flatten()  # Flatten for easy indexing

# Create a subplot grid for the combined centroid plot
fig_centroids, axs_centroids = plt.subplots(2, 3, figsize=(18, 12))  # 2 rows, 3 columns of subplots for centroids
axs_centroids = axs_centroids.flatten()  # Flatten for easy indexing

# Create a subplot grid for the combined superimposed plot
fig_superimposed, axs_superimposed = plt.subplots(2, 3, figsize=(18, 12))  # 2 rows, 3 columns of subplots for superimposed plots
axs_superimposed = axs_superimposed.flatten()  # Flatten for easy indexing

# Loop through different values of n_clusters
for i, n_clusters in enumerate(range(2, 8)):  # from 2 to 7 clusters
    # Apply K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    
    # Add the cluster labels to the dataset
    data['Cluster'] = kmeans.labels_

    # Extract the cluster centers (centroids)
    centroids = kmeans.cluster_centers_
    
    # Calculate the silhouette score
    silhouette_avg = silhouette_score(X, kmeans.labels_)
    print(f'n_clusters = {n_clusters}, Silhouette Score: {silhouette_avg}')
    
    # Plot the clusters in each subplot
    axs_clusters[i].scatter(data[x_var], data[y_var], c=data['Cluster'], cmap='viridis', marker='o', label='Data Points')
    axs_clusters[i].set_xlabel(x_var)
    axs_clusters[i].set_ylabel(y_var)
    axs_clusters[i].set_title(f'n_clusters = {n_clusters}\nSilhouette Score: {silhouette_avg:.3f}')
    
    # Save each individual plot as well
    plt.savefig(f'{folder_name}/kmeans_clustering_{n_clusters}_clusters_{y_var}_vs_{x_var}.jpg')

    # Plot the clusters individually
    plt.figure(figsize=(8,6))
    plt.scatter(data[x_var], data[y_var], c=data['Cluster'], cmap='viridis', marker='o')
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.title(f'K-Means Clustering (n_clusters = {n_clusters})\nSilhouette Score: {silhouette_avg:.3f}')
    
    # Save the figure as a jpg file
    plt.savefig(f"{folder_name}/kmeans_{n_clusters}n_{y_var}_vs_{x_var}_clusters.jpg")
    plt.close()

    # Plot the centroids in a separate subplot
    axs_centroids[i].scatter(centroids[:, 0], centroids[:, 1], c='black', s=100, marker='o', label='Centroids')
    axs_centroids[i].set_xlabel(x_var)
    axs_centroids[i].set_ylabel(y_var)
    axs_centroids[i].set_title(f'Centroids for n_clusters = {n_clusters}')
    
    # Save each individual centroid plot as a jpg file
    plt.figure(figsize=(8,6))
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=100, marker='o', label='Centroids')
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.title(f'Centroids (n_clusters = {n_clusters})')

    # Save the figure as a jpg file
    plt.savefig(f'{folder_name}/kmeans_{n_clusters}n_{y_var}_vs_{x_var}_centroids.jpg')
    plt.close()

    # Plot clusters with centroids superimposed (dark black circle) in each subplot
    axs_superimposed[i].scatter(data[x_var], data[y_var], c=data['Cluster'], cmap='viridis', marker='o', label='Data Points')
    axs_superimposed[i].scatter(centroids[:, 0], centroids[:, 1], c='black', s=100, marker='o', label='Centroids')  # Superimpose centroids
    axs_superimposed[i].set_xlabel(x_var)
    axs_superimposed[i].set_ylabel(y_var)
    axs_superimposed[i].set_title(f'n_clusters = {n_clusters}\nSilhouette Score: {silhouette_avg:.3f}')
    axs_superimposed[i].legend(loc='upper right')

# Adjust the layout and save clusters figure
plt.tight_layout()
plt.savefig(f"{folder_name}/kmeans_all_{y_var}_vs_{x_var}_cc.jpg")

# Adjust the layout and save clusters figure
plt.tight_layout()
fig_clusters.savefig(f"{folder_name}/kmeans_all_{y_var}_vs_{x_var}_clusters.jpg")

# Adjust the layout and save the combined centroid plots
plt.tight_layout()
fig_centroids.savefig(f"{folder_name}/kmeans_all_{y_var}_vs_{x_var}_centroids.jpg")

plt.show()

"""