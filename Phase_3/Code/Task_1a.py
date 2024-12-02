# Import the necessary libraries
import numpy as np
import json

from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import MDS

from Util.services import ServiceClass as Service

def get_latent_space(latent_space):
    data_map = {}
    
    # Get the inherent dimensionality for each label from Task 0a
    with open(f'../Database/inherent_dim_map_{Service.feature_space_map[latent_space]}.json', 'r') as file:
        inherent_dim_map = json.load(file)

    for label in Service.target_labels:

        s = inherent_dim_map[label]

        label_query = f"SELECT {Service.feature_space_map[latent_space]} FROM data WHERE videoID % 2 == 0 AND Action_Label='{label}';"

        data = Service.get_data_from_db(label_query, latent_space, s)

        data_map[label] = data
    
    return data_map

def get_default_threshold(latent_space):
    if latent_space == 1:
        return 1
    elif latent_space == 2:
        return 10
    elif latent_space == 3:
        return 50
    else:
        return 1

def get_threshold(threshold, ratio = None):    
    if ratio == None:
        return threshold
    elif ratio > 1:
        threshold = threshold * 1.5
    elif ratio < 0.5:
        threshold = threshold * 0.8
    
    return threshold

def create_adjacency_matrix(data, threshold):
    num_videos = len(data)
    adjacency_matrix = np.zeros((num_videos, num_videos), dtype=int)

    for i in range(num_videos):
        for j in range(num_videos):
            # Ignore diagonal of matrix
            if i != j:
                distance = np.linalg.norm(data[i] - data[j])
                if distance <= threshold:
                    adjacency_matrix[i][j] = 1
    return adjacency_matrix

def find_connected_components(adj_matrix):
    n = len(adj_matrix)
    visited = [False] * n
    components = []

    def dfs(node, component):
        visited[node] = True
        component.append(node)
        for neighbor in range(n):
            if adj_matrix[node][neighbor] == 1 and not visited[neighbor]:
                dfs(neighbor, component)

    for i in range(n):
        if not visited[i]:
            component = []
            dfs(i, component)
            components.append(component)

    return components

def get_conn_comp_count(eigenvalues):
    for index in range(len(eigenvalues)):
        if eigenvalues[index] < 1e-5:
            continue
        else:
            return index

def partition(adj_matrix, nodes):
        
        # Degree matrix: Diagonal matrix of node degrees
        degree_matrix = np.diag(adj_matrix.sum(axis=1))

        # Laplacian matrix: D - A
        laplacian = degree_matrix - adj_matrix

        # Step 2: Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

        # Step 3: Use the Fiedler vector (second smallest eigenvector) for partitioning
        index = get_conn_comp_count(eigenvalues)
        fiedler_vector = eigenvectors[:, index]

        # Step 4: Split nodes based on the sign of the Fiedler vector
        positive_set = [nodes[i] for i in range(len(nodes)) if fiedler_vector[i] > 0]
        negative_set = [nodes[i] for i in range(len(nodes)) if fiedler_vector[i] <= 0]

        return positive_set, negative_set

def get_sign_clusters(data, num_clusters, latent_space):
    default_threshold = get_default_threshold(latent_space)
    threshold = get_threshold(default_threshold)
    adjacency_matrix = create_adjacency_matrix(data, threshold)

    clusters = find_connected_components(adjacency_matrix)
    # print("Initial cluster number: ", len(clusters), "\nInitial clusters: \n", clusters)

    ratio = len(clusters) / num_clusters
    # print(f"ratio: {ratio}, threshold: {threshold}, num_clusters: {len(clusters)}, clusters_expected: {num_clusters}")

    while ratio > 1 or ratio < 0.5:
        threshold = get_threshold(threshold, ratio)
        adjacency_matrix = create_adjacency_matrix(data, threshold)
        clusters = find_connected_components(adjacency_matrix)
        ratio = len(clusters) / num_clusters
        # print(f"ratio: {ratio}, threshold: {threshold}, num_clusters: {len(clusters)}, clusters_expected: {num_clusters}")

    while len(clusters) < num_clusters:
        largest_cluster_idx = np.argmax([len(c) for c in clusters])
        largest_cluster = clusters.pop(largest_cluster_idx)

        # Get the subgraph adjacency matrix for this cluster
        subgraph_adj_matrix = adjacency_matrix[np.ix_(largest_cluster, largest_cluster)]

        # Partition the largest cluster
        positive_set, negative_set = partition(subgraph_adj_matrix, largest_cluster)

        # Add the new clusters to the list
        clusters.append(positive_set)
        clusters.append(negative_set)
    
    return clusters

# Visualization using MDS Space
def visualize_MDS(data, cluster):
    # Compute the pairwise distance matrix
    dist_matrix = pairwise_distances(data, metric='euclidean')

    # Apply MDS to reduce the data to 2D
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0)
    data_mds = mds.fit_transform(dist_matrix)

    num_clusters = len(cluster)
    colormap = cm['Spectral']

    # Visualize the data using clusters
    plt.figure(figsize=(10, 8))

    for cluster_idx, cluster in enumerate(cluster):
        cluster_points = data_mds[cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                    label=f'Cluster {cluster_idx+1}', 
                    cmap=colormap)

    plt.legend()
    plt.title("Cluster Visualization in 2D MDS Space")
    plt.grid(True)
    plt.show()

def visualize(data, clusters):
    while True:
        vis_method = int(input("\nHow would you like to visualize the Clusters: 1- MDS space or 2- Group of Video Thumbnails: "))

        if vis_method not in [1 , 2]:
            print("Invalid input. Please choose either '1' or '2'.")
            continue

        while True:
            print(f"Target Labels:\n{Service.target_labels}")
            label = input(f"Choose a label. Enter 0 to stop: ").strip().lower()
            
            if label == '0':
                break
            elif label in Service.target_labels:
                print(f"{label}: {clusters[label]}")
                if vis_method == 1:
                    visualize_MDS(data[label], clusters[label])
                else:
                    print(clusters[label])
            else:
                print("Invalid choice")
        
        next_action = input("Would you like to switch visualization methods or exit? (switch/exit): ").strip().lower()
        
        if next_action == 'exit':
            print("Task complete")
            break
        elif next_action != 'switch':
            print("Invalid input. Please choose 'switch' or 'exit'.")
            continue

def main():
    latent_space = int(input("Select a latent space: 1 - layer3 + PCA, 2 - avgpool + SVD, 3 - HOG + KMeans: "))
    sig_clusters = int(input("Select the number of c most significant clusters to be selected: "))

    data = get_latent_space(latent_space)

    clusters = {}
    for label in data:
        clusters[label] = get_sign_clusters(data[label], sig_clusters, latent_space)

    visualize(data, clusters)


if __name__ == "__main__":
    main()
