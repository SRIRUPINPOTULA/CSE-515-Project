# Import the necessary libraries
import numpy as np

# import sys
# np.set_printoptions(threshold=sys.maxsize)

from Util.services import ServiceClass as Service

def get_latent_space(latent_space):
    data_map = {}

    for label in Service.target_labels:

        # TODO: Get Inherent Dimensionality
        s = 50

        label_query = f"SELECT {Service.feature_space_map[latent_space]} FROM data WHERE videoID % 2 == 0 AND Action_Label='{label}';"

        data = Service.get_data_from_db(label_query, latent_space, s)
        
        data_map[label] = data
    return data_map

def get_threshold(threshold, ratio = None):
    # TODO change default value based on Latent Space selected
    # TODO verify if 1.5 and 0.8 factor is good
    
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

def get_sign_clusters(data, num_clusters):
    threshold = get_threshold(1)
    adjacency_matrix = create_adjacency_matrix(data, threshold)

    clusters = find_connected_components(adjacency_matrix)
    print("Initial cluster number: ", len(clusters))
    print("Initial clusters: \n", clusters)

    ratio = len(clusters) / num_clusters
    print(f"ratio: {ratio}, threshold: {threshold}, num_clusters: {len(clusters)}, clusters_expected: {num_clusters}")

    # TODO: change to if loop with limited iteration and exit condition?
    while ratio > 1 or ratio < 0.5:
        threshold = get_threshold(threshold, ratio)
        adjacency_matrix = create_adjacency_matrix(data, threshold)
        clusters = find_connected_components(adjacency_matrix)
        ratio = len(clusters) / num_clusters
        print(f"ratio: {ratio}, threshold: {threshold}, num_clusters: {len(clusters)}, clusters_expected: {num_clusters}")

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


def main():
    latent_space = int(input("Select a latent space: 1 - layer3 PCA, 2 - avgpool SVD, 3 - HOG KMeans: "))
    sig_clusters = int(input("Select the number of c most significant clusters to be selected: "))

    data = get_latent_space(latent_space)

    clusters = {}
    for label in data:
        clusters[label] = get_sign_clusters(data[label], sig_clusters)
        print(f"label: {label}\n", clusters[label])

    # TODO: loop the visualize till user ends it. Ask which label. Create function to handle visualization
    vis_method = int(input("How would you like to visualize the Clusters: 1- MDS space or 2- Group of Video Thumbnails: "))


if __name__ == "__main__":
    main()
