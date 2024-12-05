import os
import sqlite3
import json
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from prettytable import PrettyTable

class VideoSearchTool:
    def __init__(self, feature_column, num_layers, hashes_per_layer, w, dimensionality_reduction):
        self.feature_column = feature_column
        self.dimensionality_reduction = dimensionality_reduction
        self.num_layers = num_layers
        self.hashes_per_layer = hashes_per_layer # Hash functions per LSH layer
        self.w = w # Bucket width for LSH
        self.lsh_index = {layer: {} for layer in range(5*num_layers)} #initialise LSH index
        # Generate random hyperplanes and biases for hashing
        self.hyperplanes = [
            np.random.randn(self.hashes_per_layer, 256)
            for _ in range(5*self.num_layers)
        ]
        self.biases = [
            np.random.uniform(0, self.w, size=self.hashes_per_layer)
            for _ in range(5*self.num_layers)
        ]
        self.video_features = {} # Dictionary to store video features
        self.load_data() #to get preprocessed data

    video_names = []

    def load_data(self):
        conn = sqlite3.connect("../Database/Phase_3.db")
        cursor = conn.cursor()
        
        query = f"SELECT videoID, Video_Name, {self.feature_column} FROM data"
        raw_features = []
        video_ids = []
        
        for row in cursor.execute(query):
            videoID, video_name, feature_data = row
            if self.feature_column in ["Layer_3", "AvgPool"]:
                feature_vector = np.array(json.loads(feature_data)) # Parse JSON feature
            else:
                feature_data = feature_data.strip("[]").split()
                feature_vector = np.array(feature_data, dtype=int) # Parse space-separated data
            
            raw_features.append(feature_vector)
            video_ids.append(videoID)
            self.video_names.append(video_name)
        
        conn.close()

        # Ensure feature vectors have consistent size
        processed_arrays = []
        for arr in raw_features:
            if len(arr) < 480:
                padded_arr = np.pad(arr, (0, 480 - len(arr)), mode='constant', constant_values=0)
            elif len(arr) > 480 and self.feature_column not in ["Layer_3", "AvgPool"]:
                padded_arr = arr[:480]
            else:
                padded_arr = arr
            processed_arrays.append(padded_arr)
        
        raw_features = np.array(processed_arrays)
        reduced_features = self.apply_dimensionality_reduction(raw_features, self.dimensionality_reduction)

        # Store processed features and add to LSH index
        for video_id, feature_vector in zip(video_ids, reduced_features):
            self.video_features[video_id] = feature_vector
            self.add_to_lsh(video_id, feature_vector)

    def apply_dimensionality_reduction(self, data, dimensionality_reduction):
        if dimensionality_reduction=='PCA':
            # Perform PCA to reduce dimensionality
            row, column = data.shape
            latent_count = min(256, column)
            cov_matrix = np.cov(data, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            sorted_indices = np.argsort(eigenvalues)[::-1]
            eigenvectors_subset = eigenvectors[:, sorted_indices[:latent_count]]
            return np.dot(data, eigenvectors_subset)
        
        elif dimensionality_reduction=='SVD':
            DtD = np.dot(data.T, data)

            # Calculate eigenvalues and eigenvectors for D^T D
            eigenvalues_V, V = np.linalg.eigh(DtD)
            latent_count = 256
            # Sort the eigenvalues to get the top latent semantics
            sorted_indices = np.argsort(eigenvalues_V)[::-1]
            eigenvalues_V = eigenvalues_V[sorted_indices]
            V = V[:, sorted_indices]
                
            V_subset = V[:, :latent_count]

            # Data in Reduced Dimensional space
            svd_data = np.dot(data, V_subset)

            return svd_data
        
        else:
            latent_count = 256
            kmeans = KMeans(n_clusters=latent_count, random_state=42)
            kmeans.fit(data)
            
            # Get the Cluster Centers
            cluster_centers = kmeans.cluster_centers_
            latent_model = cdist(data, cluster_centers, 'euclidean')
            return latent_model
    
    def add_to_lsh(self, video_id, feature_vector):
        for layer in range(5*self.num_layers):
            rp = self.hyperplanes[layer]
            biases = self.biases[layer]
            
            # Compute hash keys for the feature vector
            hash_key = tuple(
                int(np.floor((np.dot(feature_vector, hyperplane) + bias) / self.w))
                for hyperplane, bias in zip(rp, biases)
            )
            
            if hash_key not in self.lsh_index[layer]:
                self.lsh_index[layer][hash_key] = []
            if video_id not in self.lsh_index[layer][hash_key]:
                self.lsh_index[layer][hash_key].append(video_id)
    
    def euclidean_distance(self, vec1, vec2):
        return np.linalg.norm(vec1 - vec2) # Get euclidean distance

    def search(self, query_video_id, t):
        query_features = self.video_features[query_video_id]
        candidates = set() # Unique candidates from LSH index
        overall_candidates = 0
        
        for layer in range(self.num_layers):
            rp = self.hyperplanes[layer]
            biases = self.biases[layer]
            
            hash_key = tuple(
                int(np.floor((np.dot(query_features, hyperplane) + bias) / self.w))
                for hyperplane, bias in zip(rp, biases)
            )
            
            if hash_key in self.lsh_index[layer]:
                candidates.update(self.lsh_index[layer][hash_key])
                overall_candidates += len(self.lsh_index[layer][hash_key])

        if len(candidates) < t:
            for layer in range(self.num_layers, 5*self.num_layers):
                rp = self.hyperplanes[layer]
                biases = self.biases[layer]
                
                hash_key = tuple(
                    int(np.floor((np.dot(query_features, hyperplane) + bias) / self.w))
                    for hyperplane, bias in zip(rp, biases)
                )
                
                if hash_key in self.lsh_index[layer]:
                    candidates.update(self.lsh_index[layer][hash_key])
                    overall_candidates += len(self.lsh_index[layer][hash_key])

                if len(candidates) >= t:
                    break
            print("Layers Used: ", layer+1)
        
        # Compute similarity scores for candidates
        distance_metric = [
            (video_id, self.video_names[video_id], self.euclidean_distance(query_features, self.video_features[video_id]))
            for video_id in candidates
        ]

        distances = [item[2] for item in distance_metric]
        max_distance = max(distances)
        min_distance = min(distances)

        # Normalize distances to similarity metric
        similarity_scores = [(item[0], item[1], 1 - (item[2] - min_distance) / (max_distance - min_distance)) for item in distance_metric]

        similarity_scores.sort(key=lambda x: x[2], reverse=True) # Sort by similarity score
        
        top_videos = similarity_scores[:t] # Retrieve top t videos

        print(f"\nUnique Candidates: {len(candidates)}")
        print(f"Overall Candidates: {overall_candidates}")
        
        table = PrettyTable()
        table.field_names = ["Rank", "VideoID", "Video Name", "Similarity Score"]

        index = 1
        print(f"\nTop {t}-most similar videos:")
        for video_id, name, score in top_videos:
            table.add_row([index, video_id, name, f"{score:.4f}"])
            index += 1
        print(table)
        
        self.display_thumbnails_grid(top_videos)
    
    #display the thumbnails for the searched similar videos
    def display_thumbnails_grid(self, top_videos):
        num_videos = len(top_videos)
        cols = 4
        rows = math.ceil(num_videos / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten()

        for i in range(len(axes)):
            if i < num_videos:
                video_id, name, score = top_videos[i]
                thumbnail_path = os.path.join("../Database/Thumbnails_IDs/", f"{video_id}.jpg")
                img = mpimg.imread(thumbnail_path)
                
                axes[i].imshow(img)
                axes[i].set_title(f"Video {video_id}\nSimilarity: {score:.4f}", color='white', fontsize=10)
            axes[i].axis('off')
        
        # plt.tight_layout()
        plt.show()

def main():
    latent_model = int(input("Please provide the input for 1 - Layer3 + PCA, 2 - Avgpool + SVD, 3 - KMeans + HOG: "))
    num_layers = int(input("Enter the number of layers: "))
    hashes_per_layer = int(input("Enter the number of hashes per layer: "))

    if latent_model == 1:
        w = 10
        video_search = VideoSearchTool("Layer_3", num_layers, hashes_per_layer, w, "PCA")
    elif latent_model == 2:
        w = 20
        video_search = VideoSearchTool("AvgPool", num_layers, hashes_per_layer, w, "SVD")
    elif latent_model == 3:
        w = 50
        video_search = VideoSearchTool("BOF_HOG", num_layers, hashes_per_layer, w, "KMeans")
    
    query_video_id = int(input("\nEnter the query videoID: "))
    t = int(input("Enter the number of similar videos to retrieve: "))
    
    video_search.search(query_video_id, t)


if __name__ == "__main__":
    main()
