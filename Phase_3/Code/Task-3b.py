import os
import sqlite3
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class VideoSearchTool:
    def __init__(self, db_path, feature_column, num_layers, hashes_per_layer, w, thumbnail_dir, dimensionality_reduction, Feature_Space_Map):
        self.db_path = db_path
        self.feature_column = feature_column
        self.dimensionality_reduction = dimensionality_reduction
        self.num_layers = num_layers
        self.hashes_per_layer = hashes_per_layer
        self.w = w
        self.Feature_Space_Map = Feature_Space_Map
        self.thumbnail_dir = thumbnail_dir
        self.lsh_index = {layer: {} for layer in range(num_layers)}
        self.video_features = {}
        self.load_data()

    def load_data(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        feature_column = self.Feature_Space_Map[self.feature_column]
        query = f"SELECT videoID, {self.Feature_Space_Map[self.feature_column]} FROM data"
        print("The query is: ", query)
        raw_features = []
        video_ids = []
        
        for row in cursor.execute(query):
            videoID, feature_data = row
            feature_vector = np.array(json.loads(feature_data))
            raw_features.append(feature_vector)
            video_ids.append(videoID)
        
        conn.close()
        raw_features = np.array(raw_features)
        reduced_features = self.apply_dimensionality_reduction(raw_features)

        for video_id, feature_vector in zip(video_ids, reduced_features):
            self.video_features[video_id] = feature_vector
            self.add_to_lsh(video_id, feature_vector)

    def apply_dimensionality_reduction(self, data, method='PCA'):
        if method=='PCA':
            row, column = data.shape
            latent_count = min(256, column)
            cov_matrix = np.cov(data, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            sorted_indices = np.argsort(eigenvalues)[::-1]
            eigenvectors_subset = eigenvectors[:, sorted_indices[:latent_count]]
            return np.dot(data, eigenvectors_subset)
        elif method=='SVD':
            return
        else:
            return
    
    def add_to_lsh(self, video_id, feature_vector):
        for layer in range(self.num_layers):
            rp = np.random.randn(self.hashes_per_layer, feature_vector.shape[0])
            biases = np.random.uniform(0, self.w, size=self.hashes_per_layer)
            
            hash_key = tuple(
                int(np.floor((np.dot(feature_vector, hyperplane) + bias) / self.w))
                for hyperplane, bias in zip(rp, biases)
            )
            
            if hash_key not in self.lsh_index[layer]:
                self.lsh_index[layer][hash_key] = []
            self.lsh_index[layer][hash_key].append(video_id)
    
    def euclidean_distance(self, vec1, vec2):
        return np.linalg.norm(vec1 - vec2)

    def search(self, query_video_id, t):
        query_features = self.video_features[query_video_id]
        candidates = set()
        overall_candidates = 0
        
        for layer in range(self.num_layers):
            rp = np.random.randn(self.hashes_per_layer, query_features.shape[0])
            biases = np.random.uniform(0, self.w, size=self.hashes_per_layer)
            
            hash_key = tuple(
                int(np.floor((np.dot(query_features, hyperplane) + bias) / self.w))
                for hyperplane, bias in zip(rp, biases)
            )
            
            if hash_key in self.lsh_index[layer]:
                candidates.update(self.lsh_index[layer][hash_key])
                overall_candidates += len(self.lsh_index[layer][hash_key])
        
        similarity_scores = [
            (video_id, self.euclidean_distance(query_features, self.video_features[video_id]))
            for video_id in candidates
        ]

        similarity_scores.sort(key=lambda x: x[1])
        
        top_videos = similarity_scores[:t]

        print(f"Unique Candidates: {len(candidates)}")
        print(f"Overall Candidates: {overall_candidates}")
        for video_id, score in top_videos:
            print(f"Video {video_id}: Distance {score:.4f}")
        
        self.display_thumbnails_grid(top_videos)
    
    def display_thumbnails_grid(self, top_videos):
        num_videos = len(top_videos)
        cols = 4
        rows = (num_videos + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten()

        for i in range(len(axes)):
            if i < num_videos:
                video_id, score = top_videos[i]
                thumbnail_path = os.path.join(self.thumbnail_dir, f"{video_id}.jpg")
                img = mpimg.imread(thumbnail_path)
                
                axes[i].imshow(img)
                axes[i].set_title(f"Video {video_id}\nSimilarity: {score:.4f}")
                axes[i].axis('off')
            else:
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    db_path = "../Database/Phase_3.db"
    latent_model = int(input("Please provide the input for 1 - Layer3 + PCA, 2 - Avgpool + SVD, 3 - KMeans + HOG: "))
    #feature_column = int(input("Enter the feature column to use: "))
    thumbnail_dir = "../Database/thumbnails"
    num_layers = 3
    hashes_per_layer = 5
    Feature_Space_Map = {1: "Layer_3", 2: "Layer_4", 3: "AvgPool", 4: "BOF_HOG", 5: "BOF_HOF"}
    w = 10
    if latent_model == 1:
        feature_column = 1
        video_search = VideoSearchTool(db_path, feature_column, num_layers, hashes_per_layer, w, thumbnail_dir, "PCA", Feature_Space_Map)
    elif latent_model == 1:
        feature_column = 3
        video_search = VideoSearchTool(db_path, feature_column, num_layers, hashes_per_layer, w, thumbnail_dir, "SVD", Feature_Space_Map)
    else:
        feature_column = 4
        video_search = VideoSearchTool(db_path, feature_column, num_layers, hashes_per_layer, w, thumbnail_dir, "KMeans", Feature_Space_Map)
    
    query_video_id = int(input("Enter the query videoID: "))
    t = int(input("Enter the number of similar videos to retrieve: "))
    
    video_search.search(query_video_id, t)

if __name__ == "__main__":
    main()