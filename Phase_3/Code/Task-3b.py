import os
import sqlite3
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class VideoSearchTool:
    def __init__(self, db_path, feature_column, num_layers, hashes_per_layer, w, thumbnail_dir):
        self.db_path = db_path
        self.feature_column = feature_column
        self.num_layers = num_layers
        self.hashes_per_layer = hashes_per_layer
        self.w = w
        self.thumbnail_dir = thumbnail_dir
        self.lsh_index = {layer: {} for layer in range(num_layers)}
        self.video_features = {}
        self.load_data()

    def load_data(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = f"SELECT videoID, {self.feature_column} FROM data"
        for row in cursor.execute(query):         #Load video features from the database.
            videoID, feature_data = row
            feature_vector = np.array(json.loads(feature_data)) 
            self.video_features[videoID] = feature_vector
            
            self.add_to_lsh(videoID, feature_vector)
        
        conn.close()
    
    def add_to_lsh(self, video_id, feature_vector):
       #Hash the feature vector and store in the LSH index.
        for layer in range(self.num_layers):
            rp = np.random.randn(self.hashes_per_layer, feature_vector.shape[0])
            biases = np.random.uniform(0, self.w, size=self.hashes_per_layer)
            
            hash_key = tuple(
                int(np.floor((np.dot(feature_vector, hyperplane) + bias) / self.w)) # Hashing function defined by us.
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
        cols = 3  # Display 3 thumbnails per row
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
    db_path = "Phase_2(1).db"
    feature_column = input("Enter the feature column to use (e.g., AvgPool, Layer_3): ")
    thumbnail_dir = "databsae/thumbnails"
    num_layers = 3
    hashes_per_layer = 5
    w = 10
    
    video_search = VideoSearchTool(db_path, feature_column, num_layers, hashes_per_layer, w, thumbnail_dir)
    
    query_video_id = int(input("Enter the query videoID: "))
    t = int(input("Enter the number of similar videos to retrieve: "))
    
    video_search.search(query_video_id, t)

if __name__ == "__main__":
    main()
