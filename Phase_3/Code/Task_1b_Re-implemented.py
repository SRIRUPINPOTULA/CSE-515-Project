from Util.KMeanslatentmodel import latent_model_generator
import numpy as np
import sqlite3
import json
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from collections import Counter

target_videos = ['golf',  'shoot_ball', 'brush_hair', 'handstand', 'shoot_bow', 
                'cartwheel', 'hit', 'shoot_gun', 'hug', 'sit', 'catch', 
                'jump', 'situp', 'chew', 'kick', 'smile', 'clap', 'kick_ball', 'smoke',
                'climb', 'somersault', 'climb_stairs', 'laugh', 'stand']

def initialize_centroids(features, k):
    if features is None or features.shape[0] < k:
        raise ValueError("Error: Invalid `features` input or insufficient data points for k clusters.")
    indices = np.random.choice(features.shape[0], k, replace=False)
    return features[indices]

def assign_clusters(features, centroids):
    distances = np.array([[np.linalg.norm(point - centroid) for centroid in centroids] for point in features])
    return np.argmin(distances, axis=1)

def update_centroids(features, labels, k):
    return np.array([features[labels == i].mean(axis=0) for i in range(k)])

def kmeans_clustering(features, k, max_iters=100, tol=0.3):
    centroids = initialize_centroids(features, k)
    for _ in range(max_iters):
        #print("In the kmeans clustering ")
        labels = assign_clusters(features, centroids)
        new_centroids = update_centroids(features, labels, k)
        
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids
    return labels, centroids

def visualize_clusters(features, labels, k):
    mds = MDS(n_components=2, random_state=0)
    features_2d = mds.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    for cluster in range(k):
        cluster_points = features_2d[labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster+1}')
    plt.legend()
    plt.title("Cluster Visualization in 2D MDS Space")
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.show()

    print("Groups of Labels in Each Cluster:")
    for cluster in range(k):
        cluster_labels = np.where(labels == cluster)[0]
        print(f"Cluster {cluster+1}: Labels {cluster_labels}")

def helper_visualise(features, labels, k, centroids, target_label_range):
    labels = assign_clusters(features, centroids)
    labels_list = labels.tolist()
    assignment = []
    
    for i in range(0, k):
        cluster = []
        for j in range(0, len(labels_list)):
            if labels_list[j]==i:
                cluster.append(j)
        assignment.append(cluster)
    #print(assignment)
    label_assignment = []
    for i in range(len(assignment)):
        cluster = assignment[i]
        curr_label = []
        for j in range(len(cluster)):
            element = cluster[j]
            for key, item in target_label_range.items():
                if item[0]<=element  and item[1]>=element:
                    curr_label.append(key)
        counter_label = Counter(curr_label)
        max_label, label_count = counter_label.most_common(1)[0]
        label_assignment.append(max_label)

    for i in range(k):
        cluster_points = features[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}: {label_assignment[i]}")

    plt.scatter(centroids[:, 0], centroids[:, 1], label="Centroids")
        
    plt.legend()
    plt.title("K-Means Group Clustering")
    plt.xlabel("Centriods")
    plt.ylabel("Feature Space")
    plt.show()


def main():
    s=4
    latent_model, target_label_range = latent_model_generator(s)
    #print("*******************The Final Dimension is: **********************", len(latent_model))
    #print("The Final Dimension is: ", len(latent_model[0]))
    features = np.array(latent_model)
    k=24
    while True:
        labels, centroids = kmeans_clustering(features, k)
        #visualize_clusters(features, labels, k=len(centroids))
        print(labels)
        labels = assign_clusters(features, centroids)
        labels_list = labels.tolist()
        assignment = []
        
        for i in range(0, k):
            cluster = []
            for j in range(0, len(labels_list)):
                if labels_list[j]==i:
                    cluster.append(j)
            assignment.append(cluster)
        #print(assignment)
        label_assignment = []
        for i in range(len(assignment)):
            cluster = assignment[i]
            curr_label = []
            for j in range(len(cluster)):
                element = cluster[j]
                for key, item in target_label_range.items():
                    if item[0]<=element  and item[1]>=element:
                        curr_label.append(key)
            counter_label = Counter(curr_label)
            max_label, label_count = counter_label.most_common(1)[0]
            label_assignment.append(max_label)
        unique_elements = set(label_assignment)
        print("tHe label assignment is: ", len(unique_elements))
        if len(unique_elements)==24:
            break
        k+=1
        print("the unique elements are: ", len(unique_elements))
        #k+=1
    print("THE TOTAL NUMBER OF CLUSTERS: ", k)
    helper_visualise(features, labels, k, centroids, target_label_range)

main()