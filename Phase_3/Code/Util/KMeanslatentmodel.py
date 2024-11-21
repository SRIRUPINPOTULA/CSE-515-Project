import json
from sklearn.cluster import KMeans
import sqlite3
import csv
import numpy as np

connection = sqlite3.connect('../database/Phase_3.db')
c = connection.cursor()

target_label_range = {}

target_videos = ['golf',  'shoot_ball', 'brush_hair', 'handstand', 'shoot_bow', 
                'cartwheel', 'hit', 'shoot_gun', 'hug', 'sit', 'catch', 
                'jump', 'situp', 'chew', 'kick', 'smile', 'clap', 'kick_ball', 'smoke',
                'climb', 'somersault', 'climb_stairs', 'laugh', 'stand']

def euclidean(a, b):
    distance_res=0
    for i in range(0, len(a)):
        distance_res += (a[i] - b[i])**2
    return distance_res ** 0.5

def KMeans_implementation(feature_matrix, s):
    #Before training using kmeans separate the videoID
    kmeans = KMeans(n_clusters=s, random_state=42)
    kmeans.fit(feature_matrix)
    #Compute the Cluster Centres
    cluster_centre = kmeans.cluster_centers_
    cluster_centre_list = cluster_centre.tolist()
    latent_model = []
    feature_matrix_list = feature_matrix.tolist()
    for i in range(0, len(feature_matrix_list)):
        curr_feature=feature_matrix_list[i]
        curr_dis=[]
        for j in range(0, len(cluster_centre_list)):
            dist =euclidean(curr_feature, cluster_centre_list[j])
            curr_dis.append(dist)
        latent_model.append(curr_dis)  
    return latent_model

def latent_model_generator(feature_space):
    mean_features = []
    Feature_Space_Map = {1: "Layer_3", 2: "Layer_4", 3: "AvgPool", 4: "BOF_HOG", 5: "BOF_HOF"}
    cleaned_data = []
    initial = 0
    for i in range(0, len(target_videos)):
        action = target_videos[i]
        retrieval_query = f"SELECT {Feature_Space_Map[feature_space]} FROM data WHERE All_Label == '{action}';"
        c.execute(retrieval_query)
        rows = c.fetchall()
        for row in rows:
            cleaned_data.append(list(map(int, row[0].strip("[]").split())))
        #print("The size of cleaned data:", len(cleaned_data))
        target_label_range[action]= [initial, len(cleaned_data)-1]
        initial = len(cleaned_data)
    max_len = max(len(lst) for lst in cleaned_data)
    padded_data = [lst + [0] * (max_len - len(lst)) for lst in cleaned_data]
    data = np.array(padded_data)
    ##Gathered the feature model representation
    row, column = data.shape
    latent_model_kmeans = KMeans_implementation(data, 87)
    print("The Final Dimesiosn is: ", len(latent_model_kmeans))
    print("The Final Dimesiosn is: ", len(latent_model_kmeans[0]))
    return latent_model_kmeans, target_label_range
        
def main():
    feature_space = 4
    latent_model_generator(feature_space)
    print(target_label_range)
main()