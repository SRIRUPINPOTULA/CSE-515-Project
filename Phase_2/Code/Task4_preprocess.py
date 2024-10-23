#Import Libraries
import json
import cv2
import sqlite3
import numpy as np
from sklearn.cluster import KMeans
import sqlite3

#Establish Connection
connection = sqlite3.connect('../database/Phase_2.db')
c = connection.cursor()

#Maps to store the feature values
kmeans_map = {}
cluster_centres_actions = {}

#Read the json files for total target features and category map
with open('../database/total_target_features.json', 'r') as f:
    target_data = json.load(f)
with open('../database/category_map.json', 'r') as f:
    category_map = json.load(f)

#List for the tatget videos
target_videos = ['golf',  'shoot_ball', 'brush_hair', 'handstand', 'shoot_bow', 
                'cartwheel', 'hit', 'shoot_gun', 'hug', 'sit', 'catch', 
                'jump', 'situp', 'chew', 'kick', 'smile', 'clap', 'kick_ball', 'smoke',
                'climb', 'somersault', 'climb_stairs', 'laugh', 'stand']

#Distance Function to calculate Euclidean
def euclidean(a, b):
    distance_res=0
    for i in range(0, len(a)):
        distance_res += (a[i] - b[i])**2
    return distance_res ** 0.5

#Function to calculate the mean
def feature_calculator(layer):
    feauter_np = np.array(layer)
    mean_feature = np.mean(feauter_np, axis=0)
    return mean_feature.tolist()

#Function to calculate the cluuster centres for all the video labels
def kmeans_preprocess(action):
    layer_1=[]
    layer_2=[]
    layer_3=[]
    #For each category make a feature matrix 
    for videoname, category in category_map.items():
        if category==action:
            for video in target_data:
                if videoname in video:
                    a=[]
                    a.extend(video[videoname])
                    layer_1.append(a[0])
    #Layer 1 Features for a action label
    layer_1_features = np.array(layer_1)
    #Train the kmeans model and calculate the cluster centers
    kmeans  = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(layer_1_features)
    layer_1_cluster_centres = kmeans.cluster_centers_.tolist()
    layer_map=[]
    #For each video feature compute the distance with cluster center
    for i in range(len(layer_1)):
        a=[]
        for j in range(len(layer_1_cluster_centres)):
            dist=euclidean(layer_1[i],layer_1_cluster_centres[j])
            a.append(dist)
        layer_map.append(a)
    total_layer_1 = feature_calculator(layer_map)
    #For each category make a feature matrix 
    for videoname, category in category_map.items():
        if category==action:
            for video in target_data:
                if videoname in video:
                    a=[]
                    a.extend(video[videoname])
                    layer_2.append(a[1])
    layer_2_features = np.array(layer_2)
    #Train the kmeans model and calculate the cluster centers
    kmeans  = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(layer_2_features)
    layer_2_cluster_centres = kmeans.cluster_centers_.tolist()
    layer_map=[]
    #For each video feature compute the distance with cluster center
    for i in range(len(layer_2)):
        a=[]
        for j in range(len(layer_2_cluster_centres)):
            dist=euclidean(layer_2[i],layer_2_cluster_centres[j])
            a.append(dist)
        layer_map.append(a)
    #For each category make a for layer2
    total_layer_2 = feature_calculator(layer_map)
    for videoname, category in category_map.items():
        if category==action:
            for video in target_data:
                if videoname in video:
                    a=[]
                    a.extend(video[videoname])
                    layer_3.append(a[2])
    #Similar find the features for the layer3 
    layer_3_features = np.array(layer_3)
    kmeans  = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(layer_3_features)
    layer_3_cluster_centres = kmeans.cluster_centers_.tolist()
    layer_map=[]
    #For each video feature compute the distance with cluster center
    for i in range(len(layer_3)):
        a=[]
        for j in range(len(layer_3_cluster_centres)):
            dist=euclidean(layer_3[i],layer_3_cluster_centres[j])
            a.append(dist)
        layer_map.append(a)
    
    total_layer_3 = feature_calculator(layer_map)
    ####Integrate all the features for HoG
    layer_4=[]
    layer_5=[]
    #For each video extract the HoG features
    for videoname, category in category_map.items():
        if category==action:
            for video in target_data:
                if videoname in video:
                    get_query_video_feature = f"SELECT {'BOF_HOG'} FROM data WHERE Video_Name = '{videoname}';"
                    c.execute(get_query_video_feature)
                    rows = c.fetchall()
                    cleaned_str = rows[0][0].strip("[]")
                    query_feature = list(map(int, cleaned_str.split()))
                    #query_feature = np.array(query_feature).reshape(1, -1)
                    layer_4.append(query_feature)
    #Make a numpy array for all the features which are HoG
    layer_4_features = np.array(layer_4)
    kmeans  = KMeans(n_clusters=5, random_state=42)
    #Make the cluster centers
    kmeans.fit(layer_4_features)
    layer_4_cluster_centres = kmeans.cluster_centers_.tolist()
    layer_map=[]
    for i in range(len(layer_4)):
        a=[]
        for j in range(len(layer_4_cluster_centres)):
            dist=euclidean(layer_4[i],layer_4_cluster_centres[j])
            a.append(dist)
        layer_map.append(a)
    #Total features for layer4
    total_layer_4 = feature_calculator(layer_map)
    
    #Gather all the features for HoF
    for videoname, category in category_map.items():
        if category==action:
            for video in target_data:
                if videoname in video:
                    get_query_video_feature = f"SELECT {'BOF_HOF'} FROM data WHERE Video_Name = '{videoname}';"
                    c.execute(get_query_video_feature)
                    rows = c.fetchall()
                    cleaned_str = rows[0][0].strip("[]")
                    query_feature = list(map(int, cleaned_str.split()))
                    #query_feature = np.array(query_feature).reshape(1, -1)
                    layer_5.append(query_feature)
    #Make a numpy array for all the features which are HoF
    layer_5_features = np.array(layer_5)
    #Make the cluster centers
    kmeans  = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(layer_5_features)
    layer_5_cluster_centres = kmeans.cluster_centers_.tolist()
    layer_map=[]
    for i in range(len(layer_5)):
        a=[]
        for j in range(len(layer_5_cluster_centres)):
            dist=euclidean(layer_5[i],layer_5_cluster_centres[j])
            a.append(dist)
        layer_map.append(a)
    #Total features for HoG
    total_layer_5 = feature_calculator(layer_map)
    
    final_features =[total_layer_1, total_layer_2, total_layer_3, total_layer_4, total_layer_5]
    kmeans_map[action]=final_features
    
    #Append make a list for all the clusters
    cluster_centres_actions[action]=[layer_1_cluster_centres, layer_2_cluster_centres,
                                    layer_3_cluster_centres, layer_4_cluster_centres,
                                    layer_5_cluster_centres]
    return


def main():
    #For each target videos calculate the feature for each label
    for video in target_videos:
        kmeans_preprocess(video)
    #Dump the features into a json
    with open('../database/category_map_kmeans.json', 'w') as f:
        json.dump(kmeans_map,f)
    #Store the cluster centres for each action
    with open('../database/action_centres.json', 'w') as f:
        json.dump(cluster_centres_actions,f)
        
main()