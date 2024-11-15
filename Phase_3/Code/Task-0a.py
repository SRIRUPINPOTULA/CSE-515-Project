import json
from sklearn.cluster import KMeans
import sqlite3
#Establish connection to database
from Util.PCA import PCA
from Util.SVD import SVD
import csv
import json
import numpy as np

connection = sqlite3.connect('../database/Phase_2.db')
c = connection.cursor()
with open('../database/total_target_features.json', 'r') as f:
    target_data = json.load(f)
with open('../database/videoID.json', 'r') as f:
    videoID = json.load(f)
with open('../database/category_map.json', 'r') as f:
    category_map = json.load(f)

target_videos = ['golf',  'shoot_ball', 'brush_hair', 'handstand', 'shoot_bow', 
                'cartwheel', 'hit', 'shoot_gun', 'hug', 'sit', 'catch', 
                'jump', 'situp', 'chew', 'kick', 'smile', 'clap', 'kick_ball', 'smoke',
                'climb', 'somersault', 'climb_stairs', 'laugh', 'stand']


def pca_helper(feature_space, k, action):
    Feature_Space_Map = {1: "Layer_3", 2: "Layer_4", 3: "AvgPool", 4: "BOF_HOG", 5: "BOF_HOF"}
    retrieval_query = f"SELECT {Feature_Space_Map[feature_space]} FROM data WHERE videoID % 2 == 0 AND Action_Label == '{action}';"
    c.execute(retrieval_query)
    rows = c.fetchall()

    #Make a list of features matrix
    cleaned_data = []
    if feature_space in [1, 2, 3]:
        for row in rows:
            cleaned_data.append(list(map(float, row[0].strip("[]").split(","))))
    elif feature_space in [4, 5]:
        for row in rows:
            cleaned_data.append(list(map(int, row[0].strip("[]").split())))

    data = np.array(cleaned_data)
    eigen_value_list = []
    eigen_value_list = PCA(data, k, Feature_Space_Map[feature_space])
    min_threshold = float('inf')
    diff=[]
    for i in range(0, len(eigen_value_list)-1):
        curr_diff = eigen_value_list[i] - eigen_value_list[i+1]
        diff.append(curr_diff)
    element = max(diff)
    index = diff.index(element)
    print(f"The Inherent Dimensionality for {action} element is: ", eigen_value_list[index])
    
def SVD_helper(feature_space, latent_count, action):
    Feature_Space_Map = {1: "Layer_3", 2: "Layer_4", 3: "AvgPool", 4: "BOF_HOG", 5: "BOF_HOF"}
    retrieval_query = f"SELECT {Feature_Space_Map[feature_space]} FROM data WHERE videoID % 2 == 0 AND Action_Label == '{action}';"
    c.execute(retrieval_query)
    rows = c.fetchall()

    #Make a list of features matrix
    cleaned_data = []
    if feature_space in [1, 2, 3]:
        for row in rows:
            cleaned_data.append(list(map(float, row[0].strip("[]").split(","))))
    elif feature_space in [4, 5]:
        for row in rows:
            cleaned_data.append(list(map(int, row[0].strip("[]").split())))

    data = np.array(cleaned_data)
    eigen_value_list = []
    eigen_value_list = SVD(data, latent_count, Feature_Space_Map[feature_space])
    min_threshold = float('inf')
    diff=[]
    for i in range(0, len(eigen_value_list)-1):
        curr_diff = eigen_value_list[i] - eigen_value_list[i+1]
        diff.append(curr_diff)
    element = max(diff)
    index = diff.index(element)
    print("The Inherent Dimensionality for {action} element is: ", eigen_value_list[index])

total_action_features=[]

def euclidean(a, b):
    distance_res=0
    for i in range(0, len(a)):
        distance_res += (a[i] - b[i])**2
    return distance_res ** 0.5

def neareast_neighbor(features, clusters, k, action):
    result=[]
    for i in range(0,k):
        a=clusters[i]
        nearest_neighbor=[]
        for _ in range(len(a)):
            nearest_neighbor.append(float('inf'))
        for j in range(0, len(a)):
            for k in range(0, len(a)):
                if j!=k:
                    distance=euclidean(features[i], features[j])
                    if distance<nearest_neighbor[j]:
                        nearest_neighbor[j]=distance
        result.append(nearest_neighbor)
    for i, nn_distances in enumerate(result):
        print(f"Nearest neighbor distances for Cluster {i+1} for action type {action} is: {nn_distances}")       
                    
                

def kmeans_calculator(features,k, action):
    clusters=[]
    for i in range(0,k):
        clusters.append([])

    kmeans  = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features)
    #Compute the Cluster Centres
    cluster_centre = kmeans.cluster_centers_
    cluster_centre_list = cluster_centre.tolist()
    for i in range(0, len(features)):
        dist=[]
        for j in range(0,k):
            dist.append(euclidean(features[i],cluster_centre_list[j]))
            min_value = min(dist)
            index = dist.index(min_value)
        clusters[index].append(i)
    neareast_neighbor(features, clusters, k, action)

def kmeans(layer, k, action):
    target_action = action
    for videoname, action in category_map.items():
        if action==target_action and videoID[videoname]%2==0:
            for video in target_data:
                if videoname in video:
                    if layer==1 or layer==2 or layer==3:
                        a=[]
                        a.extend(video[videoname])
                        total_action_features.append(a[layer-1])
                    elif layer==4:
                        get_query_video_feature = f"SELECT {'BOF_HOG'} FROM data WHERE Video_Name = '{videoname}';"
                        c.execute(get_query_video_feature)
                        rows = c.fetchall()
                        cleaned_str = rows[0][0].strip("[]")
                        total_action_features.append(list(map(int, cleaned_str.split())))
                    else:
                        get_query_video_feature = f"SELECT {'BOF_HOF'} FROM data WHERE Video_Name = '{videoname}';"
                        c.execute(get_query_video_feature)
                        rows = c.fetchall()
                        cleaned_str = rows[0][0].strip("[]")
                        total_action_features.append(list(map(int, cleaned_str.split())))
    kmeans_calculator(total_action_features,k, action)

def main():
    #Gather the latent semantics
    dimen_reduction = int(input("Provide the Feature Space 1 - PCA, 2 - SVD, 3 - KMeans: "))
    #Gather the feature space
    feature_space = int(input("Select a Feature Space from the following: 1 - Layer3, 2 - Layer4, 3 - AvgPool, 4- HOG, 5 - HOF, 6 - Color Histogram : "))
    if dimen_reduction==1:
        k = int(input("Please Provide value for latent semantics: "))
        for i in range(len(target_videos)):
            pca_helper(feature_space,k, target_videos[i])
    elif dimen_reduction==2:
        k = int(input("Please Provide value for latent semantics: "))
        for i in range(len(target_videos)):
            SVD_helper(feature_space, k, target_videos[i])
    else:
        k = int(input("Please Provide value for Clusters: "))
        for i in range(len(target_videos)):
            kmeans(feature_space,k, target_videos[i])

main()