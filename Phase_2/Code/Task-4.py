#Import the necessary libraries 
import json
import cv2
import sqlite3
import numpy as np
from sklearn.cluster import KMeans
#Establish connection to the database
connection = sqlite3.connect('../database/Phase_2.db')
c = connection.cursor()

#Load the features for the target features
with open('../database/total_target_features.json', 'r') as f:
    target_data = json.load(f)

#Load the category map features that are extracted for each label
with open('../database/category_map_kmeans.json', 'r') as f:
    kmeans_preprocess = json.load(f)

#Load the features for all the action centers
with open('../database/action_centres.json', 'r') as f:
    clusters = json.load(f)

# #Calculate the Euclidean Distance
# def euclidean(a, b):
#     distance_res=0
#     for i in range(0, len(a)):
#         distance_res += (a[i] - b[i])**2
#     return distance_res ** 0.5

def manhattan(a, b):
    res=0
    for i in range(0,len(a)):
        res = res + abs(a[i]-b[i])
    return res

def pca():
    return

def svd():
    return 

def lda():
    return

#Functio that lists m similar videos
def kmeans(label, feature_space,m):
    #Extract the label feature 
    query_features=kmeans_preprocess[label]
    query_feature = query_features[feature_space-1]
    #For a specific cluster extract the centroids
    all_clusters = clusters[label]
    cluster_centre = all_clusters[feature_space-1]
    res=[]
    #For the target videos gather the features
    for video in target_data:
        video_name =video.keys()
        layer_value_1=[]
        for key, value in video.items():
            #Extract the features for layer1, layer2 and layer3
            if feature_space==1 or feature_space==2 or feature_space==3:           
                video_name=key
                layer_values=value
                layer_value=layer_values[feature_space-1]
                a=[]
                #Calculate the Distance from the cluster centers
                for i in range(0, len(cluster_centre)):
                    dist = manhattan(layer_value, cluster_centre[i])
                    layer_value_1.append(dist)
            #Extract the features for HoG
            elif feature_space==4:
                video_name=key
                get_query_video_feature = f"SELECT {'BOF_HOG'} FROM data WHERE Video_Name = '{video_name}';"
                c.execute(get_query_video_feature)
                rows = c.fetchall()
                cleaned_str = rows[0][0].strip("[]")
                layer_value = list(map(int, cleaned_str.split()))
                for i in range(0, len(cluster_centre)):
                    dist = manhattan(layer_value, cluster_centre[i])
                    layer_value_1.append(dist)
            #Extract the features for HoF
            elif feature_space==5:
                video_name=key
                get_query_video_feature = f"SELECT {'BOF_HOF'} FROM data WHERE Video_Name = '{video_name}';"
                c.execute(get_query_video_feature)
                rows = c.fetchall()
                cleaned_str = rows[0][0].strip("[]")
                layer_value = list(map(int, cleaned_str.split()))
                layer_value_1=[]
                #Calculate the Distance from the cluster centers
                for i in range(0, len(cluster_centre)):
                    dist = manhattan(layer_value, cluster_centre[i])
                    layer_value_1.append(dist)
            #Calculate the distance between query video and current video
            distance = manhattan(query_feature, layer_value_1)
            res.append((distance, key))
    #Sort based on distance
    res.sort(key=lambda i:i[0])
    print(f"******The \"{m}\" most similar videos are: ********")
    #Printing the videos along with distance
    video_name = []
    for i in range(0, m):
        print(f'{res[i][1]} : {res[i][0]}')
        video_name.append(res[i][1])
    return
    

#Define a main function
def main():
    #Gather the label
    label = input("Please provide the label: ")
    #Gather the latent semantics
    latent_Semantic = int(input("Please Provide the Latent Semantics 1 - PCA, 2 - SVD, 3 - LDA, 4 - KMeans: "))
    m = int(input("Please provide a value for m: "))
    #Gather the feature space
    if latent_Semantic==1:
        pca = pca()
    elif latent_Semantic==2:
        svd = svd()
    elif latent_Semantic==3:
        lda = lda()
    else:
        feature_space = int(input("Select a Feature Space from the following: 1 - Layer3, 2 - Layer4, 3 - AvgPool, 4 - HOG, 5 - HOF, 6 - Color Histogram : "))
        kmeans(label, feature_space,m)
main()