import json
from sklearn.cluster import KMeans
import sqlite3
#Establish connection to database
from Util.PCA import PCA
import csv
import json
import numpy as np

connection = sqlite3.connect('../database/Phase_3.db')
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

def mean_eigen_value(feature_matrix):
    size = len(feature_matrix)
    total_value = 0
    for i in range(0, size):
        total_value+=feature_matrix[i]
    mean = total_value/size
    inherent_dim = 0
    for i in range(0, len(feature_matrix)):
        if feature_matrix[i]>=mean:
            inherent_dim+=1
    return inherent_dim

def pca_helper(feature_space, action):
    Feature_Space_Map = {1: "Layer_3", 2: "Layer_4", 3: "AvgPool", 4: "BOF_HOG", 5: "BOF_HOF"}
    retrieval_query = f"SELECT {Feature_Space_Map[feature_space]} FROM data WHERE All_Label == '{action}';"
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
    eigen_value_list = PCA(data, 1, Feature_Space_Map[feature_space])
    ans = mean_eigen_value(eigen_value_list)
    # min_threshold = float('inf')
    # diff=[]
    # for i in range(0, len(eigen_value_list)-1):
    #     curr_diff = eigen_value_list[i] - eigen_value_list[i+1]
    #     diff.append(curr_diff)
    # element = max(diff)
    # index = diff.index(element)
    print(f"The Inherent Dimensionality for {action} element is: ", ans)

def main():
    feature_space = int(input("Select a Feature Space from the following: 1 - Layer3, 2 - Layer4, 3 - AvgPool, 4- HOG, 5 - HOF, 6 - Color Histogram : "))
    for i in range(0, len(target_videos)):
        pca_helper(feature_space, target_videos[i])
    
main()
