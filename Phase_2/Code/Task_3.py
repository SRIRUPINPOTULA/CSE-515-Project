# Task 3: Implement a program which, given (a) a video file name or
# videoID (even or odd, target or non target), (b) a user selected feature
# model from Task 0 or latent semantics from Task 2, and (c) positive integer
# m, identifies and visualizes the most similar m target videos, along with
# their scores, under the selected model or latent space.


# Import Necessary libraries
import os
import json
import cv2

import numpy as np
from scipy.spatial.distance import cdist
import gensim
import gensim.corpora as corpora
from nltk.tokenize import word_tokenize
from prettytable import PrettyTable

import sqlite3

from Task_0b import layer3_implementation, BOF, visualise, euclidean

# Establish connection to database
connection = sqlite3.connect('../database/Phase_2.db')
c = connection.cursor()

# Read the features from the JSON
with open('../database/total_target_features.json', 'r') as f:
    target_data = json.load(f)
with open('../database/total_non_target_features.json', 'r') as f:
    non_target_data = json.load(f)


# Calculate Similar Videos using Kmeans
def kmeans_similarity(query_video, layer_number, l):
    # Read the cluster centres from the json
    with open('../Outputs/Task_2/KMeans_latent.json', 'r') as f:
        cluster_centres = json.load(f)
    layer=[]
    found = False
    # Find the feature of query video
    for video in target_data:
        if query_video in video:
            layer.extend(video[query_video])
            found=True
            break
    # If query video is non target the feature is gathered here
    if found == False:
        for video in non_target_data:
            if query_video in video:
                layer.extend(video[query_video])
                found=True
                break
    if layer_number==1:
        layer = layer[0]
    elif layer_number==2:
        layer = layer[1]
    elif layer_number==3:
        layer=layer[2]
    # Gather the features for HoG and HoF
    elif layer_number==4:
        get_query_video_feature = f"SELECT BOF_HOG FROM data WHERE Video_Name = '{query_video}';"
        c.execute(get_query_video_feature)
        rows = c.fetchall()
        cleaned_str = rows[0][0].strip("[]")
        layer = list(map(int, cleaned_str.split()))
    elif layer_number==5:
        get_query_video_feature = f"SELECT BOF_HOF FROM data WHERE Video_Name = '{query_video}';"
        c.execute(get_query_video_feature)
        rows = c.fetchall()
        cleaned_str = rows[0][0].strip("[]")
        layer = list(map(int, cleaned_str.split()))
    query_video_features=[]
    # For the query video features find the distance from the centre to the query video
    for i in range(0, len(cluster_centres)):
        a=euclidean(layer, cluster_centres[i])
        query_video_features.append(a)
    res = []
    # Loop over all the target videos
    for i in target_data:
        # Gather the feature for the gievn video
        for key, value in i.items():
            video_name=key
            if layer_number==1 or layer_number==2 or layer_number==3:
                curr_feature = value[layer_number-1]
            elif layer_number==4:
                get_query_video_feature = f"SELECT {'BOF_HOG'} FROM data WHERE Video_Name = '{query_video}';"
                c.execute(get_query_video_feature)
                rows = c.fetchall()
                cleaned_str = rows[0][0].strip("[]")
                curr_feature = list(map(int, cleaned_str.split()))
            else:
                get_query_video_feature = f"SELECT {'BOF_HOG'} FROM data WHERE Video_Name = '{query_video}';"
                c.execute(get_query_video_feature)
                rows = c.fetchall()
                cleaned_str = rows[0][0].strip("[]")
                curr_feature = list(map(int, cleaned_str.split()))
            curr_feature_values=[]
            # For the gathered video feature capture the distance from the centres
            for i in range(0, len(cluster_centres)):
                b=euclidean(cluster_centres[i], curr_feature)
                curr_feature_values.append(b)
            # calculate the distance from the query video to the curr_feature values and append it to list
            distance = euclidean(query_video_features, curr_feature_values)
            res.append((distance, key))
    res.sort(key=lambda i:i[0])
    # Print the 'm' similar video
    print(f"******The \"{l}\" most similar videos are: ********")
    video_name = []
    for i in range(0, l):
        print(f'{res[i][1]} : {res[i][0]}')
        video_name.append(res[i][1])
    return video_name

# Calculate lda for the given query video
def infer_new_document_lda(lda_model, dictionary, new_document):
    
    # Preprocess the new document: tokenize
    new_tokens = [word_tokenize(string) for string in new_document]
    
    # Convert the new document to the bag-of-words representation using the trained dictionary
    new_bow = [dictionary.doc2bow(token) for token in new_tokens]
    
    # Get the topic distribution for the new document
    topic_distribution = [lda_model.get_document_topics(bow) for bow in new_bow]
    
    # Convert to a feature matrix (dimensionality-reduced representation)
    def topic_vector(lda_output, num_topics):
        vector = np.zeros(num_topics)
        for topic_num, prob in lda_output:
            vector[topic_num] = prob
        return vector
    
    feature_matrix = np.array([topic_vector(doc, lda_model.num_topics) for doc in topic_distribution])

    return feature_matrix

# Get the closest video
def get_closest_videos(feature_space, query_feature, all_video_names, m):

    # Calculate the distance for all the video features to the query video.
    distances = cdist(query_feature, feature_space, metric='euclidean').flatten()
    # Sort the items based on distances
    indices = np.argsort(distances)[:m]

    # Print the items
    print(f"\n Top-{m} Closest Videos: ")
    t = PrettyTable(["Rank", "Video Name", "Distance"])
    # Append the results to a list for visualisation
    rank = 1
    result_names = []
    for idx in indices:
        t.add_row([rank, all_video_names[idx], distances[idx]])
        result_names.append(all_video_names[idx])
        rank += 1
    print(t)
    return result_names


# Main function
def main():
    # Gather the input from user for video name, latent space, feature space
    input_type = int(input("Provide the 1 - Video File Name or 2 - VideoID: "))
    if input_type == 1:
        video_name = input("Provide the Video File Name: ")
    else:
        video_number = int(input("Provide the VideoID: "))
        with open('../database/videoID.json', 'r') as f:
            videoID = json.load(f)
        for key, value in videoID.items():
            if value == video_number:
                video_name = key
                break
    
    feature_space = int(input("Select a Feature Space from the following: 1 - Layer3, 2 - Layer4, 3 - AvgPool, 4- HOG, 5 - HOF, 6 - Color Histogram, 7 - PCA, 8 - SVD, 9 - LDA, 10 - KMEANS: "))
    m = int(input("Provide the value for m: "))
    
    Feature_Space_Map = {1: "Layer_3", 2: "Layer_4", 3: "AvgPool", 4: "BOF_HOG", 5: "BOF_HOF"}
    Dimensionality_Reduction_Map = {7: "PCA", 8: "SVD", 9: "LDA", 10:"KMeans"}
    if feature_space in [1, 2, 3]:
        videos = layer3_implementation(video_name, feature_space, m)
    elif feature_space == 4:
        videos = BOF(video_name, 'BOF_HOG', m)
    elif feature_space == 5:
        videos = BOF(video_name,'BOF_HOF', m)
    elif feature_space == 6:
        print("Histograms")
    elif feature_space in [7, 8, 9, 10]:

        features = int(input("Select the Feature Space selected for Task2 : 1 - Layer3, 2 - Layer4, 3 - AvgPool, 4- HOG, 5 - HOF, 6 - Color Histogram: "))
        # TODO: rephrase this question
        s = int("How many latent semantics: ")
        
        # get data from db
        c.execute(f"SELECT {Feature_Space_Map[features]} FROM data WHERE Video_Name='{video_name}'")
        rows = c.fetchall()

        cleaned_str = rows[0][0].strip("[]")
        if features in [1, 2, 3]:
            query_feature = list(map(float, cleaned_str.split(",")))
        elif features in [4, 5]:
            query_feature = list(map(int, cleaned_str.split()))
        query_feature = np.array(query_feature).reshape(1, -1)

        retrieval_query = f"SELECT Video_Name, {Feature_Space_Map[features]} FROM data WHERE videoID <= 2872;"
        c.execute(retrieval_query)
        rows = c.fetchall()

        all_video_names = []
        cleaned_data = []
        if features in [1, 2, 3]:
            for row in rows:
                cleaned_data.append(list(map(float, row[1].strip("[]").split(","))))
                all_video_names.append(row[0])
        elif features in [4, 5]:
            for row in rows:
                cleaned_data.append(list(map(int, row[1].strip("[]").split())))
                all_video_names.append(row[0])

        max_len = max(len(lst) for lst in cleaned_data)
        padded_data = [lst + [0] * (max_len - len(lst)) for lst in cleaned_data]
        data = np.array(padded_data)

        # check if latent semantics present in Outputs/Task2
        check_file_name = f"../Outputs/Task_2/videoID-weight_{Dimensionality_Reduction_Map[feature_space]}_{Feature_Space_Map[features]}_{s}.json"
        if not os.path.exists(check_file_name):
            print(f"File not found. {check_file_name}")
            print("Latent Semantics were not prepared in Task 2")
            quit()
        # load the latent semantics
        if feature_space == 7:
            latent_semantic = np.load(f'../Outputs/Task_2/PCA_{Feature_Space_Map[features]}_{s}_left_matrix.npy')
        elif feature_space == 8:
            latent_semantic = np.load(f'../Outputs/Task_2/SVD_{Feature_Space_Map[features]}_{s}_right_matrix.npy')
        elif feature_space == 9:
            lda_model = gensim.models.LdaModel.load(f"../Outputs/Task_2/lda_{Feature_Space_Map[features]}_{s}_model")
            dictionary = corpora.Dictionary.load(f"../Outputs/Task_2/lda_{Feature_Space_Map[features]}_{s}_dictionary")

            string_data = []
            for sublist in data:
                string_data.append(' '.join(map(str, sublist)))
            
            all_topics = infer_new_document_lda(lda_model, dictionary, string_data)
            query_data = []
            for i in query_feature:
                query_data.append(' '.join(map(str, query_feature)))
            query_topics = infer_new_document_lda(lda_model, dictionary, query_data)
            get_closest_videos(all_topics, query_topics, all_video_names, m)

        elif feature_space == 10:
            videos = kmeans_similarity(video_name, features, m)

        if feature_space in [7, 8]:
            get_closest_videos(np.dot(data, latent_semantic), np.dot(query_feature, latent_semantic), all_video_names, m)
        
    # Gather the video name from the user for visualisation
    input_type = int(input("Please Select Visualisation Techniques 1 - Opencv : "))
    if input_type == 1:
        while True:
            value = int(input("Do you want to visualise more videos: 1 - Yes, 2 - No: "))
            if value == 2:
                print("Visualised the videos")
                break
            elif value == 1:
                print(f"Please Select Videos from 1 - {m}")
                query_video = input("Enter the Video name: ")
                found = False
                for video in target_data:
                    if query_video in video:
                        path=f'../dataset/target_videos/{query_video}'
                        found=True
                        break
                if found == False:
                    for video in non_target_data:
                        if query_video in video:
                            path=f'../dataset/non_target_videos/{query_video}'
                            found=True
                            break
                visualise(path)

main()
