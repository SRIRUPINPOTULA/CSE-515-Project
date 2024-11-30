#Import the libraries
import json
import sqlite3
import csv
import json
import numpy as np
#Establish connection to the databse
connection = sqlite3.connect('../Database/Phase_3.db')
c = connection.cursor()

#List of target labels
target_videos = ['golf',  'shoot_ball', 'brush_hair', 'handstand', 'shoot_bow', 
                'cartwheel', 'hit', 'shoot_gun', 'hug', 'sit', 'catch', 
                'jump', 'situp', 'chew', 'kick', 'smile', 'clap', 'kick_ball', 'smoke',
                'climb', 'somersault', 'climb_stairs', 'laugh', 'stand']

#Calculate the eigen values for the features
def mean_eigen_value(feature_matrix):
    size = len(feature_matrix)
    total_value = 0
    for i in range(0, size):
        total_value+=feature_matrix[i]
    #Calculate the mean
    mean = total_value/size
    inherent_dim = 0
    #Pick the eigen values that are greater than or equal to mean
    for i in range(0, len(feature_matrix)):
        if feature_matrix[i]>=mean:
            inherent_dim+=1
    return inherent_dim

#Function to caluclates the eigevn values for the feature space
def pca_helper(feature_space, action):
    #Choice of feature space
    Feature_Space_Map = {1: "Layer_3", 2: "Layer_4", 3: "AvgPool", 4: "BOF_HOG", 5: "BOF_HOF"}
    retrieval_query = f"SELECT {Feature_Space_Map[feature_space]} FROM data WHERE All_Label == '{action}';"
    c.execute(retrieval_query)
    rows = c.fetchall()
    #Retrieve all the features
    #Make a list of features matrix
    cleaned_data = []
    if feature_space in [1, 2, 3]:
        for row in rows:
            cleaned_data.append(list(map(float, row[0].strip("[]").split(","))))
    elif feature_space in [4, 5]:
        for row in rows:
            cleaned_data.append(list(map(int, row[0].strip("[]").split())))
    max_len = max(len(lst) for lst in cleaned_data)
    #Pad the data if necessary
    padded_data = [lst + [0] * (max_len - len(lst)) for lst in cleaned_data]
    data = np.array(padded_data)
    #Gather the eigen value list
    eigen_value_list = []
    cov_matrix = np.cov(data, rowvar=False)
    #Sort the eigen values and return them
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    eigen_value_list = sorted_eigenvalues
    ans = mean_eigen_value(eigen_value_list)
    
    print(f"The Inherent Dimensionality for {action} element is: ", ans)

#Main fucntion to gather the features from the user.
def main():
    feature_space = int(input("Select a Feature Space from the following: 1 - Layer3, 2 - Layer4, 3 - AvgPool, 4- HOG, 5 - HOF : "))
    #For each of the label gather the features
    for i in range(0, len(target_videos)):
       pca_helper(feature_space, target_videos[i])
    
main()
