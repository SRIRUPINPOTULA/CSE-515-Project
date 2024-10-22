import json
import numpy as np

import sqlite3

from Util.PCA import PCA
from Util.SVD import SVD
from Util.LDA import LDA
from Util.KMeans import kmeans_clustering

with open('../database/total_target_features.json', 'r') as f:
    target_data = json.load(f)
with open('../database/videoID.json', 'r') as f:
    videoID = json.load(f)

feature_space = int(input("Select a Feature Space: 1 - Layer3, 2 - Layer4, 3 - AvgPool, 4- HOG, 5 - HOF, 6 - Color Histogram : "))
latent_count = int(input("Select a value to pick top-s results: "))
DR_method = int(input("Select a Dimensionality Reduction technique: 1 - PCA, 2 - SVD, 3 - LDA, 4 - K-means : "))

Feature_Space_Map = {1: "Layer_3", 2: "Layer_4", 3: "AvgPool", 4: "BOF_HOG", 5: "BOF_HOF"}

# Get the Selected Video-Feature Matrix
connection = sqlite3.connect('../database/Phase_2.db')
c = connection.cursor()

if feature_space != 6:
    retrieval_query = f"SELECT {Feature_Space_Map[feature_space]} FROM data WHERE videoID % 2 == 0 AND Action_Label NOT NULL LIMIT 5;"
else:
    retrieval_query = ""

c.execute(retrieval_query)
rows = c.fetchall()

cleaned_data = []
if feature_space in [1, 2, 3]:
    for row in rows:
        cleaned_data.append(list(map(float, row[1].strip("[]").split(","))))
elif feature_space in [4, 5]:
    for row in rows:
        cleaned_data.append(list(map(int, row[1].strip("[]").split())))

data = np.array(cleaned_data)

if DR_method == 1:
    PCA(data, latent_count)

elif DR_method == 2:
    SVD(data, latent_count)

elif DR_method == 3:
    print("LDA")

elif DR_method == 4:
    kmeans_clustering(latent_count, feature_space, target_data, videoID)

else:
    print("Invalid Dimensionality Reduction technique selected.")


connection.close()
