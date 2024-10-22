import json
import numpy as np
from sklearn.cluster import KMeans

#Define the kmeans clustering for the latent sematics "s" 
def kmeans_clustering(s, feature_model, target_data, videoID):
    total_features=[]
    #Extract the video name from the json
    for video in target_data:
        video_name =video.keys()
        for key, value in video.items():
            video_name=key
            layer_values=value
        #Check wehter the video is even or odd
        if videoID[video_name]%2==0:
            layer=[]
            layer.append(videoID[video_name])
            #If feature is layer3 append the layer_3 features
            if feature_model==1:
                layer.append(layer_values[0])
            #If feature is layer4 append the layer_4 features
            elif feature_model==2:
                layer.append(layer_values[1])
            #If feature is avgpool append the avgpool features
            else:
                layer.append(layer_values[2])
            total_features.append(layer)
    
    #Combine the entries with a video id and its values
    combined_data=[]
    for item in total_features:
        id=item[0]
        feature=item[1]
        combined_data.append([id]+feature)
    total_features=[]
    #Convert the matrix to the numpy array which includes both the videoID, features
    total_features = np.array(combined_data)
    #Before training using kmeans separate the videoID
    filtered_features = total_features[:,1:]
    kmeans  = KMeans(n_clusters=s, random_state=42)
    kmeans.fit(filtered_features)
    #Compute the Cluster Centres
    cluster_centre = kmeans.cluster_centers_
    #kmeans_labels = kmeans.labels_
    #return
    
    video_ids = total_features[:, 0].astype(int) 
    features = total_features[:, 1:] 
    
    cluster_distance = np.linalg.norm(features[:, np.newaxis] - cluster_centre, axis=2)
    
    cluster_centre_list = cluster_centre.tolist()
    weight_mapping = {
        "centroids": cluster_centre_list,
        "video_clusters": {}
    }
    #Check for mapping of the code
    for i, centre in enumerate(cluster_centre):
        cluster_distances = cluster_distance[:, i]
        video_pairs = sorted(
            zip(video_ids, cluster_distances),
            key=lambda x: x[1]
        )
        total_videos=[]
        for j, k in video_pairs:
            a={"videoID": int(j), "weight": float(k)}
            total_videos.append(a)
        total_videos.sort(key=lambda x: x["weight"], reverse=True)
        weight_mapping["video_clusters"][f"Cluster{i+1}"]=total_videos
    
    # TODO: move to Output folder
    with open(f'../database/video_ID-weight_files_{feature_model}.json', 'w') as f:
        json.dump(weight_mapping, f, indent=4)