#Import all the libraries
import json
import numpy as np
#import the features from a JSON
with open('../database/total_target_features.json', 'r') as f:
    target_data = json.load(f)
with open('../database/category_map.json', 'r') as f:
    category_map = json.load(f)
#Define the target videos
target_videos = ['golf',  'shoot_ball', 'brush_hair', 'handstand', 'shoot_bow', 
                'cartwheel', 'hit', 'shoot_gun', 'hug', 'sit', 'catch', 
                'jump', 'situp', 'chew', 'kick', 'smile', 'clap', 'kick_ball', 'smoke',
                'climb', 'somersault', 'climb_stairs', 'laugh', 'stand']


feature_representation = {}

#Function that is used to calculate the mean
def feature_calculator(layer):
    feauter_np = np.array(layer)
    mean_feature = np.mean(feauter_np, axis=0)
    return mean_feature.tolist()
    
#For all the labels prepare a feature matrix
def feature_selection(action):
    layer_1=[]
    layer_2=[]
    layer_3=[]
    #For all the that has the label mapped use them to make a feature for each class label
    for videoname, category in category_map.items():
        if category==action:
            for video in target_data:
                if videoname in video:
                    a=[]
                    a.extend(video[videoname])
                    layer_1.append(a[0])
    #For a action gather the feature array respective to layer3
    total_layer_1 = feature_calculator(layer_1)
    for videoname, category in category_map.items():
        if category==action:
            for video in target_data:
                if videoname in video:
                    a=[]
                    a.extend(video[videoname])
                    layer_2.append(a[1])
    #For a action gather the feature array respective to layer4
    total_layer_2 = feature_calculator(layer_2)
    for videoname, category in category_map.items():
        if category==action:
            for video in target_data:
                if videoname in video:
                    a=[]
                    a.extend(video[videoname])
                    layer_3.append(a[2])
    #For a action gather the feature array respective to avgpool
    total_layer_3 = feature_calculator(layer_3)
    final_features =[total_layer_1, total_layer_2, total_layer_3]
    #Append it to the dictionary
    feature_representation[action]=final_features

#main function to loop over the target video types
def main():
    for action in target_videos:
        feature_selection(action)
    #After for all the videos it is calculated dump it to a json
    with open('../database/feature_label_representation.json', 'w') as f:
        json.dump(feature_representation,f)
        
main()  
