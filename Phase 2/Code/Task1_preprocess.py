import json
import numpy as np

with open('../database/total_target_features.json', 'r') as f:
    target_data = json.load(f)
with open('../database/category_map.json', 'r') as f:
    category_map = json.load(f)

target_videos = ['golf',  'shoot_ball', 'brush_hair', 'handstand', 'shoot_bow', 
                'cartwheel', 'hit', 'shoot_gun', 'hug', 'sit', 'catch', 
                'jump', 'situp', 'chew', 'kick', 'smile', 'clap', 'kick_ball', 'smoke',
                'climb', 'somersault', 'climb_stairs', 'laugh', 'stand']


feature_representation = {}

def feature_calculator(layer):
    feauter_np = np.array(layer)
    mean_feature = np.mean(feauter_np, axis=0)
    return mean_feature.tolist()
    

def feature_selection(action):
    layer_1=[]
    layer_2=[]
    layer_3=[]
    for videoname, category in category_map.items():
        if category==action:
            for video in target_data:
                if videoname in video:
                    a=[]
                    a.extend(video[videoname])
                    layer_1.append(a[0])
    total_layer_1 = feature_calculator(layer_1)
    for videoname, category in category_map.items():
        if category==action:
            for video in target_data:
                if videoname in video:
                    a=[]
                    a.extend(video[videoname])
                    layer_2.append(a[1])
    total_layer_2 = feature_calculator(layer_2)
    for videoname, category in category_map.items():
        if category==action:
            for video in target_data:
                if videoname in video:
                    a=[]
                    a.extend(video[videoname])
                    layer_3.append(a[2])
    total_layer_3 = feature_calculator(layer_3)
    final_features =[total_layer_1, total_layer_2, total_layer_3]
    feature_representation[action]=final_features


def main():
    for action in target_videos:
        feature_selection(action)
    with open('../database/feature_label_representation.json', 'w') as f:
        print("In the JSON File")
        json.dump(feature_representation,f)
        
main()
    

                
        
        