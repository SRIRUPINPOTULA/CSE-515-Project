#Task 1: Implement a program which, given (a) a query video file name
#or videoID (even or odd, target or non target), (b) a user selected feature
#space, and (c) positive integer l, identifies and lists l most likely matching
#labels, along with their scores, under the selected feature space.

import json
import cv2


 

def cosine_similarity(a, b):
    dot_sum=0
    list_a=0
    list_b=0
    for i in range(0,512):
        list_a += a[i]**2
    list_a=list_a ** 0.5
    for i in range(0,512):
        list_b += b[i]**2
    list_b = list_b ** 0.5
    for i in range(0,512):
        dot_sum += a[i] * b[i]
    final_ans= dot_sum/(list_a * list_b)
    return final_ans
        
def nearest_search(query_video, layer_number, l):
    with open('../database/total_target_features.json', 'r') as f:
        target_data = json.load(f)
    with open('../database/total_non_target_features.json', 'r') as f:
        non_target_data = json.load(f)
    with open('../database/feature_label_representation.json', 'r') as f:
        features_extracted = json.load(f)
    layer=[]
    found=False
    for video in target_data:
        if query_video in video:
            layer.extend(video[query_video])
            found=True
            break
    if found ==False:
        for video in non_target_data:
            if query_video in video:
                layer.extend(video[query_video])
                found=True
                break
    if layer_number==1:
        layer = layer[0]
    elif layer_number==2:
        layer = layer[1]
    else:
        layer=layer[2]   
    res=[]
    for key, value in features_extracted.items():
        distance = cosine_similarity(layer, value[layer_number-1])
        res.append((distance, key))
    res.sort(key=lambda i:i[0], reverse=True)
    print("******The \"m\" most similar labels for the video are: ********")
    for i in range(0, l):
        #for i in range(0,len(res)):
        print(f'{res[i][1]} : {res[i][0]}')

#Phase-2/dataset/target_videos/-_FREE_HUGS_-_Abrazos_Gratis_www_abrazosgratis_org_hug_u_cm_np2_ba_goo_2.avi  

def main():
    input_type = int(input("Provide the 1 - Video File Name or 2 - VideoID: "))
    if input_type==1:
        video_name = input("Provide the  Video File Name: ")
    else:
        video_number = int(input("Provide the  VideoID: "))
        with open('../database/videoID.json', 'r') as f:
            videoID = json.load(f)
        for key, value in videoID.items():
            if value==video_number:
                video_name=key
                break
    print("The Video name: ", video_name)   
    feature_space = int(input("Select a Feature Space from the following: 1 - Layer3, 2 - Layer4, 3 - AvgPool, 4- HOG, 5 - HOF, 6 - Color Histogram : "))
    m = int(input("Provide the value of m: "))
    if feature_space==1 or feature_space==2 or feature_space==3:
        nearest_search(video_name, feature_space, m)
    elif feature_space==4 or feature_space==5:
        print("Else part")
    else:
        print("Histograms")
main()

        