import json
import cv2

def visualise(video_file):
    with open('../database/total_target_features.json', 'r') as f:
        target_data = json.load(f)
    with open('../database/total_non_target_features.json', 'r') as f:
        non_target_data = json.load(f)
    found=False
    for video in target_data:
        if video_file in video:
            video_path = f'../dataset/target_videos/{video_file}'
            found=True
            break
    if found ==False:
        for video in non_target_data:
            if video_file in video:
                video_path = f'../dataset/non_target_videos/{video_file}'
                found=True
                break
    print("The video path", video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Couldnot read the video")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Visualizing the video frame using OpenCV
        cv2.imshow("The captured frame is: ", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    # Release and destroy the windows
    cap.release()
    cv2.destroyAllWindows()

def euclidean(a, b):
    distance_res=0
    for i in range(0, len(a)):
        distance_res += (a[i] - b[i])**2
        return distance_res ** 0.5

def layer3_implementation(query_video, layer_number, l):
    found = False
    with open('../database/total_target_features.json', 'r') as f:
        target_data = json.load(f)
    with open('../database/total_non_target_features.json', 'r') as f:
        non_target_data = json.load(f)
        
    layer=[]
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
    res = []
    for i in target_data:
        for key, value in i.items():
            video_name=key
            curr_feature = value[layer_number-1]
            distance = euclidean(layer, curr_feature)
            res.append((distance, key))
    for i in non_target_data:
        for key, value in i.items():
            video_name=key
            curr_feature = value[layer_number-1]
            distance = euclidean(layer, curr_feature)
            res.append((distance, key))
    res.sort(key=lambda i:i[0])
    print("******The \"m\" most similar videos are: ********")
    video_name = []
    #for i in range(0,20):
    for i in range(0, l):
        print(f'{res[i][1]} : {res[i][0]}')
        video_name.append(res[i][1])
    for i in video_name:
        visualise(i)
        
    
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
        
    feature_space = int(input("Select a Feature Space from the following: 1 - Layer3, 2 - Layer4, 3 - AvgPool, 4- HOG, 5 - HOF, 6 - Color Histogram : "))
    m = int(input("Provide the value of m: "))
    if feature_space==1 or feature_space==2 or feature_space==3:
        layer3_implementation(video_name, feature_space, m)
    elif feature_space==4 or feature_space==5:
        print("Else part")
    else:
        print("Histograms")
main()