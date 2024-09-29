import os
import shutil
import json
#import the libraries
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

from task_1 import layer3_feature
from task_1 import layer4_feature
from task_1 import avgpool_feature

target_videos = ['golf',  'shoot_ball', 'brush_hair', 'handstand', 'shoot_bow', 
                'cartwheel', 'hit', 'shoot_gun', 'hug', 'sit', 'catch', 
                'jump', 'situp', 'chew', 'kick', 'smile', 'clap', 'kick_ball', 'smoke',
                'climb', 'somersault', 'climb_stairs', 'laugh', 'stand']

videoID = {}

def create_video_id():
    counter=0
    target_path = './target_videos'
    non_target_path = './non_target_videos'
    for file in os.listdir(target_path):
        videoID[file] = counter
        counter+=1
    for file in os.listdir(non_target_path):
        videoID[file] = counter
        counter+=1

datamap = {}

def video_category_map():
    dataset = './hmdb51_org-1'
    for action in os.listdir(dataset):
        action_path = os.path.join(dataset, action)
        if action == '.DS_Store':
            continue
        else:
            for file in os.listdir(action_path):
                a = {}
                a[file]=action
                datamap.update(a)
    with open('./database/category_map.json', 'w') as f:
        json.dump(datamap,f)

def target_videos_features():
    target_path = './target_videos'
    layer_output = []
    for target_videos in os.listdir(target_path):
        video_path = os.path.join(target_path, target_videos)
        if videoID[target_videos]%2==0:
            a = {}
            features_video =[]
            features_video.append(layer3_feature(video_path))
            features_video.append(layer4_feature(video_path))
            features_video.append(avgpool_feature(video_path))
            features_video.append(datamap[target_videos])
            #Add Task2 Features
            #Add Task3 Features
            a[target_videos] = features_video
            layer_output.append(a)
            break
    with open('./database/target_features.json', 'w') as f:
        json.dump(layer_output,f)

def target_odd_videos_features():
    target_path = './target_videos'
    layer_output = []
    for target_videos in os.listdir(target_path):
        video_path = os.path.join(target_path, target_videos)
        if videoID[target_videos]%2!=0:
            a = {}
            features_video =[]
            features_video.append(layer3_feature(video_path))
            features_video.append(layer4_feature(video_path))
            features_video.append(avgpool_feature(video_path))
            #Add Task2 Features
            #Add Task3 Features
            a[target_videos] = features_video
            layer_output.append(a)
   
    with open('./database/target_odd_features.json', 'w') as f:
        json.dump(layer_output,f, indent=4)
        
def non_target_odd_videos_features():
    non_target_path = './non_target_videos'
    layer_output = []
    for non_target_videos in os.listdir(non_target_path):
        video_path = os.path.join(non_target_path, non_target_videos)
        a = {}
        features_video =[]
        features_video.append(layer3_feature(video_path))
        features_video.append(layer4_feature(video_path))
        features_video.append(avgpool_feature(video_path))
        #Add Task2 Features
        #Add Task3 Features
        a[target_videos] = features_video
        layer_output.append(a)
   
    with open('./database/non_target_odd_features.json', 'w') as f:
        json.dump(layer_output,f, indent=4)


def move_target_videos():
    #Move all the target videos to target_videos
    #print(len(target_videos))
    path = './hmdb51_org-1'
    target_path = os.listdir('./target_videos')
    for file in target_path:
        total_path = os.path.join('./target_videos', file)
        if os.path.isfile(total_path):
            os.remove(total_path)
    non_target_path = os.listdir('./non_target_videos')
    for file in non_target_path:
        total_path = os.path.join('./non_target_videos', file)
        if os.path.isfile(total_path):
            os.remove(total_path)
    for video in os.listdir(path):
        if video in target_videos:
            video_path = f'{path}/{video}'
            destination_dir = './target_videos'
            for all_videos in os.listdir(video_path):
                file_name = os.path.join(video_path, all_videos)
                shutil.copy(file_name, destination_dir)
        elif video=='.DS_Store':
            continue
        else:
            video_path = f'{path}/{video}'
            destination_dir = './non_target_videos'
            for all_videos in os.listdir(video_path):
                file_name = os.path.join(video_path, all_videos)
                shutil.copy(file_name, destination_dir)   
    
    create_video_id()
    video_category_map()
    target_videos_features()
    #target_odd_videos_features()
    #non_target_odd_videos_features()
move_target_videos()
        