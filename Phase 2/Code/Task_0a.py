import os
import shutil
import json

# import SQL library
import sqlite3

# import the python libraries
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

# import functions to generate feature spaces
from Task_1 import layer3_feature
from Task_1 import layer4_feature
from Task_1 import avgpool_feature

target_videos = ['golf',  'shoot_ball', 'brush_hair', 'handstand', 'shoot_bow', 
                'cartwheel', 'hit', 'shoot_gun', 'hug', 'sit', 'catch', 
                'jump', 'situp', 'chew', 'kick', 'smile', 'clap', 'kick_ball', 'smoke',
                'climb', 'somersault', 'climb_stairs', 'laugh', 'stand']

# Connect to the database
connection = sqlite3.connect('../database/Phase_2.db')
c = connection.cursor()

# Create database Table
create_table = """CREATE TABLE IF NOT EXISTS data (
                    videoID INTEGER PRIMARY KEY,
                    Video_Name VARCHAR(250),
                    Layer_3 VARCHAR(4000),
                    Layer_4 VARCHAR(4000),
                    AvgPool VARCHAR(4000),
                    BOF_HOG VARCHAR(4000),
                    BOF_HOF VARCHAR(4000),
                    Action_Label VARCHAR(20));"""

# TODO: create table for P1_Task3
# create_color_hist_table = """CREATE TABLE FI NOT EXISTS color_hist_data ();"""

c.execute(create_table)
# c.execute(create_color_hist_table)


# Dictionary to map video name to videoID
videoID = {}

def create_video_id():
    counter = 0
    target_path = '../dataset/target_videos'
    non_target_path = '../dataset/non_target_videos'
    for file in os.listdir(target_path):
        videoID[file] = counter
        counter += 1
    for file in os.listdir(non_target_path):
        videoID[file] = counter
        counter += 1
    
    # JSON is used as a backup to data
    with open('../database/videoID.json', 'w') as f:
        json.dump(videoID, f, indent=4)


# Dictionary to map video name to Action
datamap = {}

def video_category_map():
    dataset = '../hmdb51_org'
    for action in os.listdir(dataset):
        action_path = os.path.join(dataset, action)
        if action in target_videos:
            for file in os.listdir(action_path):
                if videoID[file]%2 == 0:
                    a = {}
                    a[file] = action
                    datamap.update(a)
        elif action == '.DS_Store':
            continue
    
    # JSON is used as a backup to data
    with open('../database/category_map.json', 'w') as f:
        json.dump(datamap, f, indent=4)

def target_videos_features():
    target_path = '../dataset/target_videos'
    layer_output = []
    for target_videos in os.listdir(target_path):
        video_path = os.path.join(target_path, target_videos)

        a = {}
        features_video =[]

        layer_3 = layer3_feature(video_path)
        features_video.append(layer_3)
        layer_4 = layer4_feature(video_path)
        features_video.append(layer_4)
        avgPool = avgpool_feature(video_path)
        features_video.append(avgPool)
        # TODO: Add Task2 Features
        # TODO: Add Task3 Features

        a[target_videos] = features_video
        layer_output.append(a)

        if videoID[target_videos]%2 == 0:
            video_insert = f"INSERT INTO data VALUES({videoID[target_videos]}, '{target_videos}', '{layer_3}', '{layer_4}', '{avgPool}', '[]', '[]', '{datamap[target_videos]}');"
        else:
            video_insert = f"INSERT INTO data VALUES({videoID[target_videos]}, '{target_videos}', '{layer_3}', '{layer_4}', '{avgPool}', '[]', '[]', NULL);"
        c.execute(video_insert)
    
    with open('../database/total_target_features.json', 'w') as f:
        json.dump(layer_output, f, indent=4)

def non_target_videos_features():
    non_target_path = '../dataset/non_target_videos'
    layer_output = []
    for non_target_videos in os.listdir(non_target_path):
        video_path = os.path.join(non_target_path, non_target_videos)

        a = {}
        features_video =[]

        layer_3 = layer3_feature(video_path)
        features_video.append(layer_3)
        layer_4 = layer4_feature(video_path)
        features_video.append(layer_4)
        avgPool = avgpool_feature(video_path)
        features_video.append(avgPool)
        # TODO: Add Task2 Features
        # TODO: Add Task3 Features

        a[target_videos] = features_video
        layer_output.append(a)

        video_insert = f"INSERT INTO data VALUES({videoID[target_videos]}, '{target_videos}', '{layer_3}', '{layer_4}', '{avgPool}', '[]', '[]', NULL);"
        c.execute(video_insert)

    with open('../database/total_non_target_features.json', 'w') as f:
        json.dump(layer_output, f, indent=4)


# Move all videos from 'hmdb51_org' to 'dataset/target_videos' and 'dataset/non_target_videos'
def move_target_videos():
    path = '../hmdb51_org'

    target_path = os.listdir('../dataset/target_videos')
    for file in target_path:
        full_path = os.path.join('../dataset/target_videos', file)
        if os.path.isfile(full_path):
            os.remove(full_path)

    non_target_path = os.listdir('../dataset/non_target_videos')
    for file in non_target_path:
        full_path = os.path.join('../dataset/non_target_videos', file)
        if os.path.isfile(full_path):
            os.remove(full_path)

    for video_dir in os.listdir(path):
        if video_dir in target_videos:
            video_path = f'{path}/{video_dir}'
            destination_dir = '../dataset/target_videos'
            for all_videos in os.listdir(video_path):
                file_name = os.path.join(video_path, all_videos)
                shutil.copy(file_name, destination_dir)
        elif video_dir == '.DS_Store':
            continue
        else:
            video_path = f'{path}/{video_dir}'
            destination_dir = '../dataset/non_target_videos'
            for all_videos in os.listdir(video_path):
                file_name = os.path.join(video_path, all_videos)
                shutil.copy(file_name, destination_dir)

    create_video_id()
    video_category_map()
    target_videos_features()
    non_target_videos_features()


move_target_videos()

# commit the changes
connection.commit()

# close the Database connection
connection.close()
