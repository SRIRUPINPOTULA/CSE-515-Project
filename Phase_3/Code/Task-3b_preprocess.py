import os
import json
import cv2

def extract_and_save_thumbnails(json_path, folder1, folder2, output_folder):
    with open(json_path, 'r') as file:
        video_map = json.load(file)

    os.makedirs(output_folder, exist_ok=True)

    for folder in [folder1, folder2]:
        for video_name in os.listdir(folder):
            video_path = os.path.join(folder, video_name)

            if video_name in video_map:
                video_id = video_map[video_name]
                thumbnail_path = os.path.join(output_folder, f"{video_id}.jpg")
                
                if not os.path.exists(thumbnail_path):
                    cap = cv2.VideoCapture(video_path)
                    success, frame = cap.read()
                    if success:
                        cv2.imwrite(thumbnail_path, frame)
                    cap.release()

    print("Thumbnails saved successfully.")

json_path = "videoID.json"
folder1 = "../target_videos"
folder2 = "../non-target_videos"
output_folder = "Database/thumbnails"

extract_and_save_thumbnails(json_path, folder1, folder2, output_folder)
