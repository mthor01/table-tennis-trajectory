import cv2 as cv
import numpy as np
import scenedetect
import os

from scenedetect import SceneManager, open_video, ContentDetector

def find_scenes(video_dir, scene_list_dir, video_name, threshold=8.0):
    video = open_video(video_dir + "/" + video_name)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold, min_scene_len=0))
    # Detect all scenes in video from current position to end.
    scene_manager.detect_scenes(video)
    # `get_scene_list` returns a list of start/end timecode pairs
    # for each scene that was found.
    scene_list = scene_manager.get_scene_list()
    with open(scene_list_dir + "/" + video_name[:-4] + ".csv", "wt") as scene_list_file:
        scenedetect.scene_manager.write_scene_list(scene_list_file, scene_list)

#find_scenes("C:/Users/mthor/Documents/uni/semester7/bachelorarbeit/table_tennis_project/test_videos/4_test.mp4")

video_dir = "single_test_vid"
scene_list_dir = "scene_lists"

for vid in os.listdir(video_dir):
    find_scenes(video_dir, scene_list_dir, vid)