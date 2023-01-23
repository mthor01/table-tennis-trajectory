import scenedetect
import os
import csv

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

video_dir = "single_test_vid"
scene_list_dir = "scene_lists"

for f in os.listdir(scene_list_dir):
    os.remove(scene_list_dir + "/" + f)

for vid in os.listdir(video_dir):
    find_scenes(video_dir, scene_list_dir, vid)
    break