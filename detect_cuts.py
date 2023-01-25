#this script detects cuts in videos stored in a defined video directory
#the frames on which cuts happen are stored in a scene list directory
#executing this script will also clean any previously stored data in the scene list directory

import scenedetect
import os
from scenedetect import SceneManager, open_video, ContentDetector

video_dir = "single_test_vid"
scene_list_dir = "scene_lists"

threshold = 8.0
min_scene_length = 0

if __name__ == "__main__":
    for f in os.listdir(scene_list_dir):
        os.remove(scene_list_dir + "/" + f)

    for video_name in os.listdir(video_dir):
        video = open_video(video_dir + "/" + video_name)
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(threshold=threshold, min_scene_len=min_scene_length))
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()
        with open(scene_list_dir + "/" + video_name[:-4] + ".csv", "wt") as scene_list_file:
            scenedetect.scene_manager.write_scene_list(scene_list_file, scene_list)