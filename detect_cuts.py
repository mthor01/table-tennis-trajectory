"""
This script detects cuts in videos stored in a defined video directory.
The frames in which cuts happen are stored in a scene frame directory.
Executing this script will also clean any previously stored data in the scene list directory.
"""

import scenedetect
from pathlib import Path
from scenedetect import SceneManager, open_video, ContentDetector

VIDEO_DIR = Path("single_test_vid")
SCENE_FRAMES_DIR = Path("scene_lists")

THRESHOLD = 8.0
MIN_SCENE_LENGTH = 0

if __name__ == "__main__":
    for f in SCENE_FRAMES_DIR.glob("*"):
        Path(f).unlink()

    for video_name in VIDEO_DIR.glob("*.mp4"):
        video = open_video(str(video_name))
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=THRESHOLD, min_scene_len=MIN_SCENE_LENGTH))
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()
        with open(str(Path(SCENE_FRAMES_DIR, video_name.stem)) + ".csv", "wt") as scene_list_file:
            scenedetect.scene_manager.write_scene_list(scene_list_file, scene_list)