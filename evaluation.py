import os
import re
import h5py
from classes.camera import camera
import matplotlib.pyplot as plt
import numpy as np
from functions import cam_to_wrld
import math
from functions import calc_vec_angle
import matplotlib
import gi
gi.require_version('Gtk', '3.0')

matplotlib.use("WebAgg")

original_data_dir = "simulated_ball_data/positions"
estimated_pos_dir = "3d_ball_positions"
pose_dir = "pose_estimates"

def get_vid_name(path):
    name = ""
    for letter in path:
        if letter != "_":
            name += letter
        else:
            break
    return name

def get_data(path):
    file = h5py.File(path)
    a_group_key = list(file.keys())[0]
    data = file[a_group_key][()]

    return data


def calc_per_angle_avg_errors(errors, poses, angle):
    horizontal_angle_errors = np.zeros(9)
    vertical_angle_errors = np.zeros(9)
    vertical_angle_counter = np.zeros(9)
    horizontal_angle_counter = np.zeros(9)
    trajectory_angle_errors = np.zeros(9)
    trajectory_angle_counter = np.zeros(9)
    for i in range(len(errors)):
        if poses[i].valid:
            wrld_cam_vec = cam_to_wrld(poses[i], [0, 0, 1]) - cam_to_wrld(poses[i], [0, 0, 0])
            vertical_cam_angle = calc_vec_angle(wrld_cam_vec, [wrld_cam_vec[0], wrld_cam_vec[1], 0])
            horizontal_cam_angle = calc_vec_angle([wrld_cam_vec[0], wrld_cam_vec[1]], [0, 1])
            trajectory_to_cam_angle = calc_vec_angle([wrld_cam_vec[0], wrld_cam_vec[1]], [math.cos(float(angle)), math.sin(float(angle))])

            if trajectory_to_cam_angle > 90:
                trajectory_to_cam_angle = 180-trajectory_to_cam_angle

           # if trajectory_to_cam_angle

            if horizontal_cam_angle > 90:
                horizontal_cam_angle = 180-horizontal_cam_angle

            horizontal_angle_errors[abs(math.floor(horizontal_cam_angle/10))] += errors[i]
            horizontal_angle_counter[abs(math.floor(horizontal_cam_angle/10))] += 1
            vertical_angle_errors[abs(math.floor(vertical_cam_angle/10))] += errors[i]
            vertical_angle_counter[abs(math.floor(vertical_cam_angle/10))] += 1
            trajectory_angle_errors[abs(math.floor(trajectory_to_cam_angle / 10))] += errors[i]
            trajectory_angle_counter[abs(math.floor(trajectory_to_cam_angle / 10))] += 1

            if i == 409 or i == 529:
                print(vertical_cam_angle)
                print(horizontal_cam_angle)



    return horizontal_angle_errors, vertical_angle_errors, horizontal_angle_counter, vertical_angle_counter, trajectory_angle_errors, trajectory_angle_counter

def divide_arrays(arr_1, arr_2):
    output = []
    for i in range(len(arr_1)):
        if arr_2[i] != 0:
            output.append(arr_1[i]/arr_2[i])
        else:
            output.append(0)
    return output

def correct_positions(positions):
    for i in range(len(positions)):
        positions[i] = ((positions[i][0] - 2.74 / 2) * 100, (positions[i][1] - 1.525 / 2) * 100, positions[i][2] * 100)

    return positions

if __name__ == "__main__":
    per_vid_errors = {}
    total_horizontal_angle_errors = np.zeros(9)
    total_vertical_angle_errors = np.zeros(9)
    total_vertical_counter = np.zeros(9)
    total_horizontal_counter = np.zeros(9)
    total_trajectory_angle_errors = np.zeros(9)
    total_trajectory_angle_counter = np.zeros(9)

    for estimated_pos_file in os.listdir(estimated_pos_dir):
        poses = []
        trajectory_errors = []

        velocity, trajectory_angle = re.findall("\d+\.\d+", estimated_pos_file)[-2], re.findall("\d+\.\d+", estimated_pos_file)[-1]
        vid_name = get_vid_name(estimated_pos_file)

        estimated_pos = get_data(estimated_pos_dir + "/" + estimated_pos_file)
        org_pos = get_data(original_data_dir + "/" + estimated_pos_file[len(vid_name)+1 :])
        org_pos = correct_positions(org_pos)
        poses_ = get_data(pose_dir + "/" + vid_name + ".hdf5")

        for pose in poses_:
            poses.append(camera([pose[0], pose[1], pose[2]], [pose[3], pose[4], pose[5]], pose[6],pose[7], pose[8]))

        for i in range(len(estimated_pos)):
            trajectory_errors.append(np.linalg.norm(estimated_pos[i] - org_pos[i % len(org_pos)]))
        avg_error = sum(trajectory_errors)/len(trajectory_errors)

        if vid_name in per_vid_errors:
            per_vid_errors[vid_name] = [per_vid_errors[vid_name][0] + avg_error, per_vid_errors[vid_name][1] + 1]
        else:
            per_vid_errors[vid_name] = [avg_error, 1]

        horizontal_angle_errors, vertical_angle_errors, horizontal_counter, vertical_counter,traj_angle_errors, traj_angle_counter = calc_per_angle_avg_errors(trajectory_errors, poses, trajectory_angle)
        total_horizontal_angle_errors += horizontal_angle_errors
        total_horizontal_counter += horizontal_counter
        total_vertical_angle_errors += vertical_angle_errors
        total_vertical_counter += vertical_counter
        total_trajectory_angle_errors += traj_angle_errors
        total_trajectory_angle_counter += traj_angle_counter

    for vid in per_vid_errors:
        per_vid_errors[vid] = per_vid_errors[vid][0]/per_vid_errors[vid][1]

    total_horizontal_angle_errors = divide_arrays(total_horizontal_angle_errors, total_horizontal_counter)
    total_vertical_angle_errors = divide_arrays(total_vertical_angle_errors, total_vertical_counter)
    total_trajectory_angle_errors = divide_arrays(total_trajectory_angle_errors, total_trajectory_angle_counter)

  #  print(per_vid_errors)



    bar_x_axe = (np.array(range(9))+1) * 10
    fig1 = plt.figure()
    plt.bar(bar_x_axe, total_vertical_angle_errors)
    print(total_vertical_angle_errors)
    plt.ylabel("AVG Euclid Error")
    plt.xlabel("Angle between camera viewing direction and net")
    fig2 = plt.figure()
    plt.bar(bar_x_axe, total_horizontal_angle_errors)
    plt.ylabel("AVG Euclid Error")
    plt.xlabel("Smallest angle between camera viewing direction and table plane")
    fig3 = plt.figure()
    plt.bar(bar_x_axe, total_trajectory_angle_errors)
    plt.ylabel("AVG Euclid Error")
    plt.xlabel("Smallest angle between camera viewing direction and trajectory")
    plt.show()


