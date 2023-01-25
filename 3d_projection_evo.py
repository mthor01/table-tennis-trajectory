# this script takes 2d ball positions and camera poses for every frame and predicts the 3d world coordinates of the ball
# with the middle of the table being the origin
# it is not finished and contains bugs

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import animation
import h5py
import os
from classes import simulation, population, camera

pose_dir = "pose_estimates"
trajectory_dir = "simulated_ball_data/positions"
bounce_dir = "simulated_ball_data/bounce_frames"
output_dir = "3d_ball_positions"

table_length = 2.74  # m
table_width = 1.525  # m
net_height = 0.15  # m


#converts positions in sim space to the desired system where the middle of the table is the origin
def convert_sim_data(positions, bounces, n_frames, poses):
    n_positions = len(positions)
    new_set_frames = [len(positions)] * int(round(n_frames/len(positions))+1)
    positions_2d = []
    positions_3d = []

    for i in range(len(new_set_frames)):
        new_set_frames[i]= new_set_frames[i]*(i+1)

    while n_frames > len(positions):
        positions = np.concatenate((positions, positions), 0)
        bounces = np.concatenate((bounces, [bounces[0]+n_positions, bounces[1]+n_positions]), 0)

    for i in range(n_frames):
        pos_3d = ((positions[i][0] - 2.74 / 2) * 100, (positions[i][1] - 1.525 / 2) * 100, positions[i][2] * 100)
        pos_2d, _ = cv.projectPoints(pos_3d, poses[i].rotation, poses[i].pos, poses[i].intrinsics, dist_coeffs)
        pos_3d = np.array(pos_3d)

        positions_2d.append(pos_2d[0][0])
        positions_3d.append(pos_3d)

    return positions_2d, bounces, new_set_frames, positions_3d

def get_event_frames(pos_list):
    bounce_indices = []
    racket_hit_indices = []

    for i in range(2,len(pos_list)):
        y_in_speed = np.linalg.norm(pos_list[i-1][1]-pos_list[i-2][1])
        y_out_speed = np.linalg.norm(pos_list[i][1]-pos_list[i-1][1])
        y_speed_change = max(y_in_speed,y_out_speed)/min(y_in_speed,y_out_speed)

        x_in_speed = np.linalg.norm(pos_list[i - 1][0] - pos_list[i - 2][0])
        x_out_speed = np.linalg.norm(pos_list[i][0] - pos_list[i - 1][0])
        x_speed_change = max(x_in_speed, x_out_speed) / min(x_in_speed, x_out_speed)

        y_sign_change = (pos_list[i-1][1]-pos_list[i-2][1] > 0) and (pos_list[i][1]-pos_list[i-1][1] < 0)
        x_sign_change = (pos_list[i-1][0]-pos_list[i-2][0] > 0) != (pos_list[i][0]-pos_list[i-1][0] > 0)

        if y_sign_change and not(x_sign_change):
            if pos_list[i-1][1] > pos_list[i][1]:
                bounce_indices.append(i-1)
            else:
                bounce_indices.append(i)
        elif x_sign_change or (x_speed_change>1.2):
            if pos_list[i-1][0] > pos_list[i][1]:
                racket_hit_indices.append(i-1)
            else:
                racket_hit_indices.append(i)

    return bounce_indices, racket_hit_indices


def display_animation(data):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    #plot wall for bounds of table
    xx, yy = np.meshgrid(np.linspace(0, table_length, 10), np.linspace(0, table_length, 10))
    z = np.ones(xx.shape) * table_width
    ax.plot_surface(xx, z, yy, alpha=0.5)

    #plot net
    zz, yy = np.meshgrid(np.linspace(0, table_width, 10), np.linspace(0, net_height, 10))
    x = np.ones(yy.shape) * table_length/2
    ax.plot_surface(x, zz, yy, alpha=0.5)

    # line generation
    def update(num, data, line):
        line.set_data(data[:2, :num])
        line.set_3d_properties(data[2, :num])

    line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])

    # Setting the axes properties
    ax.set_xlim3d([0, table_length])
    ax.set_xlabel('X')

    ax.set_ylim3d([0, table_length])
    ax.set_ylabel('Y')

    ax.set_zlim3d([0.0, table_length])
    ax.set_zlabel('Z')

    ani = animation.FuncAnimation(fig, update, N, fargs=(data, line), interval=1/90, blit=False)
    plt.show()

def create_dataset(dir, filename, data):
    f = h5py.File(dir + "/" + filename + ".hdf5", "w")
    dset = f.create_dataset(filename, data=data)

def clear_dir(path):
    for f in os.listdir(path):
        os.remove(path + "/" + f)

def simulate_trajectory(start_pos, speed, angular_velocity, framerate, num_frames):
    rotation_axis = angular_velocity / np.linalg.norm(angular_velocity)
    rpm = np.linalg.norm(angular_velocity) / (2 * math.pi /60)
    sim = simulation(start_pos, speed, rotation_axis, rpm)
    dt = 1/framerate * 10
    positions = []

    for i in range(num_frames*10):
        x, y, z, wall_hit = sim.step(dt)
        if i%10 == 0:
            positions.append([x,y,z])

    return positions


def project_2d(point, cam):
    point = (point[0], point[1], point[2])
    pos, _ = cv.projectPoints(point, cam.rotation, cam.pos, cam.intrinsics, dist_coeffs)
    return pos

# returns a list of frames where racket hits happen, based on if the velocity of the ball
# in the main line direction changes more than a given threshold
def spot_racket_hits(positions, poses):
    racket_hit_frames = []

    for i in range(3,len(positions)):
        table_orientation = project_2d([0,0,0], poses[i])[0][0] - project_2d([1,0,0], poses[i])[0][0]
        vel_1 = positions[i-2]-positions[i-1]
        vel_2 = positions[i-1]-positions[i]
        vel_1 = np.dot(vel_1, table_orientation) / np.linalg.norm(table_orientation)
        vel_2 = np.dot(vel_2, table_orientation) / np.linalg.norm(table_orientation)

        if (vel_1 - vel_2) > 0.5:
            racket_hit_frames.append(i-1)
    return racket_hit_frames

def test_population_performance(positions, velocities, angular_velocities, trajectory_2d):
    population_performances = []
    best_individual_performance = 100000

    for i in range(len(positions)):
        individual_positions = simulate_trajectory(positions[i], velocities[i], angular_velocities[i], 30, len(trajectory_2d))
        individual_errors = []

        for j in range(len(individual_positions)):
            pos_2d = project_2d(individual_positions[j], poses[last_hit + 1 + i])
            individual_errors.append(np.linalg.norm(pos_2d - trajectory_2d[j]))
        avg_performance = sum(individual_errors)/len(individual_errors)
        population_performances.append(avg_performance)

        if avg_performance < best_individual_performance:
            best_individual_performance = avg_performance
            best_individual_positions = individual_positions

    print(best_individual_performance)

    return population_performances, best_individual_positions

if __name__ == "__main__":
    for f in os.listdir(output_dir):
        os.remove(output_dir + "/" + f)
    dist_coeffs = np.zeros((4, 1))


    for vid in os.listdir(pose_dir):
        errors = []
        for trajectory in os.listdir(trajectory_dir):
            trajectory_errors = []
            pose_file= h5py.File(pose_dir + "/" + vid)
            a_group_key = list(pose_file.keys())[0]
            poses_ = pose_file[a_group_key][()]

            ballpos_file = h5py.File(trajectory_dir + "/" + trajectory)
            a_group_key = list(ballpos_file.keys())[0]
            ballpos_ = ballpos_file[a_group_key][()]

            bounce_frames_file = h5py.File(bounce_dir + "/" + trajectory)
            a_group_key = list(bounce_frames_file.keys())[0]
            bounces = bounce_frames_file[a_group_key][()]

            n_frames = len(poses_)
            poses = []
            ballpos = []
            bounce_pos = []
            final_wrld_pos = []
            vertical_cam_angles = []
            horizontal_cam_angles = []
            final_3d_pos = []
            final_candidates = []
            error_sum = 0
            error_denominator = 0
            NG = 300
            NH = 72




            #extract camera positions for every frame
            for i in range(n_frames):
                poses.append(camera([poses_[i][0], poses_[i][1], poses_[i][2]], [poses_[i][3], poses_[i][4], poses_[i][5]], poses_[i][6], poses_[i][7], poses_[i][8]))

            if len(ballpos_[0]) == 3:
                ballpos, bounces_, new_set_frames, test_positions = convert_sim_data(ballpos_, bounces, n_frames, poses)
            else:
                ballpos = ballpos_

            racket_hits = spot_racket_hits(ballpos, poses)

            last_hit = -1
            for hit_frame in racket_hits:
                trajectory_2d = ballpos[last_hit+1:hit_frame]
                pop = population()
                population_performances, _ = test_population_performance(pop.positions, pop.velocities, pop.angular_velocities, trajectory_2d)
                pop.performances = population_performances

                for i in range(NG):
                    pop.new_gen()
                    population_performances, best_pos_estimations = test_population_performance(pop.position_candidates, pop.velocity_candidates, pop.angular_velocity_candidates, trajectory_2d)
                    pop.update(population_performances)


                final_wrld_pos.append(best_pos_estimations)


                last_hit = hit_frame



