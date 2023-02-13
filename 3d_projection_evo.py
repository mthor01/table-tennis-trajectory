"""
This script takes 2d ball positions and camera poses for every frame and predicts the 3d world coordinates of
the ball, with the middle of the table being the origin.
It is not finished and contains bugs
It also takes too long to compute, as i have not parallelised the Simulations of one population yet
That is why i set NG and NH very low
"""

import cv2 as cv
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib import animation
import h5py
from classes import Population, Camera, SimulationParallel
import time
from tqdm import tqdm
from pathlib import Path
from functions import clear_dir, cam_to_wrld
import matplotlib
matplotlib.use("WebAgg")

POSE_DIR = Path("pose_estimates")
TRAJECTORY_DIR = Path("simulated_ball_data/positions")
BOUNCE_FRAMES_DIR = Path("simulated_ball_data/bounce_frames")
HIT_FRAMES_DIR = Path("simulated_ball_data/hit_frames")
OUTPUT_DIR = Path("3d_ball_positions")

TABLE_LENGTH = 2.74  # m
TABLE_WIDTH = 1.525  # m
NET_HEIGHT = 0.15  # m

NG = 300
NH = 100

def world_to_sim(positions):
    for i in range(len(positions)):
        positions[i] = np.array([(positions[i][0] + 274 / 2) / 100, (positions[i][1] + 152.5 / 2) / 100, positions[i][2] / 100])

    return positions

#converts positions in sim space to the desired system where the middle of the table is the origin
def convert_sim_data(positions, n_frames, poses):
    new_set_frames = [len(positions)] * int(round(n_frames/len(positions))+1)
    positions_2d = []
    positions_3d = []

    for i in range(len(new_set_frames)):
        new_set_frames[i]= new_set_frames[i]*(i+1)

    while n_frames > len(positions):
        positions = np.concatenate((positions, positions), 0)
  #      bounces = np.concatenate((bounces, bounces + len(bounces)/n_bounces*n_positions), 0)
   #     bounces = bounces.astype("int")


    for i in range(n_frames):
        pos_3d = ((positions[i][0] - 2.74 / 2) * 100, (positions[i][1] - 1.525 / 2) * 100, positions[i][2] * 100)
        pos_2d, _ = cv.projectPoints(pos_3d, poses[i].rotation, poses[i].pos, poses[i].intrinsics, dist_coeffs)
        pos_3d = np.array(pos_3d)

        positions_2d.append(pos_2d[0][0])
        positions_3d.append(pos_3d)

    return positions_2d, new_set_frames, positions_3d

def get_event_frames(pos_list):
    bounce_indices = []
    racket_hit_indices = []

    for i in range(2,len(pos_list)):
        y_in_speed = norm(pos_list[i-1][1]-pos_list[i-2][1])
        y_out_speed = norm(pos_list[i][1]-pos_list[i-1][1])
        y_speed_change = max(y_in_speed,y_out_speed)/min(y_in_speed,y_out_speed)

        x_in_speed = norm(pos_list[i - 1][0] - pos_list[i - 2][0])
        x_out_speed = norm(pos_list[i][0] - pos_list[i - 1][0])
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
    xx, yy = np.meshgrid(np.linspace(0, TABLE_LENGTH, 10), np.linspace(0, TABLE_LENGTH, 10))
    z = np.ones(xx.shape) * TABLE_WIDTH
    ax.plot_surface(xx, z, yy, alpha=0.5)

    #plot net
    zz, yy = np.meshgrid(np.linspace(0, TABLE_WIDTH, 10), np.linspace(0, NET_HEIGHT, 10))
    x = np.ones(yy.shape) * TABLE_LENGTH/2
    ax.plot_surface(x, zz, yy, alpha=0.5)

    # line generation
    def update(num, data, line):
        line.set_data(data[:2, :num])
        line.set_3d_properties(data[2, :num])

    line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])

    # Setting the axes properties
    ax.set_xlim3d([0, TABLE_LENGTH])
    ax.set_xlabel('X')

    ax.set_ylim3d([0, TABLE_LENGTH])
    ax.set_ylabel('Y')

    ax.set_zlim3d([0.0, TABLE_LENGTH])
    ax.set_zlabel('Z')

    ani = animation.FuncAnimation(fig, update, 200, fargs=(data, line), interval=1/90, blit=False)
    plt.show()

def create_dataset(dir, filename, data):
    f = h5py.File(dir + "/" + filename + ".hdf5", "w")
    dset = f.create_dataset(filename, data=data)


#
def simulate_trajectories(positions, velocities, angular_velocities, framerate, num_frames):
    sim = SimulationParallel(positions.copy(), velocities.copy(), angular_velocities.copy())
    dt = 1/(framerate * 3)
    output_positions = []
    pos = positions.copy()
    for i in range(num_frames*3):
        if i%3 == 0:
            output_positions.append(pos.copy())
        pos, wall_hit = sim.step(dt)

    return output_positions


def project_2d(point, cam):
    point = (point[0], point[1], point[2])
    pos, _ = cv.projectPoints(point, cam.rotation, cam.pos, cam.intrinsics, dist_coeffs)
    return pos

# returns a list of frames where racket hits happen, based on if the velocity of the ball
# in the main line direction changes more than a given threshold
def spot_racket_hits(positions, poses):
    racket_hit_frames = []
    time_since_last_hit = 100

    for i in range(3,len(positions)):
        table_orientation = project_2d([0,0,0], poses[i])[0][0] - project_2d([1,0,0], poses[i])[0][0]
        vel_1 = positions[i-2]-positions[i-1]
        vel_2 = positions[i-1]-positions[i]
        speed_1 = (vel_1 @ table_orientation) / norm(table_orientation)
        speed_2 = (vel_2 @ table_orientation) / norm(table_orientation)
        time_since_last_hit +=1

        if ((speed_1 * speed_2) < 0) and (abs(speed_1-speed_2) > 2) and (vel_1[1] > 0) and time_since_last_hit > 5:
            racket_hit_frames.append(i)
            time_since_last_hit = 0
    return racket_hit_frames

def test_population_performance(positions, velocities, angular_velocities, trajectory_2d, test_pos):
    population_performances = []
    best_individual_performance = 100000


    simulated_positions = simulate_trajectories(positions, velocities, angular_velocities, 30, len(trajectory_2d))

    for i in range(len(positions)):
        individual_positions = []
        individual_errors = []

        for j in range(len(simulated_positions)):
            individual_positions.append(simulated_positions[j][i])


        individual_positions_2d, _, individual_positions_3d = convert_sim_data(individual_positions, len(individual_positions), poses[last_hit+1:last_hit+1+len(simulated_positions)])

        for j in range(len(individual_positions)):
            #pos_2d = project_2d(individual_positions[j], poses[last_hit + 1 + j])
            pos_2d = individual_positions_2d[j]
            individual_errors.append(norm(pos_2d - trajectory_2d[j]))
        avg_performance = sum(individual_errors) / len(individual_errors)
        population_performances.append(avg_performance)
        if avg_performance < best_individual_performance:
            best_individual_performance = avg_performance
            best_individual_positions = individual_positions_3d
            best_individual_positions_2d = individual_positions_2d


        pos_2d_list = []
        for i in range(len(best_individual_positions)):
            pos_2d_list.append(project_2d(individual_positions[i], poses[last_hit + 1 + i]))



   # print(best_individual_performance)

   # test_3d_to_2d("/home/mthor9/Schreibtisch/uni/bachelorarbeit/table_tennis_code/table-tennis-trajectory/single_test_vid/test1.mp4", [best_individual_positions_2d[0]], 1)

    return population_performances, best_individual_positions, best_individual_positions_2d

def test_3d_to_2d(video_path, point_list, n):
    video = cv.VideoCapture(video_path)
    success, img = video.read()
    for i in range(n):
        img = cv.circle(img, (int(point_list[i][0]), int(point_list[i][1])), 3, (0, 0, 255), -1)
    cv.imshow("img", img)
    cv.waitKey(0)

def possible_start_positions(point_2D, cam):
    ray_point = np.matmul(np.linalg.inv(cam.intrinsics), np.array([point_2D[0], point_2D[1], 1]))
    camera_pos = [0, 0, 0]
    ray_point = cam_to_wrld(cam, ray_point)
    camera_pos = cam_to_wrld(cam, camera_pos)
    camera_pos = world_to_sim([camera_pos])[0]
    ray_point = world_to_sim([ray_point])[0]
    ray_vec = (ray_point-camera_pos)/np.linalg.norm(ray_point-camera_pos) * np.linalg.norm(cam.pos)/100 * 2

    return camera_pos, ray_vec


if __name__ == "__main__":
    clear_dir(OUTPUT_DIR)

    dist_coeffs = np.zeros((4, 1))


    for vid in tqdm(POSE_DIR.glob("*")):
        errors = []
        for trajectory in tqdm(TRAJECTORY_DIR.glob("*")):
            trajectory_errors = []
            pose_file= h5py.File(str(vid))
            a_group_key = list(pose_file.keys())[0]
            poses_ = pose_file[a_group_key][()]

            ballpos_file = h5py.File(trajectory)
            a_group_key = list(ballpos_file.keys())[0]
            ballpos_ = ballpos_file[a_group_key][()]

            bounce_frames_file = h5py.File(str(Path(BOUNCE_FRAMES_DIR, trajectory.name)))
            a_group_key = list(bounce_frames_file.keys())[0]
            bounces = bounce_frames_file[a_group_key][()]

            hit_frames_file = h5py.File(str(Path(HIT_FRAMES_DIR, trajectory.name)))
            a_group_key = list(hit_frames_file.keys())[0]
            hit_frames = hit_frames_file[a_group_key][()]

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

            #extract camera positions for every frame
            for i in range(n_frames):
                poses.append(Camera([poses_[i][0], poses_[i][1], poses_[i][2]], [poses_[i][3], poses_[i][4], poses_[i][5]], poses_[i][6], poses_[i][7], poses_[i][8]))

            if len(ballpos_[0]) == 3:
                ballpos, new_set_frames, test_positions = convert_sim_data(ballpos_, n_frames, poses)
            else:
                ballpos = ballpos_

            racket_hits = spot_racket_hits(ballpos, poses)

           # display_animation(np.transpose(ballpos_))

           # test_3d_to_2d("/home/mthor9/Schreibtisch/uni/bachelorarbeit/table_tennis_code/table-tennis-trajectory/single_test_vid/test1.mp4", ballpos, 100)

            last_hit = -1
            start_time = time.time()
            for hit_frame in tqdm([racket_hits[0]]):
                #start_time = time.time()
                trajectory_2d = ballpos[last_hit+1:hit_frame]
               # test_3d_to_2d("/home/mthor9/Schreibtisch/uni/bachelorarbeit/table_tennis_code/table-tennis-trajectory/single_test_vid/test1.mp4",trajectory_2d, len(trajectory_2d))
                if len(trajectory_2d) > 0:
                    cam_pos_wrld, start_ray_vec = possible_start_positions(trajectory_2d[0], poses[last_hit+1])
                    pop = Population(cam_pos_wrld, start_ray_vec, NH=NH)
                    population_performances, _, _ = test_population_performance(pop.positions, pop.velocities, pop.angular_velocities, trajectory_2d, test_positions)
                    pop.performances = population_performances

                    times = []

                    for i in range(NG):

                        pop.new_gen()
                      #  start_time = time.time()
                        population_performances, _, _ = test_population_performance(pop.position_candidates, pop.velocity_candidates, pop.angular_velocity_candidates, trajectory_2d, test_positions)
                       # times.append(time.time()-start_time)
                        pop.update(population_performances)



                    last_hit = hit_frame

                    population_performances, best_pos_estimations,estimations_2d = test_population_performance(pop.positions, pop.velocities,
                                                                             pop.angular_velocities, trajectory_2d,
                                                                             test_positions)

                #    test_3d_to_2d(
                 #       "/home/mthor9/Schreibtisch/uni/bachelorarbeit/table_tennis_code/table-tennis-trajectory/single_test_vid/test1.mp4",
                  #      estimations_2d, len(estimations_2d))

                    for i in range(len(best_pos_estimations)):
                        errors.append(np.linalg.norm(best_pos_estimations[i]- test_positions[i]))
                    print(sum(errors)/len(errors))

                   # display_animation(np.transpose(world_to_sim(best_pos_estimations)))

                    if final_wrld_pos == []:
                        final_wrld_pos = best_pos_estimations
                    else:
                        final_wrld_pos = np.concatenate((final_wrld_pos, best_pos_estimations), 0)



        f = h5py.File(str(Path(OUTPUT_DIR, vid.stem)) + "_" + trajectory.stem, "w")
        dset = f.create_dataset(str(vid.stem) + "_" + trajectory.stem + ".hdf5", data=final_wrld_pos)



