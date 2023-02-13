#this script simulates a bunch of table tennis trajectories with varying start postition, velocity, and angular velocity

import numpy as np
from numpy.linalg import norm
from numpy import array
import matplotlib
import matplotlib.pyplot as plt
from math import pi, cos, sin
from matplotlib import animation
import h5py
from classes import SimulationParallel
from tqdm.contrib import itertools
from pathlib import Path
from functions import clear_dir

matplotlib.use("WebAgg")

BOUNCE_FRAMES_DIR = Path("simulated_ball_data", "bounce_frames")
POSITIONS_DIR = Path("simulated_ball_data", "positions")
RACKET_HIT_FRAMES_DIR = Path("simulated_ball_data", "hit_frames")

TABLE_LENGTH = 2.74  # m
TABLE_WIDTH = 1.525  # m
NET_HEIGHT = 0.15 # m

TEST_SPEEDS = array([5])
TEST_ANGLES = array([-40, -30, -20, -10, 0, 10, 20, 30, 40]) * pi / 180
START_POSITIONS_Y = np.linspace(0.02, TABLE_WIDTH-0.02, 5)
#START_POSITIONS_Z = [0.6]
#TEST_SPEEDS = array([5.])
#TEST_ANGLES = array([0])
TEST_Z_SPEEDS = array([-2])
START_POSITIONS_X = array([0.02])
#START_POSITIONS_Y = array([TABLE_WIDTH/2])
START_POSITIONS_Z = array([0.5])


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

    ani = animation.FuncAnimation(fig, update, N, fargs=(data, line), interval=1000/30, blit=False)
    plt.show()

def create_dataset(dir, filename, data):
    f = h5py.File(dir + "/" + filename + ".hdf5", "w")
    f.create_dataset(filename, data=data)


if __name__ == "__main__":
   # sim = SimulationParallel()
    dt = 1/90
    num_samples = 0


    clear_dir(BOUNCE_FRAMES_DIR)
    clear_dir(POSITIONS_DIR)

    for values in itertools.product(TEST_SPEEDS, TEST_ANGLES,START_POSITIONS_X, START_POSITIONS_Y, START_POSITIONS_Z, TEST_Z_SPEEDS):
        speed = values[0]
        angle = values[1]
        X0 = values[2]
        Y0 = values[3]
        Z0 = values[4]
        z_speed = values[5]
        rotation_axis = [0,0,0]
        rpm = 0
       # speed, angle, Y0, Z0, rotation_axis, rpm = 3, 0, TABLE_WIDTH/2, 0.5, [0,0,0], 0

        vel = [cos(angle), sin(angle), 0]
        vel =  vel / norm(vel) * speed
        vel = [vel[0], vel[1], z_speed]
        angular_velocities = [[0,0,0]]


        sim = SimulationParallel([[X0, Y0, Z0]], [vel], angular_velocities)

        bounce_counter = 0
        hit_counter = 0
        i = 0
        j = 0
        last_ball_z_vel = sim.velocities[0][2]
        last_ball_x_vel = sim.velocities[0][0]
        data = []
        bounces = []
        hits = []
        positions = sim.positions

        # execute the Simulation for given values and store the ball positions
        # it stops after 2 bounces
        # only stores if the Simulation did not interrupt the loop with break_loop
        while i<500:
            i += 1
            break_loop = False
            data.append(positions[0])
            for _ in range(3):
                positions, wall_hit = sim.step(dt)
                if wall_hit[0]:
                    break_loop = True


            #data.append(positions[0])
            N = i

            if (last_ball_z_vel < 0) and (sim.velocities[0][2]>0):
                bounce_counter += 1
                bounces.append(i)


          #  if (last_ball_x_vel*sim.velocities[0][0] < 0):
           #     hits.append(i)
            #    hit_counter+=1
             #   data.pop()
            if wall_hit[0]:
                break

            last_ball_z_vel = sim.velocities[0][2]
       #     last_ball_x_vel = sim.velocity[0]

            if hit_counter >= 2:
                j+=1
                break

            if (bounce_counter >= 3) or break_loop:
               break

        #if len(data) >= 30:
           # display_animation(np.transpose(data))



        if N > 1:#break_loop == False:
            num_samples+=1
            create_dataset(str(BOUNCE_FRAMES_DIR), "y=" + str(Y0) + "_speed=" + str(speed) + "_angle=" + str(angle/pi*180), bounces)
            create_dataset(str(POSITIONS_DIR), "y=" + str(Y0) + "_speed=" + str(speed) + "_angle=" + str(angle/pi*180), data)
            create_dataset(str(RACKET_HIT_FRAMES_DIR), "y=" + str(Y0) + "_speed=" + str(speed) + "_angle=" + str(angle / pi * 180), hits)







