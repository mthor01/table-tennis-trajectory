#this script simulates a bunch of table tennis trajectories with varying start postition, velocity, and angular velocity

import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from matplotlib import animation
import h5py
import os
from classes_1 import simulation

matplotlib.use("WebAgg")

table_length = 2.74  # m
table_width = 1.525  # m
net_height = 0.15 # m

#test_speeds = np.array([2., 2.5, 3., 3.5 ])
#test_angles = np.array([-50,-40, -30, -20, -10, 0, 10, 20, 30, 40, 50]) * math.pi / 180
#start_positions_y = np.linspace(0, table_width, 5)
#start_positions_z = [0.2,0.4,0.6]
test_speeds = np.array([5.])
test_angles = np.array([0])
test_z_speeds = np.array([-2])
start_positions_x = np.array([0.02])
start_positions_y = np.array([table_width/2])
start_positions_z = np.array([0.5])

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
    f.create_dataset(filename, data=data)

# execute one step in the simulation and return the balls position after that step
# also returns if the simulation should be stopped because of various reasons defined in the simulation
def get_step():
    sim.step(dt)

    x = sim.position[0]
    y = sim.position[1]
    z = sim.position[2]
    return x, y, z, sim.break_loop

def clear_dir(path):
    for f in os.listdir(path):
        os.remove(path + "/" + f)

if __name__ == "__main__":
    sim = simulation()
    dt = 1/100
    num_samples = 0


    clear_dir("simulated_ball_data/bounce_frames")
    clear_dir("simulated_ball_data/positions")

    for values in itertools.product(test_speeds, test_angles,start_positions_x, start_positions_y, start_positions_z, test_z_speeds):
        speed = values[0]
        angle = values[1]
        X0 = values[2]
        Y0 = values[3]
        Z0 = values[4]
        z_speed = values[5]
        rotation_axis = [0,0,0]
        rpm = 0
       # speed, angle, Y0, Z0, rotation_axis, rpm = 3, 0, table_width/2, 0.5, [0,0,0], 0

        vel = [math.cos(angle), math.sin(angle), 0]
        vel =  vel / np.linalg.norm(vel) * speed
        vel = [vel[0], vel[1], z_speed]

        sim = simulation([X0, Y0, Z0], vel, rotation_axis, rpm)

        bounce_counter = 0
        hit_counter = 0
        i = 0
        j = 0
        last_ball_z_vel = sim.velocity[2]
        last_ball_x_vel = sim.velocity[0]
        data = []
        bounces = []
        hits = []

        # execute the simulation for given values and store the ball positions
        # it stops after 2 bounces
        # only stores if the simulation did not interrupt the loop with break_loop
        while i<200:
            i += 1
            break_loop = False
            for _ in range(3):
                x, y, z, wall_hit = get_step()
                if wall_hit and i >1:
                    break_loop = True

            data.append([x,y,z])
            N = i

            if (last_ball_z_vel < 0) and (sim.velocity[2]>0):
                bounce_counter += 1
                bounces.append(i)

            if (last_ball_x_vel*sim.velocity[0] < 0):
                hits.append(i)
                hit_counter+=1

            last_ball_z_vel = sim.velocity[2]
            last_ball_x_vel = sim.velocity[0]

            if hit_counter >= 2:
                j+=1

           # if bounce_counter >= 10:
               # if sim.position[0] < table_length/2:
                 #break_loop = True
            #    j+=1



        #display_animation(np.transpose(data))

        if break_loop == False:
            num_samples+=1
            create_dataset("simulated_ball_data/bounce_frames", "speed=" + str(speed) + "_angle=" + str(angle/math.pi*180), bounces)
            create_dataset("simulated_ball_data/positions", "speed=" + str(speed) + "_angle=" + str(angle/math.pi*180), data)
            create_dataset("simulated_ball_data/hit_frames", "speed=" + str(speed) + "_angle=" + str(angle / math.pi * 180), hits)







