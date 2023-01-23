import itertools

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import odeint
from matplotlib import animation
import h5py
import os



#simulation properties
dt = 1/90  # time elapsed in one step in seconds
max_steps = 10000  # maximum step amount

# environment properties
gravity = -9.81  # m/s^2
table_length = 2.74  # m
table_width = 1.525  # m
air_density = 1.2  # kg/m^3
drag_coefficient = 0.4
magnus_coefficient = 0.069
mass = 0.0027  # kg
radius = 0.02  # m
COR = 0.9
net_height = 0.15 # m

# set ball start position/movement/rotation
# Y is perpendicular to the table (important to note, as the formulas in the document have the z axis going in that direction)

X0 = 0.  # m
Y0 = 1.525/2  # m
Z0 = 0.5  # m
VX0 = 5.  # m/s
VY0 = 0.  # m/s
VZ0 = 0.  # m/s
rotation_axis = [0, 1, 0]
rpm = 500


class Ball:
    def __init__(self, position=[X0, Y0, Z0], velocity=[VX0, VY0, VZ0], rotation_axis=[0,0,0], rpm=0):
        self.position = position
        self.velocity = velocity
        self.FM = [0,0,0]
        self.buoyancy = 1.25 * math.pi*radius**3 * -air_density * gravity

        self.cross_section = math.pi * radius**2
        self.state_x = self.position[0], self.velocity[0]
        self.state_y = self.position[1], self.velocity[1]
        self.state_z = self.position[2], self.velocity[2]
        self.time_elapsed = 0

        if rpm > 0:
            self.angular_velocity = (rotation_axis / np.linalg.norm(rotation_axis)) * rpm * math.pi * 2 / 60
        else:
            self.angular_velocity = [0,0,0]

    # step the ball's position and self.velocity subject to an acceleration computed from all forces
    def dt_state(self, state, dt, a, drag, FM, buoyancy):
        axe, daxe = state
        ddaxe = a + (-drag + FM + buoyancy) / mass
        return daxe, ddaxe

    # simply reflects the ball when it hits walls
    # if the table is hit, a realistic bounce is applied
    def hit_wall(self):
        hit = False
        if self.position[0] - radius < 0:
            self.velocity[0] = abs(self.velocity[0])
            hit = True
           # self.state_x = self.position_x, self.velocity[0]
        if self.position[0] + radius > table_length:
            self.velocity[0] = -abs(self.velocity[0])
            hit= True
            #self.state_x = self.position_x, self.velocity[0]
        if self.position[2] - radius < 0:
            self.velocity[2] = COR * abs(self.velocity[2])
            ball.bounce_dis_moment()
        if self.position[1] - radius < 0:
            self.velocity[1] = abs(self.velocity[1])
            hit = True
            #self.state_z = self.position_z, self.velocity[2]
        if self.position[1] + radius > table_width:
            self.velocity[1] = -abs(self.velocity[1])
            hit= True
           # self.state_z = self.position_z, self.velocity[2]

        return hit

    def hit_wall_responsive(self):
        if self.position[0] - radius < 0:
            self.velocity[0] = abs(self.velocity[0])
            self.velocity[1] = -self.velocity[1]
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * 3
            self.velocity[2] = 1
           # self.state_x = self.position_x, self.velocity[0]
        if self.position[0] + radius > table_length:
            self.velocity[0] = -abs(self.velocity[0])
            self.velocity[1] = -self.velocity[1]
            self.velocity = self.velocity/np.linalg.norm(self.velocity) * 3
            self.velocity[2] = 1
            #self.state_x = self.position_x, self.velocity[0]
        if self.position[2] - radius < 0:
            self.velocity[2] = COR * abs(self.velocity[2])
            #ball.bounce()
        if self.position[1] - radius < 0:
            self.velocity[1] = abs(self.velocity[1])
            #self.state_z = self.position_z, self.velocity[2]
        if self.position[1] + radius > table_width:
            self.velocity[1] = -abs(self.velocity[1])
           # self.state_z = self.position_z, self.velocity[2]

    # calculate drag force vector
    def get_drag(self):
        CD = drag_coefficient
        # CD = 0.505 + 0.065 * self.get_h()  # can be used instead of the fixed value above
        DF = 0.5*air_density*self.cross_section * CD * np.linalg.norm(self.velocity) * np.array(self.velocity)
        return DF

    # calculate the magnus force vector
    def get_magnus_force(self):
        CM = magnus_coefficient
        if np.any(self.angular_velocity):
            return [0,0,0]
        # CM = 0.094 - 0.026 * self.get_h()  # can be used instead of the fixed value above
        FM = (4/3)*CM*air_density*math.pi*radius**3*np.cross(self.angular_velocity, self.velocity)


        return FM

    # calculates h (a value that can be used to calculate coefficients)
    def get_h(self):
        h = self.velocity[0]*self.angular_velocity[2] - self.velocity[2] * self.angular_velocity[0]
        denominator = math.sqrt(h**2 + (self.velocity[0]**2 + self.velocity[2]**2) * self.angular_velocity[1]**2)
        if denominator > 0:
            return h/denominator
        else:         # happens for example when the ball does not move in x and y direction
            return 1  # i did this because it roughly equals the fixed value for FM and FD after computation

    # check if the net got hit by the ball in order to terminate
    def net_hit(self):
        if (self.position[2] < net_height + radius) & (self.position[0] < table_length/2 + radius) & (self.position[0] > table_length / 2 - radius):
            return True
        else:
            return False

    # applies a realistic bounce to the ball, if it hits the table
    # changes direction, self.velocity and angular self.velocity of the ball
    def bounce_empirical(self):
        b1 = [[0.6278], [-0.0003], [-0.0344]]
        b2 = [[0.7796], [0.0011], [0.3273]]
        b3 = [[-0.5498], [0.8735]]
        b4 = [[7.4760], [0.1205], [39.4228]]
        b5 = [[-22.9295], [0.1838], [-13.4791]]
        b6 = [[-0.3270], [39.9528]]


        self.velocity[0] = np.dot([self.velocity[0], self.angular_velocity[1], 1], b1)
        self.velocity[1] = np.dot([self.velocity[1], self.angular_velocity[0], 1], b2)
        self.velocity[2] = np.dot([self.velocity[0], 1], b3)
        self.angular_velocity[0] = np.dot([self.velocity[1], self.angular_velocity[0], 1], b4)[0]
        self.angular_velocity[1] = np.dot([self.velocity[0], self.angular_velocity[1], 1], b5)[0]
        self.angular_velocity[2] = np.dot([self.velocity[2], 1], b6)[0]


    def bounce_dis_moment(self):
        vel = self.velocity.copy()
        ang_vel = self.angular_velocity.copy()

        contact_point_vel_direction = [-ang_vel[1], ang_vel[0], 0]/np.linalg.norm([-ang_vel[1], ang_vel[0], 0])
    #    print(contact_point_vel_direction)
        rotation_axis_angle = math.acos(min(1, np.dot(ang_vel, np.multiply(ang_vel, [1,1,0])) / (np.linalg.norm(ang_vel) *np.linalg.norm(np.multiply(ang_vel, [1,1,0])))))
        contact_point_rotation_vel = math.cos(rotation_axis_angle) * rpm * 2* math.pi * radius * np.array(contact_point_vel_direction)/60
        contact_point_velocity = [vel[0]*3.6, vel[1]*3.6,0] + contact_point_rotation_vel
        print(vel)
        print(ang_vel)
        print(contact_point_velocity)
        print(rpm * 2* math.pi * radius * np.array(contact_point_vel_direction)/60)
        COF = 0.0017 * np.linalg.norm(contact_point_velocity) * 3.6 + 0.1635
        #print(COF)
        #print(self.angular_velocity)

        roll_or_slide_denominator = math.sqrt((vel[0] - ang_vel[1]*radius)**2 + (vel[1] + ang_vel[0]*radius)**2)
       # print(roll_or_slide_denominator)
        #print(COF*(1+COR)*abs(vel[2]))

        if COF*(1+COR)*abs(vel[2])/roll_or_slide_denominator >= 0.4:
            print(1)
            self.velocity[0] = 0.6*vel[0] + 0.4*ang_vel[1]*radius
            self.velocity[1] = 0.6*vel[1] - 0.4*ang_vel[0]*radius
            self.angular_velocity[0] = 0.4*ang_vel[0] - 0.6*vel[1]/radius
            self.angular_velocity[1] = 0.4*ang_vel[1] + 0.6*vel[0]/radius

        else:
            print(2)
            self.velocity[0] = vel[0] - (COF * (1+COR) * abs(vel[2]) * (vel[0] - ang_vel[1] * radius)) / roll_or_slide_denominator
            self.velocity[1] = vel[1] - (COF * (1+COR) * abs(vel[2]) * (vel[1] + ang_vel[0] * radius)) / roll_or_slide_denominator
            self.angular_velocity[0] = ang_vel[0] - (3 * COF * (1+COR) * abs(vel[2]) * (vel[1] + ang_vel[0] * radius)) / (2 * radius * roll_or_slide_denominator)
            self.angular_velocity[1] = ang_vel[1] - (3 * COF * (1+COR) * abs(vel[2]) * (vel[0] - ang_vel[1] * radius)) / (2 * radius * roll_or_slide_denominator)


    # execute one step of length dt and update the ball
    def step(self, dt):
        """execute one time step of length dt and update state"""
        self.state_x = self.position[0], self.velocity[0]
        self.state_y = self.position[1], self.velocity[1]
        self.state_z = self.position[2], self.velocity[2]

        self.FM = self.get_magnus_force()
        FD = self.get_drag()
        self.state_x = odeint(self.dt_state, self.state_x, [0, dt], args=(0, FD[0], self.FM[0], 0))[1]
        self.state_y = odeint(self.dt_state, self.state_y, [0, dt], args=(0, FD[1], self.FM[1], 0))[1]
        self.state_z = odeint(self.dt_state, self.state_z, [0, dt], args=(gravity, FD[2], self.FM[2], self.buoyancy))[1]

        self.position[0], self.velocity[0] = self.state_x
        self.position[1], self.velocity[1] = self.state_y
        self.position[2], self.velocity[2] = self.state_z

        self.time_elapsed += dt
        self.break_loop = self.hit_wall()
        #self.break_loop = self.net_hit()

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


def get_step():
    global ball, dt, time_elapsed
    ball.step(dt)
    time_elapsed.append(time_elapsed)

    x = ball.position[0]
    y = ball.position[1]
    z = ball.position[2]
    return x, y, z, ball.break_loop

def clear_dir(path):
    for f in os.listdir(path):
        os.remove(path + "/" + f)

ball = Ball()
dt = 1/1000
time_elapsed = [0]
steps = 10000


test_velocities = np.array([2., 2.5, 3., 3.5 ])
test_angles = np.array([-50,-40, -30, -20, -10, 0, 10, 20, 30, 40, 50]) * math.pi / 180
start_positions_y = np.linspace(0, table_width, 5)
start_positions_z = [0.2,0.4,0.6]
num_samples = 0


clear_dir("simulated_ball_data/bounce_frames")
clear_dir("simulated_ball_data/positions")

for values in itertools.product(test_velocities, test_angles, start_positions_y, start_positions_z):
    vel = values[0]
    angle = values[1]
    Y0 = values[2]
    Z0 = values[3]
    vel, angle, Y0, Z0, rotation_axis, rpm = 3, 0, table_width/2, 0.5, [0,-1,0], 1000
    bounce_counter = 0
    hit_counter = 0
    i = 0
    j = 0
    start_vec = [math.cos(angle), math.sin(angle), VZ0]
    ball.velocity = start_vec/np.linalg.norm(start_vec) * vel
    ball.position = [X0, Y0, Z0]
    last_ball_vel = ball.velocity[2]
    last_ball_x_vel = ball.velocity[0]
    ball.angular_velocity = (rotation_axis / np.linalg.norm(rotation_axis)) * rpm * math.pi * 2 / 60
    data = []
    bounces = []

    while j<100:
        i += 1
        break_loop = False
        for _ in range(3):
            x, y, z, wall_hit = get_step()
            if wall_hit and i >1:
                break_loop = True

        data.append([x,y,z])
        N = i

        if (last_ball_vel < 0) and (ball.velocity[2]>0):
            bounce_counter += 1
            bounces.append(i)



        if (last_ball_x_vel*ball.velocity[0] < 0):
            hit_counter+=1

        last_ball_vel = ball.velocity[2]
        last_ball_x_vel = ball.velocity[0]


        if bounce_counter >= 2:
            if ball.position[0] < table_length/2:
                break_loop = True
            j+=1

      #  if ((abs(ball.velocity[0]) < 0.1) and (ball.position[2] < 0.03)) or break_loop:
       #     break

    display_animation(np.transpose(data))
    if break_loop == False:
        num_samples+=1
     #   create_dataset("simulated_ball_data/bounce_frames", "vel=" + str(vel) + "_angle=" + str(angle/math.pi*180), bounces)
      #  create_dataset("simulated_ball_data/positions", "vel=" + str(vel) + "_angle=" + str(angle/math.pi*180), data)







