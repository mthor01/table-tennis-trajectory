import numpy as np
import cv2 as cv
import math
import random
import time
from functions import calc_vec_angle

#stores the coordinates of an intersection point as well as the lines that created it
class IntersecPoint:
    def __init__(self, pos=[], line1_index=None, line2_index=None):
        self.pos = np.array(pos)
        self.line1_index = line1_index
        self.line2_index = line2_index

    def pos_tuple(self):
        return (int(self.pos[0]), int(self.pos[1]))

#stores a line built from 2 points and its angle in the image
class Line:
    def __init__(self, pos1=[], pos2=[], angle=None):
        self.pos1 = np.array(pos1)
        self.pos2 = np.array(pos2)
        self.angle = angle

    def pos1_tuple(self):
        return (int(self.pos1[0]), int(self.pos1[1]))

    def pos2_tuple(self):
        return (int(self.pos2[0]), int(self.pos2[1]))

class Bounds:
    def __init__(self, min_x, max_x, min_y, max_y):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

#stores everything about a camera pose
class Camera:
    def __init__(self, pos, rotation, valid, img_height, img_width):
        self.img_height = img_height
        self.img_width = img_width
        self.valid = valid
        self.pos = np.array(pos)
        self.rotation = np.array(rotation)
        self.intrinsics = np.array([(img_width, 0, img_width / 2),
                                          (0, img_height, img_height / 2),
                                          (0, 0, 1)])

        self.rotation_matrix, _ = cv.Rodrigues(self.rotation)

        self.extrinsics = np.concatenate((self.rotation_matrix, np.transpose([self.pos])), axis=1)
        self.homo_extrinsics = np.concatenate((self.extrinsics, [np.array([0, 0, 0, 1])]), axis=0)



# holds a population of ball start values and can update it
# there are still bugs in this
class Population():
    def __init__(self,cam_wrld_pos, start_ray_vec, NH=72, F=0.5, CR=0.9, delta=1.5):
        self.positions = []
        self.velocities = []
        self.angular_velocities = []
        self.rpm = []
        self.rotation_axes = []
        self.NH = NH
        self.F = F
        self.CR = CR
        self.delta = delta
        self.position_candidates = []
        self.velocity_candidates = []
        self.angular_velocity_candidates = []
        self.performances = []
        self.candidate_performances = []
        self.times_since_bounce = np.ones(NH)*10
        self.start_point_distances = []
        self.start_point_distances_candidates = []
        self.cam_wrld_pos = cam_wrld_pos
        self.start_ray_vec = start_ray_vec


        for i in range(NH):
            #self.positions.append([0.02, 0.7625, 0.5])
            k = random.uniform(0,1)
            self.start_point_distances.append(k)
            self.positions.append(cam_wrld_pos+start_ray_vec*k)
           # self.positions.append(np.array([0,0,0]) + np.array([0.02, 0.7625, 0.5]) * k*2)
            #self.positions.append(np.array(cam_wrld_pos) + np.array(start_ray_vec) * 7.85)
           # self.velocities.append([5, 0, -2])
            self.angular_velocities.append([0,0,0])
           # self.positions.append([random.uniform(-2, 4.74), random.uniform(-1, 2.525), random.uniform(-0.2, 1)])
            self.velocities.append(
                np.array([random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-10, 10)]) / 3.6)
           # self.rotation_axes.append(np.array([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]))
           # self.rpm = random.uniform(0, 5000)
            #self.angular_velocities.append(self.rotation_axes[i] * self.rpm * 2 * math.pi / 60)

    def new_gen(self):
        self.position_candidates = []
        self.velocity_candidates = []
        self.angular_velocity_candidates = []
        self.start_point_distances_candidates = []
        vectors = []
        for i in range(self.NH):
            vectors.append(np.concatenate(([self.start_point_distances[i]], self.velocities[i], self.angular_velocities[i])))

        for i in range(len(vectors)):
            random_vecs = self.pick_3_random(vectors)
            trial_vec = random_vecs[0] + self.F * (random_vecs[1] - random_vecs[2])
            idx = random.randint(0, len(vectors[i]) - 1)
            for j in range(len(vectors[i])):
                if (j == idx) or random.uniform(0, 1) <= self.F:
                    vectors[i][j] = trial_vec[j]

            self.start_point_distances_candidates.append(vectors[i][0])
            self.velocity_candidates.append(vectors[i][1:4])
            self.angular_velocity_candidates.append(vectors[i][4:7])
            self.position_candidates.append(self.cam_wrld_pos+self.start_ray_vec*vectors[i][0])

    def pick_3_random(self, vectors):
        random_picks_idx = []
        random_picks = []
        while len(random_picks) <= 3:
            idx = random.choice(range(0, len(vectors)))
            if not (idx in random_picks_idx):
                random_picks_idx.append(idx)
                random_picks.append(np.array(vectors[idx]))

        return random_picks

    def update(self, candidate_performances):
        for i in range(self.NH):
            if candidate_performances[i] < self.performances[i]:
                self.positions[i] = self.position_candidates[i]
                self.velocities[i] = self.velocity_candidates[i]
                self.angular_velocities[i] = self.angular_velocity_candidates[i]
                self.start_point_distances[i] = self.start_point_distances_candidates[i]
                self.performances[i] = candidate_performances[i]

#for simulating table tennis
#also contains all environmental variables
class SimulationParallel:
    def __init__(self, positions=[[1,1,1]], velocities=[[1, 1, 1]], angular_velocities = [[0,0,0]]):
        self.gravity = -9.81  # m/s^2
        self.table_length = 2.74  # m
        self.table_width = 1.525  # m
        self.air_density = 1.2  # kg/m^3
        self.drag_coefficient = 0.4
        self.magnus_coefficient = 0.069
        self.mass = 0.0027  # kg
        self.radius = 0.02  # m
        self.COR = 0.9
        self.net_height = 0.15  # m
        self.positions = positions
        self.velocities = velocities
        self.FM = [0,0,0]
        self.buoyancy = 1.25 * math.pi*self.radius**3 * -self.air_density * self.gravity
        self.cross_section = math.pi * self.radius**2
        self.time_elapsed = 0
        self.table_height = 0
        self.start_time = time.time()
        self.angular_velocities = []
        self.states_x = []
        self.states_y = []
        self.states_z = []
        self.times_since_bounce = np.ones(len(positions)) * 10

        for i in range(len(positions)):
            self.states_x.append([self.positions[i][0], self.velocities[i][0]])
            self.states_y.append([self.positions[i][1], self.velocities[i][1]])
            self.states_z.append([self.positions[i][2], self.velocities[i][2]])

            if sum(map(abs, angular_velocities[i])) > 0:
                self.angular_velocities.append(angular_velocities[i])
            else:
                self.angular_velocities.append([0.0001,0,0]) # numerical fix

    # step the ball's position and self.velocity subject to an acceleration computed from all forces
    def dt_state(self, positions, velocities, dt, a, drag_forces, FM_list, buoyancy):

        def calc_acceleration(drag_force, FM):
            return a + (-drag_force + FM + buoyancy) / self.mass

        accelerations = list(map(calc_acceleration, drag_forces, FM_list))
        velocities = velocities + np.array(accelerations) * dt
        positions = positions + np.array(velocities)*dt

        return [positions, velocities]

    # simply reflects the ball when it hits walls
    # if it hits the table or ground, a realistic bounce is applied
    def bounds_check(self):
        hits = []

        def calc_state(position, velocity, angular_velocity, time_since_bounce):
            hit = False
            if position[0] - self.radius < 0:
             #   self.velocity[0] = abs(self.velocity[0])
                velocity[0] = 3
                velocity[2] = abs(position[2]-1)*4
                angular_velocity = [0.0001, 0, 0]
                hit = True
                #hit = True
            if position[0] + self.radius > self.table_length:
                #self.velocity[0] = -abs(self.velocity[0])
                velocity[0] = -3
                velocity[2] = abs(position[2] - 1)*4
                angular_velocity = [0.0001, 0, 0]
                hit = True
            if (position[2] - self.radius < self.table_height):
                in_x_bounds = 0 < position[0] < self.table_length
                in_y_bounds = 0 < position[1] < self.table_width
                if (position[2] - self.radius < 0) or (in_x_bounds and in_y_bounds):
                    velocity[2] = self.COR * abs(velocity[2])
                    if time_since_bounce > 3:
                        bounce_dis_moment(velocity, angular_velocity)
            if position[1] - self.radius < 0:
                velocity[1] = abs(velocity[1])
                hit = True
            if position[1] + self.radius > self.table_width:
                velocity[1] = -abs(velocity[1])
                hit = True


            return [position, velocity, angular_velocity, time_since_bounce, hit]
        # calculate the bounce following a discrete momentum theorem based model
        def bounce_dis_moment(velocity, angular_velocity):
            vel = velocity
            ang_vel = angular_velocity

            contact_point_vel_direction = [-ang_vel[1], ang_vel[0], 0] / np.linalg.norm([-ang_vel[1], ang_vel[0], 0])
            rotation_ground_factor = min(1, np.dot(ang_vel, np.multiply(ang_vel, [1, 1, 0])) / (
                        np.linalg.norm(ang_vel) * np.linalg.norm(np.multiply(ang_vel, [1, 1, 0]))))
            contact_point_rotation_vel = rotation_ground_factor * np.linalg.norm(angular_velocity) * self.radius * np.array(contact_point_vel_direction)
            contact_point_velocity = [vel[0] * 3.6, vel[1] * 3.6, 0] + contact_point_rotation_vel
            COF = 0.0017 * np.linalg.norm(contact_point_velocity) * 3.6 + 0.1635

            roll_or_slide_denominator = math.sqrt(
                (vel[0] - ang_vel[1] * self.radius) ** 2 + (vel[1] + ang_vel[0] * self.radius) ** 2)

            if COF * (1 + self.COR) * abs(vel[2]) / roll_or_slide_denominator >= 0.4:
                velocity[0] = 0.6 * vel[0] + 0.4 * ang_vel[1] * self.radius
                velocity[1] = 0.6 * vel[1] - 0.4 * ang_vel[0] * self.radius
                angular_velocity[0] = 0.4 * ang_vel[0] - 0.6 * vel[1] / self.radius
                angular_velocity[1] = 0.4 * ang_vel[1] + 0.6 * vel[0] / self.radius

            else:
                velocity[0] = vel[0] - (COF * (1 + self.COR) * abs(vel[2]) * (
                            vel[0] - ang_vel[1] * self.radius)) / roll_or_slide_denominator
                velocity[1] = vel[1] - (COF * (1 + self.COR) * abs(vel[2]) * (
                            vel[1] + ang_vel[0] * self.radius)) / roll_or_slide_denominator
                angular_velocity[0] = ang_vel[0] - (
                            3 * COF * (1 + self.COR) * abs(vel[2]) * (vel[1] + ang_vel[0] * self.radius)) / (
                                                       2 * self.radius * roll_or_slide_denominator)
                angular_velocity[1] = ang_vel[1] - (
                            3 * COF * (1 + self.COR) * abs(vel[2]) * (vel[0] - ang_vel[1] * self.radius)) / (
                                                       2 * self.radius * roll_or_slide_denominator)

            for i in range(3):
                if math.isnan(angular_velocity[i]):
                    angular_velocity[i] = 0

            time_since_bounce = 0


        l = list(map(calc_state, self.positions, self.velocities, self.angular_velocities, self.times_since_bounce))
        for i in range(len(l)):
            self.positions[i] = l[i][0]
            self.velocities[i] = l[i][1]
            self.angular_velocities[i] = l[i][2]
            self.times_since_bounce[i] = l[i][3]
            hits.append(l[i][4])

        return hits

    # calculate drag force vector
    def get_drag(self):
        CD = self.drag_coefficient
        # CD = 0.505 + 0.065 * self.get_h()  # can be used instead of the fixed value above

        def calc_drag(velocity):
            return 0.5 * self.air_density * self.cross_section * CD * np.linalg.norm(velocity) * np.array(velocity)

        drag_forces = list(map(calc_drag, self.velocities))
        return drag_forces

    # calculate the magnus force vector
    def get_magnus_force(self):
        CM = self.magnus_coefficient
        # CM = 0.094 - 0.026 * self.get_h()  # can be used instead of the fixed value above
        def calc_magnus_force(angular_velocity, velocity):
            return (4 / 3) * CM * self.air_density * math.pi * self.radius ** 3 * cross_product(angular_velocity, velocity)

        def cross_product(vec1, vec2):
            return np.array([vec1[1] * vec2[2] - vec1[2] * vec2[1],
                    vec1[2] * vec2[0] - vec1[0] * vec2[2],
                    vec1[0] * vec2[1] - vec1[1] * vec2[0]])

        FM_list = list(map(calc_magnus_force, self.angular_velocities, self.velocities))

        return FM_list

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
        if (self.position[2] < self.net_height + self.radius) & (self.position[0] < self.table_length/2 + self.radius) & (self.position[0] > self.table_length / 2 - self.radius):
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



    # execute one step of length dt and update the ball
    def step(self, dt):
        """execute one time step of length dt and update state"""
        start_time = time.time()

        FM_list = np.transpose(self.get_magnus_force())

        FD_list = np.transpose(self.get_drag())



        t_positions = np.transpose(self.positions)
        t_velocities = np.transpose(self.velocities)

        pos_x, vel_x = self.dt_state(t_positions[0], t_velocities[0], dt, 0, FD_list[0], FM_list[0], 0)
        pos_y, vel_y = self.dt_state(t_positions[1], t_velocities[1], dt, 0, FD_list[1], FM_list[1], 0)
        pos_z, vel_z = self.dt_state(t_positions[2], t_velocities[2], dt, self.gravity, FD_list[2], FM_list[2], self.buoyancy)

        start_time = time.time()

        for i in range(len(pos_x)):
            self.positions[i] = [pos_x[i], pos_y[i], pos_z[i]]
            self.velocities[i] = [vel_x[i], vel_y[i], vel_z[i]]



        self.time_elapsed += dt
        self.break_loop = self.bounds_check()
        self.times_since_bounce += 1

        return self.positions, self.break_loop
