import numpy as np
import cv2 as cv
import math
import random
import time

#stores the coordinates of an intersection point as well as the lines that created it
class intersec_point:
    def __init__(self, pos=[], line1_index=None, line2_index=None):
        self.pos = np.array(pos)
        self.line1_index = line1_index
        self.line2_index = line2_index

    def pos_tuple(self):
        return (int(self.pos[0]), int(self.pos[1]))

#stores a line built from 2 points and its angle in the image
class line:
    def __init__(self, pos1=[], pos2=[], angle=None):
        self.pos1 = np.array(pos1)
        self.pos2 = np.array(pos2)
        self.angle = angle

    def pos1_tuple(self):
        return (int(self.pos1[0]), int(self.pos1[1]))

    def pos2_tuple(self):
        return (int(self.pos2[0]), int(self.pos2[1]))

class bounds:
    def __init__(self, min_x, max_x, min_y, max_y):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

#stores everything about a camera pose
class camera:
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

#for simulating table tennis
#also contains all environmental variables
class simulation:
    def __init__(self, position=[1,1,1], velocity=[1, 1, 1], rotation_axis=[0,0,0], rpm=0, radius = 0.02, gravity = -9.81):
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
        self.rpm = max(rpm, 0.001) # numerical fix
        self.radius = radius
        self.position = position
        self.velocity = velocity
        self.FM = [0,0,0]
        self.buoyancy = 1.25 * math.pi*self.radius**3 * -self.air_density * gravity
        self.cross_section = math.pi * self.radius**2
        self.state_x = self.position[0], self.velocity[0]
        self.state_y = self.position[1], self.velocity[1]
        self.state_z = self.position[2], self.velocity[2]
        self.time_elapsed = 0
        self.time_since_bounce = 10
        self.table_height = 0
        self.start_time = time.time()




        if sum(map(abs, rotation_axis)) > 0:
            self.angular_velocity = (rotation_axis / np.linalg.norm(rotation_axis)) * self.rpm * math.pi * 2 / 60
        else:
            self.angular_velocity = [0.0001,0,0] # numerical fix

    # step the ball's position and self.velocity subject to an acceleration computed from all forces
    def dt_state(self, state, dt, a, drag, FM, buoyancy):
        pos, vel = state
        acceleration = a + (-drag + FM + buoyancy)/self.mass
        vel = vel + acceleration * dt
        pos = pos + vel*dt

        return pos, vel

    # simply reflects the ball when it hits walls
    # if it hits the table or ground, a realistic bounce is applied
    def bounds_check(self):
        hit = False
        if self.position[0] - self.radius < 0:
         #   self.velocity[0] = abs(self.velocity[0])
            self.velocity[0] = 3
            self.velocity[2] = abs(self.position[2]-1)*4
            self.angular_velocity = [0.0001, 0, 0]

            #hit = True
        if self.position[0] + self.radius > self.table_length:
            #self.velocity[0] = -abs(self.velocity[0])
            self.velocity[0] = -3
            self.velocity[2] = abs(self.position[2] - 1)*4
            self.angular_velocity = [0.0001, 0, 0]
            #hit = True
        if (self.position[2] - self.radius < self.table_height):
            in_x_bounds = 0 < self.position[0] < self.table_length
            in_y_bounds = 0 < self.position[1] < self.table_width
            if (self.position[2] - self.radius < 0) or (in_x_bounds and in_y_bounds):
                self.velocity[2] = self.COR * abs(self.velocity[2])
                if self.time_since_bounce > 3:
                    self.bounce_dis_moment()
        if self.position[1] - self.radius < 0:
            self.velocity[1] = abs(self.velocity[1])
            hit = True
        if self.position[1] + self.radius > self.table_width:
            self.velocity[1] = -abs(self.velocity[1])
            hit = True

        return hit

    # calculate drag force vector
    def get_drag(self):
        CD = self.drag_coefficient
        # CD = 0.505 + 0.065 * self.get_h()  # can be used instead of the fixed value above
        DF = 0.5*self.air_density*self.cross_section * CD * np.linalg.norm(self.velocity) * np.array(self.velocity)
        return DF

    # calculate the magnus force vector
    def get_magnus_force(self):
        CM = self.magnus_coefficient
        # CM = 0.094 - 0.026 * self.get_h()  # can be used instead of the fixed value above
        FM = (4/3)*CM*self.air_density*math.pi*self.radius**3*np.cross(self.angular_velocity, self.velocity)
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

    # calculate the bounce following a discrete momentum theorem based model
    def bounce_dis_moment(self):
        vel = self.velocity.copy()
        ang_vel = self.angular_velocity.copy()

        contact_point_vel_direction = [-ang_vel[1], ang_vel[0], 0]/np.linalg.norm([-ang_vel[1], ang_vel[0], 0])
        rotation_axis_angle = math.acos(min(1, np.dot(ang_vel, np.multiply(ang_vel, [1,1,0])) / (np.linalg.norm(ang_vel) *np.linalg.norm(np.multiply(ang_vel, [1,1,0])))))
        contact_point_rotation_vel = math.cos(rotation_axis_angle) * self.rpm * 2* math.pi * self.radius * np.array(contact_point_vel_direction)/60
        contact_point_velocity = [vel[0]*3.6, vel[1]*3.6,0] + contact_point_rotation_vel
        COF = 0.0017 * np.linalg.norm(contact_point_velocity) * 3.6 + 0.1635

        roll_or_slide_denominator = math.sqrt((vel[0] - ang_vel[1]*self.radius)**2 + (vel[1] + ang_vel[0]*self.radius)**2)

        if COF*(1+self.COR)*abs(vel[2])/roll_or_slide_denominator >= 0.4:
            self.velocity[0] = 0.6*vel[0] + 0.4*ang_vel[1]*self.radius
            self.velocity[1] = 0.6*vel[1] - 0.4*ang_vel[0]*self.radius
            self.angular_velocity[0] = 0.4*ang_vel[0] - 0.6*vel[1]/self.radius
            self.angular_velocity[1] = 0.4*ang_vel[1] + 0.6*vel[0]/self.radius

        else:
            self.velocity[0] = vel[0] - (COF * (1+self.COR) * abs(vel[2]) * (vel[0] - ang_vel[1] * self.radius)) / roll_or_slide_denominator
            self.velocity[1] = vel[1] - (COF * (1+self.COR) * abs(vel[2]) * (vel[1] + ang_vel[0] * self.radius)) / roll_or_slide_denominator
            self.angular_velocity[0] = ang_vel[0] - (3 * COF * (1+self.COR) * abs(vel[2]) * (vel[1] + ang_vel[0] * self.radius)) / (2 * self.radius * roll_or_slide_denominator)
            self.angular_velocity[1] = ang_vel[1] - (3 * COF * (1+self.COR) * abs(vel[2]) * (vel[0] - ang_vel[1] * self.radius)) / (2 * self.radius * roll_or_slide_denominator)

        for i in range(3):
            if math.isnan(self.angular_velocity[i]):
                self.angular_velocity[i] = 0

        self.time_since_bounce = 0

    # execute one step of length dt and update the ball
    def step(self, dt):
        """execute one time step of length dt and update state"""
        self.start_time = time.time()
        self.state_x = self.position[0], self.velocity[0]
        self.state_y = self.position[1], self.velocity[1]
        self.state_z = self.position[2], self.velocity[2]

        self.FM = self.get_magnus_force()
        FD = self.get_drag()

        self.state_x = self.dt_state(self.state_x, dt, 0, FD[0], self.FM[0], 0)
        self.state_y = self.dt_state(self.state_y, dt, 0, FD[1], self.FM[1], 0)
        self.state_z = self.dt_state(self.state_z, dt, self.gravity, FD[2], self.FM[2], self.buoyancy)

        self.position[0], self.velocity[0] = self.state_x
        self.position[1], self.velocity[1] = self.state_y
        self.position[2], self.velocity[2] = self.state_z

        self.time_elapsed += dt
        self.break_loop = self.bounds_check()
        self.time_since_bounce += 1

        x = self.position[0]
        y = self.position[1]
        z = self.position[2]
        return x, y, z, self.break_loop

class functions:
    #converts camera coordinates to world coordinates
    def cam_to_wrld(cam, cam_coord):
        extrinsics = np.concatenate((cam.extrinsics, [np.array([0, 0, 0, 1])]), axis=0)
        extrinsics = np.linalg.inv(extrinsics)
        homo_cam_coord = np.array([[cam_coord[0]], [cam_coord[1]], [cam_coord[2]], [1]])
        wrld_coord = np.matmul(extrinsics, homo_cam_coord).astype("double")

        return np.array([wrld_coord[0][0], wrld_coord[1][0], wrld_coord[2][0]])

    # converts world coordinates to camera coordinates
    def wrld_to_cam(cam, wrld_coord):
        extrinsics = np.concatenate((cam.extrinsics, [np.array([0, 0, 0, 1])]), axis=0)
        extrinsics = extrinsics
        homo_wrld_coord = np.array([[wrld_coord[0]], [wrld_coord[1]], [wrld_coord[2]], [1]])
        cam_coord = np.matmul(extrinsics, homo_wrld_coord).astype("double")

        return np.array([cam_coord[0], cam_coord[1], cam_coord[2]])

    # calculate angle between 2 vectors
    def calc_vec_angle(vec1, vec2):
        return math.acos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))) / math.pi * 180

# holds a population of ball start values and can update it
# there are still bugs in this
class population():
    def __init__(self, NH=72, F=0.5, CR=0.9, delta=1.5):
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

        for i in range(NH):
            self.positions.append([random.uniform(-2, 4.74), random.uniform(-1, 2.525), random.uniform(0, 1)])
            self.velocities.append(
                np.array([random.uniform(5, 10), random.uniform(5, 10), random.uniform(5, 10)]) / 3.6)
            self.rotation_axes.append(np.array([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]))
            self.rpm = random.uniform(0, 1000)
            self.angular_velocities.append(self.rotation_axes[i] * self.rpm * 2 * math.pi / 60)

          #  self.positions.append([1, 1, 1])
           # self.velocities.append([1, 1, 1])
            #self.rotation_axes.append([1, 1, 1])
            #self.rpm = 1
            #self.angular_velocities.append(self.rotation_axes[i] * self.rpm * 2 * math.pi / 60)

    def new_gen(self):
        self.position_candidates = []
        self.velocity_candidates = []
        self.angular_velocity_candidates = []
        vectors = []
        for i in range(self.NH):
            vectors.append(np.concatenate((self.positions[i], self.velocities[i], self.angular_velocities[i])))

        for i in range(len(vectors)):
            random_vecs = self.pick_3_random(vectors)
            trial_vec = random_vecs[0] + self.F * (random_vecs[1] - random_vecs[2])
            idx = random.randint(0, len(vectors[i]) - 1)
            for j in range(len(vectors[i])):
                if (j == idx) or random.uniform(0, 1) <= self.F:
                    vectors[i][j] = trial_vec[j]

            self.position_candidates.append(vectors[i][0:3])
            self.velocity_candidates.append(vectors[i][3:6])
            self.angular_velocity_candidates.append(vectors[i][6:9])

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
                self.performances[i] = candidate_performances[i]
