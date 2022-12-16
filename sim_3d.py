import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import odeint
from matplotlib import animation
import h5py

# set ball start position/movement/rotation 
# Y is perpendicular to the table (important to note, as the formulas in the document have the z axis going in that direction)
X0 = 0.  # m
Y0 = 0.5  # m
Z0 = 1.525/2  # m
VX0 = 5.  # m/s
VY0 = -2.  # m/s
VZ0 = 0.  # m/s
rotation_axis = [0, 1, 0]
rpm = 0

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
net_height=0.15

#simulation properties
dt = 1/90  # time elapsed in one step in seconds
max_steps = 10000  # maximum step amount


class Ball:
    def __init__(self):
        self.position_y = Y0
        self.position_x = X0
        self.position_z = Z0
        self.velocity = [VX0, VY0, VZ0]
        self.FM = [0,0,0]
        self.buoyancy = 1.25 * math.pi*radius**3 * -air_density * gravity
        self.angular_velocity = (rotation_axis / np.linalg.norm(rotation_axis)) * rpm * math.pi * 2/60
        self.cross_section = math.pi * radius**2
        self.state_x = self.position_x, self.velocity[0]
        self.state_y = self.position_y, self.velocity[1]
        self.state_z = self.position_z, self.velocity[2]
        self.time_elapsed = 0

    # step the ball's position and self.velocity subject to an acceleration computed from all forces
    def dt_state(self, state, dt ,a ,drag, FM, buoyancy):
        axe, daxe = state
        ddaxe = a + (-drag + FM + buoyancy)/mass
        return daxe, ddaxe

    # simply reflects the ball when it hits walls
    # if the table is hit, a realistic bounce is applied
    def hit_wall(self):
        if self.position_x - radius < 0:
            self.velocity[0] = abs(self.velocity[0])
            self.state_x = self.position_x, self.velocity[0]
        if self.position_x + radius > table_length:
            self.velocity[0] = -abs(self.velocity[0])
            self.state_x = self.position_x, self.velocity[0]
        if self.position_y - radius < 0:
            ball.bounce()
        if self.position_z - radius < 0:
            self.velocity[2] = abs(self.velocity[2])
            self.state_z = self.position_z, self.velocity[2]
        if self.position_z + radius > table_width:
            self.velocity[2] = -abs(self.velocity[2])
            self.state_z = self.position_z, self.velocity[2]

    # calculate drag force vector
    def get_drag(self):
        CD = drag_coefficient
        # CD = 0.505 + 0.065 * self.get_h()  # can be used instead of the fixed value above
        DF = 0.5*air_density*self.cross_section * CD * np.linalg.norm(self.velocity) * np.array(self.velocity)
        return DF

    # calculate the magnus force vector
    def get_magnus_force(self):
        CM = magnus_coefficient
        # CM = 0.094 - 0.026 * self.get_h()  # can be used instead of the fixed value above
        self.FM = (4/3)*CM*air_density*math.pi*radius**3*np.cross(self.angular_velocity, self.velocity)
        return self.FM

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
        if (self.position_y < net_height + radius) & (self.position_x < table_length/2 + radius) & (self.position_x > table_length / 2 - radius):
            return True
        else:
            return False

    # applies a realistic bounce to the ball, if it hits the table
    # changes direction, self.velocity and angular self.velocity of the ball
    def bounce(self):
        CF = -0.0011 * self.velocity[1] * 3.6+0.2526
        CR = min(0.0058 * self.velocity[1] * 3.6 + 1, 0.9)
        numerical_fix = 10**10
        denominator = math.sqrt((self.velocity[0] - self.angular_velocity[2] * radius) ** 2 + (self.velocity[2] + self.angular_velocity[0] * radius) ** 2)
        x = CF * (1+CR) * abs(self.velocity[1]) / denominator

        if x >= 0.4:
            self.velocity[0] = 0.6 * self.velocity[0] - 0.4 * self.angular_velocity[2] * radius #changed first + to -
            self.velocity[2] = 0.6 * self.velocity[2] + 0.4 * self.angular_velocity[0] * radius #changed first - to +
           # self.angular_velocity[0] = 0.4 * self.angular_velocity[0] - 0.6 * self.velocity[2] / radius
            #self.angular_velocity[2] = 0.4 * self.angular_velocity[2] + 0.6 * self.velocity[0] / radius

        else:
            self.velocity[0] = self.velocity[0] + CF * (1+CR) * abs(self.velocity[1]) * (self.velocity[0] - self.angular_velocity[2] * radius) / denominator #changed first - to +
            self.velocity[2] = self.velocity[2] + CF * (1+CR) * abs(self.velocity[1]) * (self.velocity[2] + self.angular_velocity[0] * radius) / denominator #changed first - to +
            #self.angular_velocity[0] = self.angular_velocity[0] - 3 * CF * (1+CR) * abs(self.velocity[1]) * (self.velocity[2] + self.angular_velocity[0] * radius) / (2*radius*denominator)
            #self.angular_velocity[2] = self.angular_velocity[2] - 3 * CF * (1+CR) * abs(self.velocity[1]) * (-self.velocity[0] + self.angular_velocity[2] * radius) / (2*radius*denominator)


        self.state_x = self.position_x, self.velocity[0]
        self.state_z = self.position_z, self.velocity[2]

        self.velocity[1] = COR * abs(self.velocity[1])
        self.state_y = self.position_y, self.velocity[1]

    # execute one step of length dt and update the ball
    def step(self, dt):
        """execute one time step of length dt and update state"""
        self.FM = self.get_magnus_force()
        FD = self.get_drag()
        self.state_x = odeint(self.dt_state, self.state_x, [0, dt], args=(0, FD[0], self.FM[0], 0))[1]
        self.state_y = odeint(self.dt_state, self.state_y, [0, dt], args=(gravity, FD[1], self.FM[1], self.buoyancy))[1]
        self.state_z = odeint(self.dt_state, self.state_z, [0, dt], args=(0, FD[2], self.FM[2], 0))[1]
        self.position_x, self.velocity[0] = self.state_x
        self.position_y, self.velocity[1] = self.state_y
        self.position_z, self.velocity[2] = self.state_z
        self.time_elapsed += dt
        self.hit_wall()
        self.break_loop = self.net_hit()



ball = Ball()
dt = 1/90
time_elapsed = [0]
steps = 10000
data = np.zeros((3, steps))

def get_step():
    global ball, dt, time_elapsed
    ball.step(dt)
    time_elapsed.append(time_elapsed)

    x = ball.position_x
    y = ball.position_y
    z = ball.position_z
    return x, y, z, ball.break_loop

for i in range(steps):
    x, y, z, break_loop = get_step()
    data[0, i] = x
    data[2, i] = y
    data[1, i] = z
    N = i
    if (abs(ball.velocity[1])<0.1) & (ball.position_y < 0.03) | break_loop:
        break




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
ax.set_ylabel('Z')

ax.set_zlim3d([0.0, table_length])
ax.set_zlabel('Y')

#run the animation
#ani = animation.FuncAnimation(fig, update, N, fargs=(data, line), interval=1/90, blit=False)
#plt.show()

data = np.transpose(data)

data_30= []
for i in range(len(data)):
    if i%3 == 0:
        data_30.append(data[i])



position_file = h5py.File("ball_data/test1_ballpos.hdf5", "w")
position_data = position_file.create_dataset("positions", data=data_30)

bounces = []
last_bounce = 0
for i in range(2, len(data_30)):
    a = (data_30[i-1][2] - data_30[i-2][2]) < 0
    b = (data_30[i][2] - data_30[i-1][2]) > 0
    if a and b:
        if data_30[i-1][2] < data_30[i][2]:
            bounces.append([last_bounce, i-1])
            last_bounce = i - 1
            print(1)
        else:
            bounces.append([last_bounce, i])
            last_bounce = i

print(bounces)

bounce_file = h5py.File("ball_data/test1_bouncepos.hdf5", "w")
bounce_data = bounce_file.create_dataset("bounce_positions", data=bounces)





