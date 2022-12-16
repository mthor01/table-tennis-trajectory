import h5py
import numpy as np
import cv2 as cv
import os
from sympy import Plane, Point3D, Line3D
import time
import pyrr

# class that stores the intrinsics and extrinsics of a camera
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

#calculate the table plane in camera space
def make_table_plane(cam):
    p1 = [1.0, 0, 0]
    p2 = [0, 1.0, 0]
    p3 = [0, 0, 0]

    p1 = wrld_to_cam(cam, p1)
    p2 = wrld_to_cam(cam, p2)
    p3 = wrld_to_cam(cam, p3)

    p1 = pyrr.Vector3(p1)
    p2 = pyrr.Vector3(p2)
    p3 = pyrr.Vector3(p3)

    return pyrr.plane.create_from_points(p1, p2, p3)

# transform camera space to world space
def cam_to_wrld(cam, cam_coord):
    extrinsics = np.concatenate((cam.extrinsics, [np.array([0, 0, 0, 1])]), axis=0)
    extrinsics = np.linalg.inv(extrinsics)
    homo_cam_coord = np.array([[cam_coord[0]], [cam_coord[1]], [cam_coord[2]], [1]])
    wrld_coord = np.matmul(extrinsics, homo_cam_coord).astype("double")

    return [wrld_coord[0][0], wrld_coord[1][0], wrld_coord[2][0]]

# transform world space to camera space
def wrld_to_cam(cam, wrld_coord):
    extrinsics = np.concatenate((cam.extrinsics, [np.array([0, 0, 0, 1])]), axis=0)
    extrinsics = extrinsics
    homo_wrld_coord = np.array([[wrld_coord[0]], [wrld_coord[1]], [wrld_coord[2]], [1]])
    cam_coord = np.matmul(extrinsics, homo_wrld_coord).astype("double")

    return [cam_coord[0], cam_coord[1], cam_coord[2]]


# draw points on the image to test if 3D to 2D conversion works
def test_3d_to_2d(video_path, point_list, n):
    video = cv.VideoCapture(video_path)
    success, img = video.read()
    for i in range(n):
        img = cv.circle(img, (int(point_list[i][0]), int(point_list[i][1])), 3, (0, 0, 255), -1)
    cv.imshow("img", img)
    cv.waitKey(0)

#find intersection point of a plane and a ray, shot through a 2D position on the screen
def find_intersection(plane, pos_2D, cam):
    arr = np.matmul(np.linalg.inv(cam.intrinsics), np.array([pos_2D[0], pos_2D[1], 1]))
    vec = pyrr.Vector3(arr)
    line = pyrr.line.create_from_points([0,0,0], vec)
    ray = pyrr.ray.create_from_line(line)
    ray[1] = vec
    intersection = pyrr.geometric_tests.ray_intersect_plane(ray, plane)

    return intersection

#project 3D points in world space to 2D screen coordinates
def project_2d(point, cam):
    point = (point[0], point[1], point[2])
    pos, _ = cv.projectPoints(point, cam.rotation, cam.pos, cam.intrinsics, dist_coeffs)
    return pos

#create a plane on which the ball travels, given two bounce points in world coordinates
def make_trajectory_plane(point1, point2):
    point3 = point1+np.array([0,0,1])

    p1 = pyrr.Vector3(point1)
    p2 = pyrr.Vector3(point2)
    p3 = pyrr.Vector3(point3)

    return pyrr.plane.create_from_points(p1,p2,p3)

def get_bounce_frames(pos_list):
    bounce_indices = []
    for i in range(2,len(pos_list)):
        y_sign_change = (pos_list[i-1][1]-pos_list[i-2][1] > 0) and (pos_list[i][1]-pos_list[i-1][1] < 0)
        x_sign_change = (pos_list[i-1][0]-pos_list[i-2][0] > 0) != (pos_list[i][0]-pos_list[i-1][0] > 0)
        if y_sign_change and not(x_sign_change):
            if pos_list[i-1][1] > pos_list[i][1]:
                bounce_indices.append(i-1)
                print(pos_list[i - 1])
            else:
                bounce_indices.append(i)
                print(pos_list[i])
            print([pos_list[i-1], pos_list[i]])

    return bounce_indices

if __name__ == "__main__":
    pose_dir = "pose_estimates"

    dist_coeffs = np.zeros((4, 1))

    for vid in os.listdir(pose_dir):
        pose_file= h5py.File(pose_dir + "/" + vid)
        a_group_key = list(pose_file.keys())[0]
        poses_ = pose_file[a_group_key][()]

        ballpos_file = h5py.File("ball_data/test1_ballpos.hdf5")
        a_group_key = list(ballpos_file.keys())[0]
        ballpos_ = ballpos_file[a_group_key][()]

        bounce_frames_file = h5py.File("ball_data/test1_bouncepos.hdf5")
        a_group_key = list(bounce_frames_file.keys())[0]
        bounces = bounce_frames_file[a_group_key][()]

        n_frames = len(poses_)


        poses = []
        ballpos = []
        bounce_pos = []
        trajectory_planes = []
        tables = []
        final_wrld_pos = []
        error_sum = 0
        error_denominator = 0

        #extract camera positions for every frame
        for i in range(n_frames):
            poses.append(camera([poses_[i][0], poses_[i][1], poses_[i][2]], [poses_[i][3], poses_[i][4], poses_[i][5]], poses_[i][6], poses_[i][7], poses_[i][8]))

            #convert 3D points from a simulation to 2D screen coordinates
            if len(ballpos_[0])==3:
                pos_3d = ((ballpos_[i][0]-2.74/2)*100, (ballpos_[i][1]-1.525/2)*100, ballpos_[i][2]*100)
                ballpos_[i] = pos_3d

                pos_2d, _ = cv.projectPoints(pos_3d, poses[i].rotation, poses[i].pos, poses[i].intrinsics, dist_coeffs)

                ballpos.append(pos_2d[0][0])

            tables.append(make_table_plane(poses[i]))

        #print(bounces)
        #print(get_bounce_frames(ballpos))
      #  bounces = get_bounce_frames(ballpos)

        #calculate world positions of the ball for all bounce frames
        for i in range(len(bounces)):
            frame = bounces[i][1]
            cam = poses[frame]
            img_center = [cam.img_width/2, cam.img_height/2, 0]
            bounce = ballpos[frame]
            table_plane = tables[frame]
            bounce_3d = find_intersection(table_plane, bounce, cam)
            wrld_bounce_3d = cam_to_wrld(cam, bounce_3d)
            bounce_pos.append(wrld_bounce_3d)

        #calculate final world coordinates for every frame by using the bounce positions to create a trajectory plane
        bounce_index = 1
        interv = bounces[bounce_index]
        for i in range(n_frames):
            if (i >= interv[0]):
                if i> interv[1]:
                    bounce_index+=1
                    if bounce_index * 2 - 1 >= len(bounce_pos):
                        break
                    interv = bounces[bounce_index]



                point_2D = ballpos[i]
                trajectory_plane = make_trajectory_plane(bounce_pos[bounce_index * 2 - 2], bounce_pos[bounce_index * 2 - 1])
                ray_point = np.matmul(np.linalg.inv(cam.intrinsics), np.array([point_2D[0], point_2D[1], 1]))
                camera_pos = [0,0,0]
                ray_point = cam_to_wrld(poses[i], ray_point)
                camera_pos = cam_to_wrld(poses[i], camera_pos)
                ray_line = pyrr.line.create_from_points(ray_point,camera_pos)
                ray = pyrr.ray.create_from_line(ray_line)
                test_final = pyrr.geometric_tests.ray_intersect_plane(ray, trajectory_plane)
                error_sum += np.linalg.norm(test_final-ballpos_[i])
                error_denominator += 1
                #print(np.linalg.norm(test_final-ballpos_[i]))
                #print(project_2d(test_final, poses[i]))
                #print(project_2d(ballpos_[i], poses[i]))

    #print(ballpos)
    print("avg error in cm" + str(error_sum/error_denominator))

    test_3d_to_2d("/home/mthor9/Schreibtisch/uni/bachelorarbeit/table_tennis_code/table-tennis-trajectory/single_test_vid/test1.mp4", ballpos, 100)
