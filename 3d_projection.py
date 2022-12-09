import h5py
import numpy as np
import cv2 as cv
import os
from sympy import Plane, Point3D, Line3D

# class that stores the intrinsics and extrinsics of a camera
class camera:
    def __init__(self, pos, rotation, valid, img_height, img_width):
        self.valid = valid
        self.pos = np.array(pos)
        self.rotation = np.array(rotation)
        self.intrinsics = np.array([(img_width, 0, img_width / 2),
                                          (0, img_height, img_height / 2),
                                          (0, 0, 1)])

        self.rotation_matrix, _ = cv.Rodrigues(self.rotation)

        self.extrinsics = np.concatenate((self.rotation_matrix, np.transpose([self.pos])), axis=1)
        self.homo_extrinsics = np.concatenate((self.extrinsics, [np.array([0, 0, 0, 1])]), axis=0)

# calculate the table plane in camera space
def make_table_plane(cam):
    p1 = [1.0, 0, 0]
    p2 = [0, 1.0, 0]
    p3 = [0, 0, 1.0]

    p1 = wrld_to_cam(cam, p1)
    p2 = wrld_to_cam(cam, p2)
    p3 = wrld_to_cam(cam, p3)

    p1 = Point3D(p1[0][0], p1[1][0], p1[2][0])
    p2 = Point3D(p2[0][0], p2[1][0], p2[2][0])
    p3 = Point3D(p3[0][0], p3[1][0], p3[2][0])

    return Plane(p1, p2, p3)

# transform camera space to world space
def cam_to_wrld(cam, cam_coord):
    extrinsics = np.concatenate((cam.extrinsics, [np.array([0, 0, 0, 1])]), axis=0)
    extrinsics = np.linalg.inv(extrinsics)
    homo_cam_coord = np.array([[cam_coord[0]], [cam_coord[1]], [cam_coord[2]], [1]])
    wrld_coord = np.matmul(extrinsics, homo_cam_coord).astype("double")

    return [wrld_coord[0], wrld_coord[1], wrld_coord[2]]

# transform world space to camera space
def wrld_to_cam(cam, wrld_coord):
    extrinsics = np.concatenate((cam.extrinsics, [np.array([0, 0, 0, 1])]), axis=0)
    extrinsics = np.linalg.inv(extrinsics)
    homo_wrld_coord = np.array([[wrld_coord[0]], [wrld_coord[1]], [wrld_coord[2]], [1]])
    cam_coord = np.matmul(extrinsics, homo_wrld_coord).astype("double")

    return [cam_coord[0], cam_coord[1], cam_coord[2]]


# draw points on the image to test if 3D to 2D conversion works
def test_3d_to_2d(video_path, point_list):
    video = cv.VideoCapture(video_path)
    success, img = video.read()
    for i in range(100):
        img = cv.circle(img, (int(point_list[i][0]), int(point_list[i][1])), 3, (0, 0, 255), -1)
    cv.imshow("img", img)
    cv.waitKey(0)

def find_intersection(plane, pos_2D, cam):
    ray = np.matmul(np.linalg.inv(cam.intrinsics), np.array([pos_2D[0], pos_2D[1], 1]))
    ray_line = Line3D(Point3D(0, 0, 0), Point3D(ray[0], ray[1], ray[2]))
    intersection = plane.intersection(ray_line)
    intersection = np.array([float(intersection[0].x), float(intersection[0].y), float(intersection[0].z), 1])

    return intersection

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

        poses = []
        ballpos = []

        for i in range(len(poses_)):
            poses.append(camera([poses_[i][0], poses_[i][1], poses_[i][2]], [poses_[i][3], poses_[i][4], poses_[i][5]], poses_[i][6], poses_[i][7], poses_[i][8]))

            #for positions from the simulation
            if len(ballpos_[0])==3:
                pos_3d = ((ballpos_[i][0]-2.74/2)*100, (ballpos_[i][1]-1.525/2)*100, ballpos_[i][2]*100)

                pos_2d, _ = cv.projectPoints(pos_3d, poses[i].rotation, poses[i].pos, poses[i].intrinsics, dist_coeffs)

                ballpos.append(pos_2d[0][0])


        for i in range(len(bounces)):
            frame = bounces[i][1]
            bounce = ballpos[frame]
            table_plane = make_table_plane(poses[frame])
            bounce_3d = find_intersection(table_plane, bounce, poses[frame])
            wrld_bounce_3d = cam_to_wrld(poses[frame], bounce_3d)
            #print(wrld_bounce_3d)


            #pos_2d, _ = cv.projectPoints((bounce1_3d[0], bounce1_3d[1], bounce1_3d[2]), poses[i].rotation, poses[i].pos, poses[i].intrinsics, dist_coeffs)


    test_3d_to_2d("/home/mthor9/Schreibtisch/uni/bachelorarbeit/table_tennis_code/table-tennis-trajectory/single_test_vid/test1.mp4", ballpos)
