import h5py
import numpy as np
import cv2 as cv
import os
import pyrr
import re
from classes.camera import camera

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

    return np.array([wrld_coord[0][0], wrld_coord[1][0], wrld_coord[2][0]])

# transform world space to camera space
def wrld_to_cam(cam, wrld_coord):
    extrinsics = np.concatenate((cam.extrinsics, [np.array([0, 0, 0, 1])]), axis=0)
    extrinsics = extrinsics
    homo_wrld_coord = np.array([[wrld_coord[0]], [wrld_coord[1]], [wrld_coord[2]], [1]])
    cam_coord = np.matmul(extrinsics, homo_wrld_coord).astype("double")

    return np.array([cam_coord[0], cam_coord[1], cam_coord[2]])


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
    if point1[1] == point2[1]:
        point2 = point2+np.array([1,0,0])

    point3 = point1+np.array([0,0,1])

    p1 = pyrr.Vector3(point1)
    p2 = pyrr.Vector3(point2)
    p3 = pyrr.Vector3(point3)

    print(point1, point2, point3)

  #  print(point1, point2, point3)



    return pyrr.plane.create_from_points(p1,p2,p3)

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

if __name__ == "__main__":
    pose_dir = "pose_estimates"
    trajectory_dir = "simulated_ball_data/positions"
    bounce_dir = "simulated_ball_data/bounce_frames"
    output_dir = "3d_ball_positions"
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
            trajectory_planes = []
            tables = []
            final_wrld_pos = []
            vertical_cam_angles = []
            horizontal_cam_angles = []
            final_3d_pos = []
            error_sum = 0
            error_denominator = 0


            #extract camera positions for every frame
            for i in range(n_frames):
                poses.append(camera([poses_[i][0], poses_[i][1], poses_[i][2]], [poses_[i][3], poses_[i][4], poses_[i][5]], poses_[i][6], poses_[i][7], poses_[i][8]))
                tables.append(make_table_plane(poses[i]))

            if len(ballpos_[0]) == 3:
                ballpos, bounces_, new_set_frames, test_positions = convert_sim_data(ballpos_, bounces, n_frames, poses)

            bounces, racket_hits = get_event_frames(ballpos)

            print(bounces)
            print(bounces_)

            #calculate world positions of the ball for all bounce frames
            recent_bounce_pos=0
            for i in range(len(bounces)):
                frame = bounces[i]
                cam = poses[frame]
                img_center = [cam.img_width/2, cam.img_height/2, 0]
                bounce = [ballpos[frame][0], ballpos[frame][1]]
                table_plane = tables[frame]
             #   print(table_plane)
            #    print(bounce)
             #   print(frame)
              #  print(cam.extrinsics)
                bounce_3d = find_intersection(table_plane, bounce, cam)
                wrld_bounce_3d = cam_to_wrld(cam, bounce_3d)
                bounce_pos.append(wrld_bounce_3d)

             #   if wrld_bounce_3d[1] == recent_bounce_pos:
                #    print(table_plane)
               #     print(ballpos[frame], ballpos[frame-1])

                recent_bounce_pos = wrld_bounce_3d[1]



          #  final_event_list = []
           # last_racket_hit = None
            #last_used_bounce = -1
            #new_set = True

            #for i in range(len(racket_hits)):
             #   for j in range(len(bounces)):
              #      if (bounces[j] > racket_hits[i]) and new_set and j > last_used_bounce:
               #         final_event_list.append()


            #calculate final world coordinates for every frame by using the bounce positions to create a trajectory plane
            bounce_index = 0

           # print(bounces)

            for i in range(n_frames):
                if i> bounces[bounce_index+1]:
                    bounce_index+=1
                    if bounce_index+1 >= len(bounce_pos):
                        break
                point_2D = ballpos[i]

              #  print(bounce_index)
               # print(ballpos[bounce_index], ballpos[bounce_index+1])
                #print(bounce_pos[bounce_index], bounce_pos[bounce_index+1])
                trajectory_plane = make_trajectory_plane(bounce_pos[bounce_index], bounce_pos[bounce_index+1])
                ray_point = np.matmul(np.linalg.inv(cam.intrinsics), np.array([point_2D[0], point_2D[1], 1]))
                camera_pos = [0,0,0]
                ray_point = cam_to_wrld(poses[i], ray_point)
                camera_pos = cam_to_wrld(poses[i], camera_pos)
                ray_line = pyrr.line.create_from_points(ray_point,camera_pos)
                ray = pyrr.ray.create_from_line(ray_line)
                final_pos = pyrr.geometric_tests.ray_intersect_plane(ray, trajectory_plane)
                errors.append(np.linalg.norm(final_pos - test_positions[i]))
                final_3d_pos.append(final_pos)

          #  print(errors)
        #    print(sum(errors)/(len(errors)))


            f = h5py.File("3d_ball_positions" + "/" + vid[:-5] + "_" + trajectory, "w")
            dset = f.create_dataset(vid[:-4] + "_" + trajectory + ".hdf5", data=final_3d_pos)

   # test_3d_to_2d("/home/mthor9/Schreibtisch/uni/bachelorarbeit/table_tennis_code/table-tennis-trajectory/single_test_vid/test1.mp4", ballpos, 100)
