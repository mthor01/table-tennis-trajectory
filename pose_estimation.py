# this script estimates camera poses for every frame in every video from the video directory
# these poses are stored in the pose estimates directory
# executing this script will first delete everything that was previously stored in the estimates directory
# use show_images(<array_of_images>) at any point to look at desired images

import csv
import numpy as np
import cv2 as cv
import math
import copy
import os
import h5py
from classes import intersec_point, line, bounds

scene_list_dir = "scene_lists"
video_dir = "single_test_vid"
pose_estimation_dir = "pose_estimates"


# returns the amount of pixels in a given color range
def num_pixels_in_range(img, lower_bound, upper_bound):
    counter = 0
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if hsv[i][j][1] > 30:
                if (hsv[i][j][0] > lower_bound[0]) and (hsv[i][j][0] < upper_bound[0]):
                    counter += 1
    cv.count

    return counter


# return color bounds for green or blue based on which color is more present in the center of the image
def set_color_range(img):
    blue_counter = 0
    green_counter = 0
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    for i in range(int(img.shape[0] / 4), int(img.shape[0] / 4 * 3)):
        for j in range(int(img.shape[1] / 4), int(img.shape[1] / 4 * 3)):
            if hsv[i][j][1] > 30:
                if (hsv[i][j][0] > 90) and (hsv[i][j][0] < 135):
                    blue_counter += 1
                elif (hsv[i][j][0] > 20) and (hsv[i][j][0] < 75):
                    green_counter += 1

    if blue_counter >= green_counter:
        lower_bound = np.array([90, 30, 90])
        upper_bound = np.array([135, 255, 255])
    else:
        lower_bound = np.array([40, 30, 90])
        upper_bound = np.array([75, 255, 255])

    return lower_bound, upper_bound


# return mask of img after applying a filter of given color ranges
def color_filter(img, lower_bound, upper_bound):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_bound, upper_bound)

    return mask


# apply hough transformation on an edge image with a defined sensitivity(higher sensitivity => more lines)
# the lines are further sorted out if a similar line already exists or if there is no other line with roughle the same angle
def get_defining_lines(edge_img, sensitivity):
    img_width = edge_img.shape[1]
    img_height = edge_img.shape[0]
    final_lines = []
    reduced_lines = []

    lines = cv.HoughLines(edge_img, 1, np.pi / 180 / 2, int(img_height / sensitivity), None, 0, 0)
    if lines is not None:
        for i in range(0, len(lines)):
            new = True
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = np.array([int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))])
            pt2 = np.array([int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))])
            line_dir = pt2 - pt1
            line_dir_x = np.sign(line_dir[0]) * max(abs(line_dir[0]), 0.001)
            line_dir_y = np.sign(line_dir[1]) * max(abs(line_dir[1]), 0.001)

            # counter div by zero with small values
            if line_dir_x == 0:
                line_dir_x = 0.00001
            if line_dir_y == 0:
                line_dir_y = 0.00001

            # get crossing points of the lines with both axis
            new_pt1_x = pt1 + line_dir * (-pt1[0] / line_dir_x)
            new_pt1_y = pt1 + line_dir * (-pt1[1] / line_dir_y)

            # choose new points,placed at the edges of the img, describing the lines
            if np.linalg.norm(new_pt1_x) < np.linalg.norm(new_pt1_y):
                new_pt1 = new_pt1_x
                new_pt2 = new_pt1 + line_dir * (img_width / line_dir_x)
            else:
                new_pt1 = new_pt1_y
                new_pt2 = new_pt1 + line_dir * (img_height / line_dir_y)

            # sort out lines that are very similar
            for l in reduced_lines:
                l_pt1 = l.pos1
                l_pt2 = l.pos2

                if np.linalg.norm(new_pt1 - l_pt1) > np.linalg.norm(new_pt1 - l_pt2):
                    l_pt1, l_pt2 = l_pt2, l_pt1

                dist_pt1 = np.linalg.norm(new_pt1 - l_pt1)
                dist_pt2 = np.linalg.norm(new_pt2 - l_pt2)

                if ((dist_pt1 < img_width / 20) and (dist_pt2 < img_width / 20)):
                    new = False
                    break

            if new:
                reduced_lines.append(line(new_pt1, new_pt2, theta))

    return reduced_lines


# returns the intersection point of 2 lines if is is inside of the image range (if not returns empty intersec_point)
def get_single_intersection(line1, line2, img_width, img_height):
    da = line1.pos2 - line1.pos1
    db = line2.pos2 - line2.pos1
    dp = line1.pos1 - line2.pos1

    dap = [0, 0]
    dap[0] = -da[1]
    dap[1] = da[0]
    denom = np.dot(dap, db)

    if denom == 0:
        return intersec_point()
    num = np.dot(dap, dp)
    p = (num / denom) * db + line2.pos1

    if (p != [] and (p[0] >= 0) and (p[1] >= 0) and (p[0] <= img_width) and (p[1] <= img_height)):
        return (intersec_point(np.array([int(p[0]), int(p[1])])))
    else:
        return intersec_point()


# calculate the intersection points of given lines and store their belonging lines indices
def get_intersections(lines, img_width, img_height):
    intersection_points = []

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            candidate_point = get_single_intersection(lines[i], lines[j], img_width, img_height)

            if len(candidate_point.pos) == 2:
                intersection_points.append(intersec_point(candidate_point.pos, i, j))

    return intersection_points


# draws points on an image and attaches a number above or below them
def draw_points(points, img, text_position="above", color=[255, 0, 0]):
    new_points = []

    if not isinstance(points[0], intersec_point):
        for i in range(len(points)):
            new_points.append(intersec_point(points[i]))
    else:
        new_points = points

    if text_position == "above":
        text_y_change = 15
    else:
        text_y_change = -15

    for i in range(len(points)):
        pointed_img = cv.circle(img, new_points[i].pos_tuple(), radius=3, color=color, thickness=-1)
        cv.putText(img, str(i), (int(new_points[i].pos[0]), int(new_points[i].pos[1] + text_y_change)),
                   cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return pointed_img


def draw_lines(lines, img):
    line_img = copy.copy(img)
    for line in lines:
        cv.line(line_img, line.pos1_tuple(), line.pos2_tuple(), (255, 0, 0), 3, cv.LINE_AA)
    return line_img


# displays all images from an array of images one after another
def show_images(images):
    for image in images:
        cv.imshow("img", cv.cvtColor(image, cv.COLOR_RGB2BGR))
        cv.waitKey(0)


# determine the min and max points of both axis out of a given set of points
# a point cant be chosen twice
def get_corners(points):
    x_arr = []
    y_arr = []
    corners = []

    for point in points:
        x_arr.append(point.pos[0])
        y_arr.append(point.pos[1])

    indices = []

    max_x, max_x_index = 0, 0
    min_y, min_y_index = 10000, 0
    max_y, max_y_index = 0, 0
    indices.append(x_arr.index(min(x_arr)))

    for i in range(4):
        if not (i in indices) and (x_arr[i] > max_x):
            max_x = x_arr[i]
            max_x_index = i
    indices.append(max_x_index)

    for i in range(4):
        if not (i in indices) and (y_arr[i] < min_y):
            min_y = y_arr[i]
            min_y_index = i
    indices.append(min_y_index)

    for i in range(4):
        if not (i in indices) and (y_arr[i] > max_y):
            max_y = y_arr[i]
            max_y_index = i
    indices.append(max_y_index)

    for i in indices:
        corners.append(points[i])

    return corners


# out of all contours only return the 2 with maximal area enclosed which bounding boxes cover the center point of the image
# if the second largest area is much smaller than min_second_size times the largest area, it is assumed that
# the table was not split in half by contours and therefore only the contours of the largest area will be returned
# if there is no area that covers the center point, the one closest to the center point will be returned
def get_table_contours(contours, min_second_size, img_width, img_height):
    max1 = [0, -1]
    max2 = [0, 0]

    for i in range(len(contours)):
        area = cv.contourArea(contours[i])
        contour_bounds = get_contour_bounds(contours[i])

        if not ((contour_bounds.min_x > img_width / 2) or (contour_bounds.min_y > img_height / 2) or (
                contour_bounds.max_x < img_width / 2) or (contour_bounds.max_y < img_height / 2)):
            if area > max1[0]:
                max1 = [area, i]
            elif area > max2[0]:
                max2 = [area, i]

    table_contours = []

    if max1[1] == -1:
        min = [10000, -1]
        for i in range(len(contours)):
            x = min(abs(contour_bounds.min_x - img_width / 2), abs(contour_bounds.min_y - img_height / 2),
                    abs(contour_bounds.max_x - img_width / 2), abs(contour_bounds.max_y - img_height / 2))
            if x < min[0]:
                min = [x, i]

        table_contours.append(max1[1])

    else:
        table_contours.append(contours[max1[1]])

        if (max1[0] * min_second_size < max2[0]) or (len(contours) == 1):
            table_contours.append(contours[max2[1]])

    return table_contours


# calculate the x and y span in which the contours are placed and store them as a bounds object
def get_contour_bounds(contour):
    x, y, w, h = cv.boundingRect(contour)

    return bounds(x, x + w, y, y + h)


# sort out points that are guranteed to be no table corners
# returns candidate points for table corners and a bool,
# indicating if at least one point is close to every span extremum
def get_possible_points(points, bounds, img_height):
    points_in_bounds = []
    allowed_error = img_height / 20
    boarders_reached_check = [0, 0, 0, 0]

    for point in points:

        contact_to_boarder = False

        in_x_bounds = (point.pos[0] > bounds.min_x - allowed_error) and (point.pos[0] < bounds.max_x + allowed_error)
        in_y_bounds = (point.pos[1] > bounds.min_y - allowed_error) and (point.pos[1] < bounds.max_y + allowed_error)

        # check if a point lies in a given span and allows some error
        if in_x_bounds and in_y_bounds:
            if abs(point.pos[0] - bounds.min_x) < allowed_error:
                boarders_reached_check[0] = 1
                contact_to_boarder = True
            elif abs(point.pos[0] - bounds.max_x) < allowed_error:
                boarders_reached_check[1] = 1
                contact_to_boarder = True
            elif abs(point.pos[1] - bounds.min_y) < allowed_error:
                boarders_reached_check[2] = 1
                contact_to_boarder = True
            elif abs(point.pos[1] - bounds.max_y) < allowed_error:
                boarders_reached_check[3] = 1
                contact_to_boarder = True

        if contact_to_boarder == True:
            points_in_bounds.append(point)

    return points_in_bounds, sum(boarders_reached_check) == 4


# takes possible corners and tries to fix wrong ones created by occlusion
def fix_occlusions(corners, in_bound_points, lines):
    line_hits = [0] * len(lines)
    corner_indices = []
    for corner in corners:
        line_hits[corner.line1_index] += 1
        line_hits[corner.line2_index] += 1

    if (line_hits.count(1) == 0) and (line_hits.count(2) == 4):
        return corners, True

    elif (line_hits.count(1) == 2):
        possibly_wrong_corners = []
        one_hit_lines = []
        for i in range(len(line_hits)):
            if line_hits[i] == 1:
                one_hit_lines.append(i)

        for i in range(len(corners)):
            if (corners[i].line1_index in one_hit_lines):
                possibly_wrong_corners.append(corners[i])
                corner_indices.append(i)
            if (corners[i].line2_index in one_hit_lines):
                possibly_wrong_corners.append(
                    intersec_point(corners[i].pos, corners[i].line2_index, corners[i].line1_index))
                corner_indices.append(i)

        corner_candidate_1 = get_single_intersection(lines[possibly_wrong_corners[0].line2_index],
                                                     lines[possibly_wrong_corners[1].line1_index])
        corner_candidate_2 = get_single_intersection(lines[possibly_wrong_corners[0].line1_index],
                                                     lines[possibly_wrong_corners[1].line2_index])

        candidate_1_dist = np.linalg.norm([corner_candidate_1.pos - possibly_wrong_corners[0].pos])
        candidate_2_dist = np.linalg.norm([corner_candidate_2.pos - possibly_wrong_corners[1].pos])

        final_corners = corners

        if candidate_1_dist > candidate_2_dist:
            final_corners[corner_indices[1]] = corner_candidate_2
            return final_corners, True
        else:
            final_corners[corner_indices[0]] = corner_candidate_1
            return final_corners, True

    else:
        return [], False


# draw a coordinate frame in an img
def draw_coordinate_frame(img, rvec, tvec, camera_matrix, dist_coeffs):
    orig = np.array([[0], [0], [0.0001]])
    x_vec = np.array([[1], [0], [0]]) * 137.0
    y_vec = np.array([[0], [1], [0]]) * 76.25
    z_vec = np.array([[0], [0], [1]]) * 76.25

    orig, _ = cv.projectPoints(orig, rvec, tvec, camera_matrix, dist_coeffs)
    x_vec, _ = cv.projectPoints(x_vec, rvec, tvec, camera_matrix, dist_coeffs)
    y_vec, _ = cv.projectPoints(y_vec, rvec, tvec, camera_matrix, dist_coeffs)
    z_vec, _ = cv.projectPoints(z_vec, rvec, tvec, camera_matrix, dist_coeffs)

    cv.line(img, (int(orig[0][0][0]), int(orig[0][0][1])), (int(x_vec[0][0][0]), int(x_vec[0][0][1])), (255, 0, 0), 2)
    cv.line(img, (int(orig[0][0][0]), int(orig[0][0][1])), (int(y_vec[0][0][0]), int(y_vec[0][0][1])), (255, 0, 0), 2)
    cv.line(img, (int(orig[0][0][0]), int(orig[0][0][1])), (int(z_vec[0][0][0]), int(z_vec[0][0][1])), (0, 0, 255), 2)

# returns the 2d projection of 3d points based on a cameras position
def project_points_on_image(points_3D, rvec, tvec, camera_matrix, dist_coeffs):
    projected_corners = []
    for point in points_3D:
        projection, _ = (cv.projectPoints(point, rvec, tvec, camera_matrix, dist_coeffs))
        projected_corners.append(np.array([projection[0][0][0], projection[0][0][1]], dtype="double"))

    return projected_corners

# returns maximum euclidean error between 2 points of same index
def max_error(corners, projected_corners):
    maxi = 0.
    error_sum = 0

    for i in range(4):
        error_sum += np.linalg.norm(projected_corners[i] - corners[i].pos)
        maxi = max(np.linalg.norm(projected_corners[i] - corners[i].pos), maxi)

    return maxi

# returns the minimum euclidean distance between any pair of points
def min_dist(points):
    min_dist = 10000
    for i in range(0, len(points) - 1):
        for j in range(i + 1, len(points)):
            min_dist = min(min_dist, np.linalg.norm(points[i].pos - points[j].pos))
    return min_dist

# reorders the corners such that the first and the second point have maximum distance
def reorder_corners(corners):
    corner_0_lines = [corners[0].line1_index, corners[0].line2_index]
    new_corners = [corners[0], 0, 0, 0]
    x = 2

    for i in range(1, 4):
        if not ((corners[i].line1_index in corner_0_lines) or (corners[i].line2_index in corner_0_lines)):
            new_corners[1] = corners[i]
        else:
            if x > 3:
                new_corners[1] = corners[i]
            else:
                new_corners[x] = corners[i]
                x += 1

    for i in range(4):
        if new_corners[i] == 0:
            new_corners[i] = new_corners[0]

    return new_corners

# returns all line indices of lines that go through given intersection points
def get_involved_lines(intersec_points):
    index_list = []
    for c in intersec_points:
        index_list.append(c.line1_index)
        index_list.append(c.line2_index)
    return set(index_list)

# distorts a given image based on distortion coefficients
def distort_image(img, cam_mat, distortion_dist_coeffs):
    img_height= img.shape[0]
    img_width = img.shape[1]
    new_cam_mat, roi = cv.getOptimalNewCameraMatrix(cam_mat, distortion_dist_coeffs, (img_width, img_height), 1,(img_width, img_height))
    dst = cv.undistort(img, cam_mat, distortion_dist_coeffs, None, new_cam_mat)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]

    return dst

# estimates the camera poses of every frame in a video
def estimate_poses(vid_path, scene_list_path):
    frame_indices = []
    frame_poses = []
    scene_poses = []
    cut_frames = []


    video = cv.VideoCapture(vid_path)
    success, candidate_img = video.read()
    img = candidate_img
    frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))

    # get img size
    img_height = img.shape[0]
    img_width = img.shape[1]

    # assume no distortion and estimate camera matrix
    dist_coeffs = np.zeros((4, 1))

    camera_matrix = np.array([(img_width, 0, img_width / 2),
                              (0, img_height, img_height / 2),
                              (0, 0, 1)])

    # set 3d corner coordinates of the table without camera pose
    points_3D = np.array([(137, -76.25, 0),
                          (-137, 76.25, 0),
                          (-137, -76.25, 0),
                          (137, 76.25, 0)], dtype="double")

    #set color ranges based on table color
    lower_bound, upper_bound = set_color_range(img)

    # read and store given cut frames
    with open(scene_list_path) as scene_file:
        file_reader = csv.reader(scene_file, delimiter=",")
        i = 0
        for row in file_reader:
            i += 1
            if i > 2:
                cut_frames.append(row[4])

    cut_frames = list(map(int, cut_frames))
    cut_frames.append(int(frame_count))

    #for every scene, choose a frame with low amounts of occlusion and estimate a camera pose
    last_cut_frame = 1
    frame = 1
    best_frame_index = 0

    for i in range(len(cut_frames)):
        max_pixels_in_bound = 0

        if (cut_frames[i] - last_cut_frame) < 40:
            while frame < cut_frames[i]:
                frame += 1
                success, candidate_img = video.read()
                if frame == int(last_cut_frame + (cut_frames[i] - last_cut_frame) / 2):
                    img = candidate_img
                    best_frame_index = frame

        else:
            while frame < cut_frames[i]:
                frame += 1
                success, candidate_img = video.read()

                if (frame > last_cut_frame + 30) and (frame < cut_frames[i] - 10):
                    num_pixels_in_range = cv.countNonZero(color_filter(candidate_img, lower_bound, upper_bound))
                    if (frame % 1 == 0) and (num_pixels_in_range > max_pixels_in_bound):
                        max_pixels_in_bound = num_pixels_in_range
                        img = candidate_img
                        best_frame_index = frame


        #img = distort_image(img, camera_matrix, np.array([[-1], [0], [0], [0]]), img_width, img_height)

        frame_indices.append(best_frame_index)
        last_cut_frame = cut_frames[i]

        # process img such that an image showing the contours of the table on a black background is created
        mask = color_filter(img, lower_bound, upper_bound)
        kernel = np.ones((5, 5), np.uint8)
        closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        contours, hierarchy = cv.findContours(closing, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        table_contours = get_table_contours(contours, 1 / 4, img_width, img_height)
        contour_img = cv.drawContours(np.zeros((img_height, img_width), dtype="uint8"), table_contours, -1, 255, 1)

        # apply hough transformation to find the tables edges and calculate the camera pose from their intersection points
        hough_sensitivity = 10
        max_hough_sensitivity = 25
        maximum_lines = 5
        too_many_lines = False
        points_found = False

        while hough_sensitivity <= max_hough_sensitivity:
            if too_many_lines:
                break

            hough_sensitivity += 2
            defining_lines = get_defining_lines(contour_img, hough_sensitivity)

            # stop if there are too many lines(no hope for good solution)
            if len(defining_lines) > maximum_lines:
                too_many_lines = True

            line_img = draw_lines(defining_lines, img)
           # show_images([line_img])
            intersections = get_intersections(defining_lines, img_width, img_height)

            if len(intersections) >= 4:
                corners = get_corners(intersections)
                corners = reorder_corners(corners)
                draw_points(corners, img, "above", (255, 0, 0))

                points_2D = np.array([corners[0].pos_tuple(),
                                      corners[1].pos_tuple(),
                                      corners[2].pos_tuple(),
                                      corners[3].pos_tuple()], dtype="double")

                # estimate pose given by rotation vector and translation vector
                success, rotation_vector, translation_vector = cv.solvePnP(points_3D,
                                                                                          points_2D,
                                                                                          camera_matrix,
                                                                                          dist_coeffs,
                                                                                          flags=cv.SOLVEPNP_ITERATIVE)
                #rotate the prediction if the z axis is pointing downwards
                rotation_mat, _ = cv.Rodrigues(rotation_vector)
                projected_corners = project_points_on_image(points_3D, rotation_vector, translation_vector,
                                                            camera_matrix, dist_coeffs)
                draw_points(projected_corners, img, "below", (0, 255, 0))

                if np.matmul(rotation_mat, np.array([[0], [0], [1]]))[1] > 0:
                    m = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
                    new_rotation_mat = np.matmul(rotation_mat, m)
                    rotation_vector, _ = cv.Rodrigues(new_rotation_mat)

                draw_coordinate_frame(img, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                #show_images([img])

                # check if the pose is reasonable, by reprojecting the given 3D corners on the image
                # and comparing them with the estimated corners
                max_err = max_error(corners, projected_corners)

                if (max_err < img_height / 15) and (min_dist(corners) > img_height / 10) and (len(get_involved_lines(corners)) == 4):
                    for p in points_2D:
                        cv.circle(img, (int(p[0]), int(p[1])), 3, (255, 0, 0), -1)

                    draw_coordinate_frame(img, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                    points_found = True
                    break

        scene_poses.append([translation_vector, rotation_vector, corners, points_found])

    num_bad_estimations = 0
    # look for possible fixes of frames that could not get detected properly
    # this is done by searching for frames for which the prediction worked well
    # if that is the case and the 2 frames have similar corners, the predicted pose is just copied
    for i in range(len(scene_poses)):
        max_similar_points = 0
        if scene_poses[i][3] == False:
            for pose2 in scene_poses:
                if pose2[3] == True:
                    similar_points = 0
                    for j in range(4):
                        if (np.linalg.norm(scene_poses[i][2][j].pos - pose2[2][j].pos) < img_height / 40):
                            similar_points += 1

                    if similar_points > max_similar_points:
                        max_similar_points = similar_points
                        candidate_pose = pose2

        if max_similar_points > 0:
            scene_poses[i] = candidate_pose
        if scene_poses[i][3] == False:
            num_bad_estimations += 1

    last_cut_frame = 0

    # store the estimated poses for every frame
    for i in range(len(cut_frames)):
        for j in range(last_cut_frame, cut_frames[i]):
            frame_poses.append(scene_poses[i])
        last_cut_frame = cut_frames[i]

    final_poses = []
    for i in range(len(frame_poses)):
        final_poses.append(np.concatenate((np.transpose(frame_poses[i][0])[0], np.transpose(frame_poses[i][1])[0],
                                           np.array([int(frame_poses[i][3]), img_height, img_width]))))

    return final_poses, frame_indices, camera_matrix, dist_coeffs


def create_dataset(dir, filename, data):
    f = h5py.File(dir + "/" + filename + ".hdf5", "w")
    f.create_dataset(filename, data=data)


def show_poses(poses, indices, vid_path, camera_matrix, dist_coeffs):
    frame = 0

    vid = cv.VideoCapture(vid_path)
    for i in range(len(indices)):
        while True:
            success, img = vid.read()
            frame += 1
            if frame == indices[i]:
                draw_coordinate_frame(img, np.array([poses[frame][3], poses[frame][4], poses[frame][5]]),
                                      np.array([poses[frame][0], poses[frame][1], poses[frame][2]]), camera_matrix,
                                      dist_coeffs)
                show_images([img])
                break


if __name__ == "__main__":

    for vid in os.listdir(video_dir):
        pose_list, frame_indices, cam_mat, dist_coeffs = estimate_poses(video_dir + "/" + vid, scene_list_dir + "/" + vid[:-4] + ".csv")
       # show_poses(pose_list, frame_indices, video_dir + "/" + vid, cam_mat, dist_coeffs)
        create_dataset(pose_estimation_dir, vid[:-4], pose_list)
        break
