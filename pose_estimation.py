import csv

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math
from sympy import Point, Line
import copy
import os
import h5py
import pyrr
import time


class intersec_point:
    def __init__(self, pos=[], line1_index=None, line2_index=None):
        self.pos = np.array(pos)
        self.line1_index = line1_index
        self.line2_index = line2_index

    def pos_tuple(self):
        return (int(self.pos[0]), int(self.pos[1]))


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


def num_pixels_in_range(img, lower_bound, upper_bound, img_width, img_height):
    counter = 0
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    for i in range(img_height):
        for j in range(img_width):
            if hsv[i][j][1] > 30:
                if (hsv[i][j][0] > lower_bound[0]) and (hsv[i][j][0] < upper_bound[0]):
                    counter += 1
    cv.count

    return counter

# set color bounds based on if green or blue is more present in the center of the image
def set_color_range(img, img_width, img_height):
    blue_counter = 0
    green_counter = 0
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    for i in range(int(img_height / 4), int(img_height / 4 * 3)):
        for j in range(int(img_width / 4), int(img_width / 4 * 3)):
            if hsv[i][j][1] > 30:
                if (hsv[i][j][0] > 90) and (hsv[i][j][0] < 135):
                    blue_counter += 1
                elif (hsv[i][j][0] > 20) and (hsv[i][j][0] < 75):
                    green_counter += 1

    if blue_counter >= green_counter:
        lower_bound = np.array([90, 30, 90])
        upper_bound = np.array([135, 255, 255])
        color = "blue"
    else:
        lower_bound = np.array([40, 30, 90])
        upper_bound = np.array([75, 255, 255])
        color = "green"

    return lower_bound, upper_bound


# return mask of img after applying a filter of given color ranges
def color_filter(img, lower, upper):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower, upper)
    return mask


# reduce noise in a grayscaled img by removing and filling out of noise
def noise_reduction(gray_img):
    blurred_img = gray_img
    for i in range(5):
        blurred_img = cv.GaussianBlur(blurred_img, (5, 5), 0)
        mask = cv.inRange(blurred_img, 100, 255)
        blurred_img = cv.bitwise_and(blurred_img, blurred_img, mask=mask)
    return blurred_img


# use hough transformation with decreasing minimum number of points to detect lines in a given edge img until 4 edges are found
# the lines are further sorted out if a similar line already exists or if there is no other line with roughle the same angle
def get_defining_lines(edge_img, fraction, img_width, img_height):
    final_lines = []
    reduced_lines = []
    lines = cv.HoughLines(edge_img, 1, np.pi / 180 / 2, int(img_height / fraction), None, 0, 0)
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

            # choose new points describing the lines, placed at the edges of the img
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

            # append list of remaining lines and further sort out lines that are completely in the top third of the image
            if new: #and ((new_pt1[1] > img_height / 3) or (new_pt2[1] > img_height / 3)):
                reduced_lines.append(line(new_pt1, new_pt2, theta))


        for i in range(len(reduced_lines)):
            for j in range(len(reduced_lines)):
                if True: #not(i == j) and abs(reduced_lines[i].angle - reduced_lines[j].angle) < 10/180*math.pi:
                    final_lines.append(reduced_lines[i])
                    break


    return final_lines


# takes 2 lines as input and returns their intersection as an int tuple
def get_single_intersection(line1, line2, img_width, img_height):
    l1 = Line(Point(line1.pos1[0], line1.pos1[1]), Point(line1.pos2[0], line1.pos2[1]))
    l2 = Line(Point(line2.pos1[0], line2.pos1[1]), Point(line2.pos2[0], line2.pos2[1]))
    p = l1.intersection(l2)

    if (p != [] and (p[0].x >= 0) and (p[0].y >= 0) and (p[0].x <= img_width) and (p[0].y <= img_height)):
        return (intersec_point(np.array([int(p[0].x), int(p[0].y)])))
    else:
        return intersec_point()

def get_single_intersection_1(line1, line2, img_width, img_height):
    da = line1.pos2 - line1.pos1
    db = line2.pos2 - line2.pos1
    dp = line1.pos1 - line2.pos1

    dap = [0,0]
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
            candidate_point = get_single_intersection_1(lines[i], lines[j], img_width, img_height)

            if len(candidate_point.pos) == 2:
                intersection_points.append(intersec_point(candidate_point.pos, i, j))

    return intersection_points


def draw_points(points, img, text_pos, color):
    new_points = []
    if not isinstance(points[0], intersec_point):
        for i in range(len(points)):
            new_points.append(intersec_point(points[i]))
    else:
        new_points = points


    if text_pos == "above":
        text_y_change = 15
    else:
        text_y_change = -15

    for i in range(len(points)):
        pointed_img = cv.circle(img, new_points[i].pos_tuple(), radius=3, color=color, thickness=-1)
      #  print(new_points[i].pos_tuple())
       # print((new_points[i].pos[0], new_points[i].pos[1]+text_y_change))
        cv.putText(img, str(i), (int(new_points[i].pos[0]), int(new_points[i].pos[1]+text_y_change)), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return pointed_img


def draw_lines(lines, img):
    line_img = copy.copy(img)
    for line in lines:
        cv.line(line_img, line.pos1_tuple(), line.pos2_tuple(), (255, 0, 0), 3, cv.LINE_AA)
    return line_img


# displays all images in an array one after another
def show_images(images):
    for image in images:
        cv.imshow("img", cv.cvtColor(image, cv.COLOR_RGB2BGR))
        cv.waitKey(0)


# determine the min and max points of both axis out of a given set of points
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
        if not(i in indices) and (x_arr[i]> max_x):
            max_x = x_arr[i]
            max_x_index = i
    indices.append(max_x_index)

    for i in range(4):
        if not(i in indices) and (y_arr[i]< min_y):
            min_y = y_arr[i]
            min_y_index = i
    indices.append(min_y_index)

    for i in range(4):
        if not(i in indices) and (y_arr[i]> max_y):
            max_y = y_arr[i]
            max_y_index = i
    indices.append(max_y_index)

    for i in indices:
        corners.append(points[i])

    return corners


# out of all contours only return the 2 with maximal area enclosed
# if the second largest area is much smaller than min_second_size times the largest area
# it is assumed that the table was not split in half by contours and therefore only the largest will be returned
def get_table_contours(contours, min_second_size, img_width, img_height):
    max1 = [0, -1];
    max2 = [0, 0];
    for i in range(len(contours)):
        area = cv.contourArea(contours[i])
        contour_bounds = get_contour_bounds(contours[i])
        if not((contour_bounds.min_x > img_width/2) or (contour_bounds.min_y > img_height/2) or (contour_bounds.max_x < img_width/2)  or (contour_bounds.max_y < img_height/2)):
            if area > max1[0]:
                max1 = [area, i]
            elif area > max2[0]:
                max2 = [area, i]

    table_contours = []

    if max1[1] == -1:
        min = [10000, -1]
        for i in range(len(contours)):
            x = min(abs(contour_bounds.min_x - img_width/2), abs(contour_bounds.min_y - img_height/2), abs(contour_bounds.max_x - img_width/2), abs(contour_bounds.max_y - img_height/2))
            if x < min[0]:
                min = [x, i]
        table_contours.append(max1[1])
    else:
        table_contours.append(contours[max1[1]])
        if (max1[0] * min_second_size < max2[0]) or (len(contours) == 1):
            table_contours.append(contours[max2[1]])

    return table_contours


# creates a black image with white filled given contours
def fill_table_contours(contour_img, img_height):
    im_floodfill = contour_img.copy()
    mask = np.zeros((img_height + 2, img_width + 2), np.uint8)
    cv.floodFill(im_floodfill, mask, (0, 0), 255);
    im_floodfill_inv = cv.bitwise_not(im_floodfill)
    filled_img = contour_img | im_floodfill_inv
    filled_img = cv.morphologyEx(filled_img, cv.MORPH_CLOSE, kernel)
    return filled_img


# calculate the x and y span in which the contours are placed
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

        corner_candidate_1 = get_single_intersection_1(lines[possibly_wrong_corners[0].line2_index],
                                                     lines[possibly_wrong_corners[1].line1_index])
        corner_candidate_2 = get_single_intersection_1(lines[possibly_wrong_corners[0].line1_index],
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


# draw a rotated coordinate frame in an img
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


def max_error(corners, points_3D, rvec, tvec, camera_matrix, dist_coeffs,img):
    projected_corners = []
    for point in points_3D:
        projection, _ = (cv.projectPoints(point, rvec, tvec, camera_matrix, dist_coeffs))
        projected_corners.append(np.array([projection[0][0][0], projection[0][0][1]], dtype="double"))
       # cv.circle(img, (int(projection[0][0][0]), int(projection[0][0][1])), 3, (0, 255, 0), -1)

    maxi = 0.
    error_sum = 0
    for i in range(4):
        error_sum += np.linalg.norm(projected_corners[i] - corners[i].pos)
        maxi = max(np.linalg.norm(projected_corners[i] - corners[i].pos), maxi)

    draw_points(projected_corners, img, "below", (0,255,0))

    return maxi

def min_dist(corners):
    min_dist = 10000
    for i in range(0, len(corners)-1):
        for j in range(i+1, len(corners)):
            min_dist = min(min_dist, np.linalg.norm(corners[i].pos-corners[j].pos))
    return min_dist


def reorder_corners(corners):
    corner_0_lines = [corners[0].line1_index, corners[0].line2_index]
    new_corners = [corners[0],0,0,0]
   # print(corner_0_lines)
    x = 2
    for i in range(1,4):
      #  print(corners[i].line1_index, corners[i].line2_index)
        if not((corners[i].line1_index in corner_0_lines) or (corners[i].line2_index in corner_0_lines)):
            new_corners[1] = corners[i]
     #   elif x > 3:
      #      new_corners[1] = corners[1]
        else:
            new_corners[x] = corners[i]
            x+=1

    for i in range(4):
        if new_corners[i] == 0:
            new_corners[i] = new_corners[0]



    return new_corners

def get_involved_lines(corners):
    index_list= []
    for c in corners:
        index_list.append(c.line1_index)
        index_list.append(c.line2_index)
    return set(index_list)

def distort_image(img, cam_mat, distortion_dist_coeffs, img_width, img_height):
    new_cam_mat, roi = cv.getOptimalNewCameraMatrix(cam_mat, distortion_dist_coeffs, (img_width, img_height), 1, (img_width, img_height))
    dst = cv.undistort(img, cam_mat, distortion_dist_coeffs, None, new_cam_mat)
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    return dst

def estimate_pose(vid_path, scene_list_path):
    frame_indices = []
    frame_poses = []
    scene_poses = []

    video = cv.VideoCapture(vid_path)
    success, candidate_img = video.read()
    img = candidate_img
    frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))

    # get img size
    img_width = img.shape[1]
    img_height = img.shape[0]

    lower_bound, upper_bound = set_color_range(img, img_width, img_height)

    cut_frames = []

    with open(scene_list_path) as scene_file:
        file_reader = csv.reader(scene_file, delimiter=",")
        i = 0
        for row in file_reader:
            i+=1
            if i > 2:
                cut_frames.append(row[4])

    cut_frames = list(map(int, cut_frames))
    cut_frames.append(int(frame_count))
    start_frame = 1
    frame = 1
    chosen_frame_index = 0

    for i in range(len(cut_frames)):
        max_pixels_in_bound = 0

        if (cut_frames[i] - start_frame) < 40:
            while frame < cut_frames[i]:
                frame+=1
                success, candidate_img = video.read()
                if frame == int(start_frame+(cut_frames[i]-start_frame)/2):
                    img = candidate_img
                    chosen_frame_index = frame

        else:
            while frame < cut_frames[i]:
                frame += 1
                success, candidate_img = video.read()

                if (frame > start_frame + 30) and (frame < cut_frames[i]-10):
                    num_pixels_in_range = cv.countNonZero(color_filter(candidate_img, lower_bound, upper_bound))
                    if (frame % 1 == 0) and (num_pixels_in_range > max_pixels_in_bound):
                        max_pixels_in_bound = num_pixels_in_range
                        img = candidate_img
                        chosen_frame_index = frame
       # print(2)

        camera_matrix = np.array([(img_width, 0, img_width / 2),
                                  (0, img_height, img_height / 2),
                                  (0, 0, 1)])
        img = distort_image(img, camera_matrix, np.array([[-1], [0], [0], [0]]), img_width, img_height)

        frame_indices.append(chosen_frame_index)
        start_frame = cut_frames[i]
       # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # process img and apply all transformations
        mask = color_filter(img, lower_bound, upper_bound)
        kernel = np.ones((5, 5), np.uint8)
        closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        contours, hierarchy = cv.findContours(closing, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        table_contours = get_table_contours(contours, 1 / 4, img_width, img_height)
        contour_img = cv.drawContours(np.zeros((img_height, img_width), dtype="uint8"), table_contours, -1, 255, 1)
        bounding_box = get_contour_bounds(contour_img)

        hough_denominator = 10
        max_hough_denominator = 25
        fix_successful = True
        all_edges_reached = False
        points_found = False
       # show_images([cv.drawContours(np.zeros((img_height, img_width), dtype="uint8"), contours, -1, 255, 1)])

        translation_vector = [0,0,0]
        rotation_vector = [0,0,0]



        while hough_denominator <= max_hough_denominator:
            hough_denominator += 2
            defining_lines = get_defining_lines(contour_img, hough_denominator, img_width, img_height)

            #stop if there are too many lines(no hope for good solution and long computation time)
            if len(defining_lines) > 5:
                break

          #  print(len(defining_lines))

            line_img = draw_lines(defining_lines, img)
            intersections = get_intersections(defining_lines, img_width, img_height)

            #in_bound_points, all_edges_reached = get_possible_points(intersections, bounding_box, img_height)

            show_images([line_img])


            if len(intersections) >= 4: #(all_edges_reached == True):
                corners = get_corners(intersections)
              #  print(corners)
              #  show_images([line_img])
                corners = reorder_corners(corners)
                draw_points(corners, img, "above", (255,0,0))

                #for i in range(2):
                #corners, fix_successful = fix_occlusions(corners, in_bound_points, defining_lines)



                points_2D = np.array([corners[0].pos_tuple(),
                                      corners[1].pos_tuple(),
                                      corners[2].pos_tuple(),
                                      corners[3].pos_tuple()], dtype="double")

                points_3D = np.array([(137, -76.25, 0),
                                      (-137, 76.25, 0),
                                      (-137, -76.25, 0),
                                      (137, 76.25, 0)], dtype="double")

                points_2D_ = np.array([[corners[0].pos_tuple(), corners[1].pos_tuple(), corners[2].pos_tuple(), corners[3].pos_tuple()]], dtype="float32")

                points_3D_ = np.array([[(137, -76.25, 0), (-137, 76.25, 0), (-137, -76.25, 0), (137, 76.25, 0)]] , dtype="float32")


                # make empty distance coefficients and estimate camera matrix
                dist_coeffs = np.zeros((4, 1))
               # retval, camera_matrix, dist_coeffs, rvecs, tvecs= cv.calibrateCamera(points_3D_, points_2D_, (img_height, img_width), None, None)
               # print(camera_matrix, dist_coeffs)
                print(dist_coeffs)




               # ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(points_3D_, points_2D_,(img_height, img_width), None ,None)


                show_images([img])

                # estimate rotation vector and translation vector
                success, rotation_vector, translation_vector, inliers = cv.solvePnPRansac(points_3D, points_2D, camera_matrix,
                                                                                              dist_coeffs, flags=cv.SOLVEPNP_P3P)
                rotation_mat, _ = cv.Rodrigues(rotation_vector)
                print(rotation_mat)
                print(np.array([[0],[0],[1]]))
                print(np.matmul(rotation_mat, np.array([[0],[0],[1]])))
                if np.matmul(rotation_mat, np.array([[0],[0],[1]]))[1] > 0:
                    m = [[1,0,0],[0,-1,0],[0,0,-1]]
                    new_rotation_mat = np.matmul(rotation_mat, m)
                    rotation_vector,_ = cv.Rodrigues(new_rotation_mat)
                print(rotation_vector, translation_vector)


                get_involved_lines(corners)

                max_err = max_error(corners, points_3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs, img)

              #  draw_points(points_2D, img, "above", (0,255,0))
                print(translation_vector)

                draw_coordinate_frame(img, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                show_images([img])

                #check if the projection is reasonable
                if (max_err < img_height/15) and (min_dist(corners) > img_height/10) and (len(get_involved_lines(corners)) == 4):
                    for p in points_2D:
                        cv.circle(img, (int(p[0]), int(p[1])), 3, (255, 0, 0), -1)

                    draw_coordinate_frame(img, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                    points_found = True
                   # show_images([img, line_img])

                    break

       # show_images([cv.cvtColor(img, cv.COLOR_BGR2RGB)])
        scene_poses.append([translation_vector, rotation_vector, corners, points_found])

    num_bad_estimations = 0
    #look for possible fixes of frames that could not get detected properly
    for i in range(len(scene_poses)):
        max_similar_points = 0
        if scene_poses[i][3] == False:
            for pose2 in scene_poses:
                if pose2[3] == True:
                    similar_points = 0
                    for j in range(4):
                        if (np.linalg.norm(scene_poses[i][2][j].pos-pose2[2][j].pos) < img_height/40):
                            similar_points += 1
                    if similar_points > max_similar_points:
                        max_similar_points = similar_points
                        candidate_pose = pose2
                       # scene_poses[i][3] == True
        if max_similar_points > 0:
            scene_poses[i] = candidate_pose
        if scene_poses[i][3] == False:
            num_bad_estimations+=1

  #  print(len(scene_poses))
 #   print(num_bad_estimations)

    start_frame = 0

    for i in range(len(cut_frames)):
        for j in range(start_frame, cut_frames[i]):
            frame_poses.append(scene_poses[i])
        start_frame = cut_frames[i]

    final_poses = []
    for i in range(len(frame_poses)):
        final_poses.append(np.concatenate((np.transpose(frame_poses[i][0])[0], np.transpose(frame_poses[i][1])[0], np.array([int(frame_poses[i][3]), img_height, img_width]))))

    return final_poses, frame_indices, camera_matrix, dist_coeffs

def create_dataset(dir, filename, data):
    f = h5py.File(dir + "/" + filename + ".hdf5", "w")
    dset = f.create_dataset(filename, data=data)

def show_poses(poses, indices, vid_path, camera_matrix, dist_coeffs):
    frame = 0

    vid = cv.VideoCapture(vid_path)
    for i in range(len(indices)):
        while True:
            success, img = vid.read()
            frame+=1
            if frame == indices[i]:
                print(poses[frame][6])
              #  print(frame)

                draw_coordinate_frame(img, np.array([poses[frame][3], poses[frame][4], poses[frame][5]]), np.array([poses[frame][0], poses[frame][1], poses[frame][2]]), camera_matrix, dist_coeffs)
                show_images([img])
                break



if __name__ == "__main__":
    scene_list_dir = "scene_lists"
    video_dir = "single_test_vid"
    pose_estimation_dir = "pose_estimates"
    start_time = time.time()

    for vid in os.listdir(video_dir):
        pose_list, frame_indices, cam_mat, dist_coeffs = estimate_pose(video_dir + "/" + vid, scene_list_dir + "/" + vid[:-4] + ".csv")
        show_poses(pose_list, frame_indices, video_dir + "/" + vid, cam_mat, dist_coeffs)
        create_dataset(pose_estimation_dir, vid[:-4], pose_list)
        break
