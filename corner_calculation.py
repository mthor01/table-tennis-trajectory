import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import statistics
import math
from sympy import Point, Line
import copy


#set path of img file
#there is a folder with 8 images to try out wich you can acces by changing the number x in x.jpg
file = 'test_images/1.jpg'

# set hsv color range for detection of blue
lower_blue = np.array([90, 30, 90])
upper_blue = np.array([135, 255, 255])



# return mask of img after applying a filter of given color ranges
def color_filter(img, lower, upper):
    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    mask = cv.inRange(hsv, lower, upper)
    return mask

#reduce noise in a grayscaled img by removing and filling out of noise
def noise_reduction(gray_img):
    blurred_img = gray_img
    for i in range(5):
        blurred_img = cv.GaussianBlur(blurred_img, (5, 5), 0)
        mask = cv.inRange(blurred_img, 100, 255)
        blurred_img = cv.bitwise_and(blurred_img, blurred_img, mask=mask)
    return blurred_img

# calulate area of a triangle with points a,b,c
def calc_tri_area(a, b, c):
    return 0.5 * abs(a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))

# use hough transformation with decreasing minimum number of points to detect lines in a given edge img until 4 edges are found
# the lines are further sorted out if a similar line already exists or if there is no other line with roughle the same angle
def get_defining_lines(edge_img):
    final_lines = []
    fraction = 5
    while len(final_lines) < 4:
        fraction += 1
        lines = cv.HoughLines(edge_img, 1, np.pi / 180, int(img_width/fraction), None, 0, 0)
        if lines is not None:
            reduced_lines = []
            final_lines = []
            for i in range(0, len(lines)):
                new = True
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = np.array([int(x0 + 1000*(-b)), int(y0 + 1000*(a))])
                pt2 = np.array([int(x0 - 1000*(-b)), int(y0 - 1000*(a))])
                line_dir = pt2-pt1
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


                #choose new points describing the lines, placed at the edges of the img
                if np.linalg.norm(new_pt1_x) < np.linalg.norm(new_pt1_y):
                    new_pt1 = new_pt1_x
                    new_pt2 = new_pt1 + line_dir * (img_width / line_dir_x)
                else:
                    new_pt1 = new_pt1_y
                    new_pt2 = new_pt1 + line_dir * (img_height / line_dir_y)

                #sort out lines that are very similar
                for l in reduced_lines:
                    l_pt1 = l[0][0]
                    l_pt2 = l[0][1]

                    if np.linalg.norm(new_pt1-l_pt1) > np.linalg.norm(new_pt1-l_pt2):
                        l_pt1, l_pt2 = l_pt2, l_pt1

                    dist_pt1 = np.linalg.norm(new_pt1-l_pt1)
                    dist_pt2 = np.linalg.norm(new_pt2-l_pt2)
                    if ((dist_pt1 < img_width/10) and (dist_pt2 < img_width/10)):
                        new = False
                        break

                # append list of remaining lines and further sort out lines that are completely in the top third of the image
                if (new) and (new_pt1[1]> img_height/3) and (new_pt2[1]> img_height/3):
                    reduced_lines.append([[new_pt1, new_pt2], theta])

            # only add lines to final output that have another almost parallel line present in the img
            for l1 in reduced_lines:
                for l2 in reduced_lines:
                    if (abs(l1[1]-l2[1])*180/math.pi < 25) and (l1[1] != l2[1]):
                        final_lines.append(l1)
                        break

    return final_lines


#calculate the intersection points of given lines
def get_intersections(lines):
    intersection_points = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            l1 = Line(Point(lines[i][0][0][0], lines[i][0][0][1]), Point(lines[i][0][1][0], lines[i][0][1][1]))
            l2 = Line(Point(lines[j][0][0][0], lines[j][0][0][1]), Point(lines[j][0][1][0], lines[j][0][1][1]))
            p = l1.intersection(l2)

            if (p != [] and (p[0].x>=0) and (p[0].y>=0) and (p[0].x<= img_width) and (p[0].y<=img_height)):
                intersection_points.append([int(p[0].x), int(p[0].y)])

    return intersection_points

def draw_points(points, img):
    for point in points:
        pointed_img = cv.circle(img, (point[0], point[1]), radius=3, color=(255,0,0), thickness = -1)
    return img

def draw_lines(lines, img):
    line_img = copy.copy(img)
    for line in lines:
        cv.line(line_img, line[0][0].astype(int), line[0][1].astype(int), (255,0,0), 3, cv.LINE_AA)
    return line_img

def show_images(images):
    for image in images:
        cv.imshow("img", cv.cvtColor(image, cv.COLOR_RGB2BGR))
        cv.waitKey(0)

#determine the min and max points of both axis out of a given set of points
def get_corners(points):
    x_arr = []
    y_arr = []
    corners = []
    for point in points:
        x_arr.append(point[0])
        y_arr.append(point[1])

    corners.append(points[x_arr.index(max(x_arr))])
    corners.append(points[x_arr.index(min(x_arr))])
    corners.append(points[y_arr.index(max(y_arr))])
    corners.append(points[y_arr.index(min(y_arr))])

    return corners


# read img and convert to RGB color
img = cv.imread(file)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# get img size
img_width = img.shape[1]
img_height = img.shape[0]
img_area = img_width * img_height

#process img and apply all transformations
mask = color_filter(img, lower_blue, upper_blue)
blurred_img = noise_reduction(mask)
edged_img = cv.Canny(blurred_img, 10, 200)
defining_lines = get_defining_lines(edged_img)
line_img = draw_lines(defining_lines, img)
intersections = get_intersections(defining_lines)
#corners = get_corners(intersections)
#pointed_img = draw_points(corners, img)

#show desired images
show_images([blurred_img, edged_img, line_img])




