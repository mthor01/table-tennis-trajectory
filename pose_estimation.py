import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math
from sympy import Point, Line
import copy


#set path of img file
#there is a folder with 8 images to try out wich you can acces by changing the number x in x.jpg
file = 'test_images/3.jpg'

# set color bounds based on if green or blue is more present in the center of the image
def set_color_range(img):
    blue_counter = 0
    green_counter = 0
    for i in range(int(img_height/4), int(img_height/4*3)):
        for j in range(int(img_width/4), int(img_width/4*3)):
            if (img[i][j][2] > img[i][j][0]) and (img[i][j][2] > img[i][j][1]):
                blue_counter += 1
            elif (img[i][j][1] > img[i][j][0]) and (img[i][j][1] > img[i][j][2]):
                green_counter +=1

    if blue_counter >= green_counter:
        lower_bound = np.array([90, 30, 90])
        upper_bound = np.array([135, 255, 255])
        print([blue_counter, green_counter])
    else:
        lower_bound = np.array([40, 30, 90])
        upper_bound = np.array([75, 255, 255])
        print([blue_counter, green_counter])

    return lower_bound, upper_bound

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
def get_defining_lines(edge_img, fraction, ):
    final_lines = []
    lines = cv.HoughLines(edge_img, 1, np.pi / 180 / 2, int(img_height/fraction), None, 0, 0)
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
                if ((dist_pt1 < img_width/20) and (dist_pt2 < img_width/20)):
                    new = False
                    break

            # append list of remaining lines and further sort out lines that are completely in the top third of the image
            if (new) and ((new_pt1[1]> img_height/3) or (new_pt2[1]> img_height/3)):
                reduced_lines.append([[new_pt1, new_pt2], theta])

    return reduced_lines


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

#out of all contours only return the 2 with maximal area enclosed
#if the second largest area is much smaller than the largest,
#it is assumed that the table was not split in half by contours and therefore only the largest will be returned
def get_table_contours(contours):
    max1 = [0, 0];
    max2 = [0, 0];
    for i in range(len(contours)):
        area = cv.contourArea(contours[i])
        if area > max1[0]:
            max1 = [area, i]
        elif area > max2[0]:
            max2 = [area, i]

    table_contours = [contours[max1[1]]]
    if (max1[0] < 3 * max2[0]) or (len(contours) == 1):
        table_contours.append(contours[max2[1]])
    return table_contours

# creates a black image with white filled given contours
def fill_table_contours(contour_img):
    im_floodfill = contour_img.copy()
    mask = np.zeros((img_height + 2, img_width + 2), np.uint8)
    cv.floodFill(im_floodfill, mask, (0, 0), 255);
    im_floodfill_inv = cv.bitwise_not(im_floodfill)
    filled_img = contour_img | im_floodfill_inv
    filled_img = cv.morphologyEx(filled_img, cv.MORPH_CLOSE, kernel)
    return filled_img

# calculate the x and y span in which the contours are placed
def get_contour_bounds(contour_img):
    x,y,w,h = cv.boundingRect(contour_img)
    x_span = [x, x+w]
    y_span = [y, y+h]

    return x_span, y_span

# sort out points that are guranteed to be no table corners
# returns candidate points for table corners and a bool,
# indicating if at least one point is close to every span extremum
def get_possible_points(points, x_span, y_span):
    points_in_bounds = []
    allowed_error = img_height/20
    boarders_reached_check = [0, 0, 0, 0]

    for point in points:

        contact_to_boarder = False

        # check if a point lies in a given span and allows some error
        if ((point[0]>x_span[0]-allowed_error) and (point[0]<x_span[1]+allowed_error) and (point[1]>y_span[0]-allowed_error ) and (point[1]<y_span[1]+allowed_error)):
            if abs(point[0] - x_span[0]) < allowed_error:
                boarders_reached_check[0] = 1
                contact_to_boarder = True
            elif abs(point[0] - x_span[1]) < allowed_error:
                boarders_reached_check[1] = 1
                contact_to_boarder = True
            elif abs(point[1] - y_span[0]) < allowed_error:
                boarders_reached_check[2] = 1
                contact_to_boarder = True
            elif abs(point[1] - y_span[1]) < allowed_error:
                boarders_reached_check[3] = 1
                contact_to_boarder = True

        if contact_to_boarder == True:
            points_in_bounds.append(point)

    return points_in_bounds, sum(boarders_reached_check)==4




# read img and convert to RGB color
img = cv.imread(file)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# get img size
img_width = img.shape[1]
img_height = img.shape[0]
img_area = img_width * img_height

lower_bound, upper_bound = set_color_range(img)

#process img and apply all transformations
mask = color_filter(img, lower_bound, upper_bound)
kernel = np.ones((5,5),np.uint8)
closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
contours, hierarchy = cv.findContours(closing, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
table_contours = get_table_contours(contours)
contour_img = cv.drawContours(np.zeros((img_height, img_width), dtype="uint8"), table_contours, -1, 255, 1)
x_span, y_span = get_contour_bounds(contour_img)

hough_denominator = 3
valid_corners = False

while hough_denominator <= 10:
    hough_denominator += 1
    defining_lines = get_defining_lines(contour_img, hough_denominator)
    line_img = draw_lines(defining_lines, img)
    intersections = get_intersections(defining_lines)
    possible_corners, all_edges_reached = get_possible_points(intersections, x_span, y_span)
    if (all_edges_reached == True):
        corners = get_corners(possible_corners)
        valid_corners = True
        break

if valid_corners == False:
    print("Corners not found")
else:

    #set 2D and 3D corner coordinates
    points_2D = np.array([(corners[0][0], corners[0][1]),
                          (corners[1][0], corners[1][1]),
                          (corners[2][0], corners[2][1]),
                          (corners[3][0], corners[3][1])], dtype="double")

    points_3D = np.array([(137, -76.25, 0),
                          (-137, 76.25, 0),
                          (-137, -76.25, 0),
                          (137, 76.25, 0)], dtype="double")

    # make empty distance coefficients and estimate camera matrix
    dist_coeffs = np.zeros((4, 1))

    camera_matrix=np.array([(img_width, 0, img_width/2),
                            (0, img_height, img_height/2),
                            (0,0,1)])

    #estimate rotation vector and translation vector
    success, rotation_vector, translation_vector, inliers = cv.solvePnPRansac(points_3D, points_2D, camera_matrix, dist_coeffs, flags=0)

    #calculate 2D surface normal
    nose_end_point2D, jacobian = cv.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector,camera_matrix, dist_coeffs)

    for p in points_2D:
      cv.circle(img, (int(p[0]), int(p[1])), 3, (255,0,0), -1)

    # draw surface normal line
    point1 = [points_2D[0][0], points_2D[0][1]]
    point2 = [nose_end_point2D[0][0][0], nose_end_point2D[0][0][1]]

    if point2[1] > point1[1]:
        point2[0] = point1[0]-(point2[0]-point1[0])
        point2[1] = point1[1] - (point2[1] - point1[1])

    point1 = (int(point1[0]), int(point1[1]))
    point2 = (int(point2[0]), int(point2[1]))

    cv.line(img, point1, point2, (255, 0, 0), 2)

    # Display desired images
    show_images([img])




