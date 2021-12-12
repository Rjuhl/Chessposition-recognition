import os
import random
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from skimage import feature
from PIL import Image
from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import resize
import math
import time

# Resize Image
# Board Detection Premise
# Find board edges and crop
# Use Canny Edge on board
# Use Hough Transform to find grid lines
# Compute intersection to get square
# Get image/piece window
# Transform image to appropriate size

# Overall Notes
# 1) Resizing the images slightly decreases accuracy of grid lines (.5 to 1.5 misplaced lines on avr)
#    but it speed up the process by close to a magnitude
# 2) Some hyper params may still need to be tweaked

# All Hyperparams

# Hyperparams Normal
image_size = (800, 800) # Size to transform board to  (runs faster this way one high def imgs)
piece_img_size = (224, 224) # Size to transform pieces to so that they can run through the CNN
dcanny = 0.3 # The default hyperparam for sigma in the toCanny convolution
lineSig = [2, 2] # lineSig[0] is sigma for finding boards lines (horizontal) and lineSig[1] is the same but for vertical
offset = 20 # How much to enroach on board space found
marg = 20 # Extra distance below (or to the right depending on context) from corner point required to add it eligible neighbor list
wf = 0.5 # The piece window will be 1 + wf times taller than the square

# houghBoard[0][x] = horizontal board params and houghBoard[1][x] = vertical board params
# houghBoard[x][0] = min_dist, houghBoard[x][1] = threshhold, and houghBoard[x][2] = numpeaks
houghBoard = [[10, 100, 16], [10, 100, 16]]

# houghGrid[0][x] = horizontal grid params and houghGrid[1][x] = vertical grid params
# houghGrid[x][0] = min_dist, houghGrid[x][1] = threshhold, and houghGrid[x][2] = numpeaks
houghGrid = [[20, 5, 9], [20, 5, 9]]
short_cut_board = False # If False it performs find board operations first; otherwise, it does not at all


# Hyperparams Graping
pOrgin = True # Plots the original image
pProImg = False # Plots the original image after pre processing
hBoardLines = False # Plots horizontal lines when finding board edges
vBoardLines = False # Plots vertical lines when finding board edges
gBoard = False # Plots board outline
gCanny = False # Plots board after canny convolution
gGrid = True # Plots out board grid lines
pInter = True # Plots out the grid lines intersections (Needs pGrid to be True to plot)
gCorner = True # Plots out corner points
gNeigh = False # Plots out 8 selected corner points and their closest appropriate neighbors
pDecompPeices = True # Plots chess board as image_arr



# Test Images (Range of Qualities)
image0 = Image.open("board7.JPG")
image1 = Image.open("chessboardtest3.jpeg")
image2 = Image.open("phpIvIkul.jpeg")
image3 = Image.open("Chess_board_opening_staunton.jpg")
image4 = Image.open("chessboardtest.jpeg")
image5 = Image.open("chessboardtest2.jpg")



# Find outer lines from a a set of lines (For horizontal lines)
def getLineAvrgX(point, slope, max):
    x, y = point
    y_val_at_zero = y - (x * slope)
    y_val_at_max = y + ((max - x) * slope)
    average = (y_val_at_max + y_val_at_zero) / 2
    return average


# Find outer lines from a a set of lines (For vertical lines)
def getLineAvrgY(point, slope, max):
    x, y = point
    x_val_at_zero = (-y / slope) + x
    x_val_at_max = ((max - y) / slope) + x
    average = (x_val_at_max + x_val_at_zero) / 2
    return average


def greyScale(image):
    preproc = transforms.Compose([transforms.Grayscale(), transforms.Resize(image_size)])
    return np.asarray(preproc(image))


def pieceTransform(image):
    return resize(image, piece_img_size)


def orderPeiceImg(img_arr): # Test this

    new_arr = []
    left_over_arr = img_arr.copy()
    for i in range(8):
        sort1 = sorted(left_over_arr, key=lambda arg: arg[0][1])
        left_over_arr = sort1[8:]
        sort2 = sorted(sort1[:8], key=lambda arg: arg[0][0])
        new_arr = new_arr + sort2[:8]
    return new_arr


# Performs Canny transformation to image
def toCanny(image, sig=dcanny, plot=False):
    image = feature.canny(image, sig)
    if plot:
        plt.imshow(image, cmap='gray')
        plt.show()
    return image


# Find a series of horizontal lines on the image
def getHoriLines(image, plot=False):
    image = toCanny(image, sig=lineSig[0])
    test_angles = np.linspace(np.pi / 1.5, np.pi / 2.5, 90)
    hspace, theta, dist = hough_line(image, test_angles)
    h, q, d = hough_line_peaks(hspace, theta, dist, min_distance=houghBoard[0][0], threshold=houghBoard[0][1], num_peaks=houghBoard[0][2])
    lines = [[q[i], d[i]] for i in range(q.shape[0])]
    point_slope_form = []
    for ang, len in lines:
        if plot: plt.imshow(image, cmap='gray')
        point = len * np.array([np.cos(ang), np.sin(ang)])
        slope = np.tan(ang + (np.pi / 2))
        if plot:
            plt.axline(point, slope=slope)
        point_slope_form.append([point, slope])
    if plot: plt.show()
    return point_slope_form


# Find a series of vertical lines on the image
def getVertLines(image, plot=False):
    image = toCanny(image, sig=lineSig[1])
    test_angles = np.linspace(-np.pi / 6, np.pi / 6, 90)
    hspace, theta, dist = hough_line(image, test_angles)
    h, q, d = hough_line_peaks(hspace, theta, dist, min_distance=houghBoard[1][0], threshold=houghBoard[1][1], num_peaks=houghBoard[1][2])
    lines = [[q[i], d[i]] for i in range(q.shape[0])]
    point_slope_form = []
    for ang, len in lines:
        if plot: plt.imshow(image, cmap='gray')
        point = len * np.array([np.cos(ang), np.sin(ang)])
        slope = np.tan(ang + (np.pi / 2))
        if plot:
            plt.axline(point, slope=slope)
        point_slope_form.append([point, slope])
    if plot: plt.show()
    return point_slope_form


# Finds the board outline (with an offset)
def getBoard(image, plot=False):
    hl = getHoriLines(image, plot=hBoardLines)
    vl = getVertLines(image, plot=vBoardLines)
    xmax, ymax = image.shape

    # Find the outer horizontal lines
    hl_high = [-1, None]
    hl_low = [math.inf, None]
    for i in range(len(hl)):
        point, slope = hl[i]
        avr = getLineAvrgX(point, slope, xmax)
        if avr > hl_high[0]:
            hl_high[0] = avr
            hl_high[1] = i
        if avr < hl_low[0]:
            hl_low[0] = avr
            hl_low[1] = i

    # Find the outer vertical lines
    vl_high = [-1, None]
    vl_low = [math.inf, None]
    for i in range(len(vl)):
        point, slope = vl[i]
        avr = getLineAvrgY(point, slope, ymax)
        if avr > vl_high[0]:
            vl_high[0] = avr
            vl_high[1] = i
        if avr < vl_low[0]:
            vl_low[0] = avr
            vl_low[1] = i

    # Changes the image pixel to black if outside the board
    for i in range(image.shape[0]): 
        for j in range(image.shape[1]):
            p1, s1 = hl[hl_high[1]]
            t1 = p1[1] - ((p1[0] - j) * s1) if i < p1[0] else p1[1] + ((j - p1[0]) * s1)
            if s1 > 0:
               t1 = p1[1] + ((p1[0] - j) * s1) if i < p1[0] else p1[1] - ((j - p1[0]) * s1)
            if i + (offset * 2.5) > t1:
                image[i][j] = random.randint(0, 255)

            p2, s2 = hl[hl_low[1]]
            t2 = p2[1] - ((p2[0] - j) * s2) if i < p2[0] else p2[1] + ((j - p2[0]) * s2)
            if s2 > 0:
                t2 = p2[1] + ((p2[0] - j) * s2) if i < p2[0] else p2[1] - ((j - p2[0]) * s2)
            if i - (offset * 1) < t2:
                image[i][j] = random.randint(0, 255)

            p3, s3 = vl[vl_high[1]]
            t3 = p3[0] + ((i - p3[1]) / s3) if p3[1] < i else p3[0] - ((p3[1] - i) / s3)
            if s3 < 0:
                t3 = p3[0] - ((i - p3[1]) / s3) if p3[1] < i else p3[0] + ((p3[1] - i) / s3)
            if j + offset > t3:
                image[i][j] = random.randint(0, 255)

            p4, s4 = vl[vl_low[1]]
            t4 = p4[0] + ((i - p4[1]) / s4) if p4[1] < i else p4[0] - ((p4[1] - i) / s4)
            if s4 > 0:
                t4 = p4[0] - ((i - p4[1]) / s4) if p4[1] < i else p4[0] + ((p4[1] - i) / s4)
            if j - offset < t4:
                image[i][j] = random.randint(0, 255)

    if plot:
        plt.imshow(image, cmap='gray')
        p, s = hl[hl_high[1]]
        p2, s2 = hl[hl_low[1]]
        p3, s3 = vl[vl_high[1]]
        p4, s4 = vl[vl_low[1]]
        plt.axline(p, slope=s)
        plt.axline(p2, slope=s2)
        plt.axline(p3, slope=s3)
        plt.axline(p4, slope=s4)
        plt.show()

    return image


# Horizontal and vertical lines are computed separately to improve results
def getGridLines(image, plot=False):

    image = greyScale(image)
    org_img = image.copy()
    if not short_cut_board:
        image = getBoard(image, plot=gBoard)
    image = toCanny(image, sig=3, plot=gCanny)

    h_test_angles = np.linspace(np.pi / 1.5, np.pi / 2.5, 90)
    hhspace, htheta, hdist = hough_line(image, h_test_angles)
    hh, hq, hd = hough_line_peaks(hhspace, htheta, hdist, min_distance=houghGrid[0][0], threshold=houghGrid[0][1], num_peaks=houghGrid[0][2])
    hlines = [[hd[i], hq[i]] for i in range(hq.shape[0])]

    v_test_angles = np.linspace(-np.pi / 6, np.pi / 6, 90)
    vhspace, vtheta, vdist = hough_line(image, v_test_angles)
    vh, vq, vd = hough_line_peaks(vhspace, vtheta, vdist, min_distance=houghGrid[1][0], threshold=houghGrid[1][1], num_peaks=houghGrid[1][2])
    vlines = [[vd[i], vq[i]] for i in range(vq.shape[0])]

    def findCornerLines(hl, vl):
        hh = [-1, None]
        for r, t in hl:
            avrg = (r / np.sin(t)) - (image_size[0] / (2 * np.tan(t)))
            if avrg > hh[0]:
                hh[0] = avrg
                hh[1] = [r, t]

        vh = [-1, None]
        for r, t in vl:
            avrg = (r / np.cos(t)) - (image_size[1] * np.tan(t) / 2)
            if avrg > vh[0]:
                vh[0] = avrg
                vh[1] = [r, t]
        hl.remove(hh[1])
        vl.remove(vh[1])
        return hl, vl


    # Computes the intersection of all grid lines
    def intersections(h, v):
        points = []
        for d1, a1 in h:
            for d2, a2 in v:
                A = np.array([[np.cos(a1), np.sin(a1)], [np.cos(a2), np.sin(a2)]])
                b = np.array([d1, d2])
                point = np.linalg.solve(A, b)
                points.append(point)
        return np.array(points)

    dots = intersections(hlines, vlines)

    hl = hlines.copy()
    vl = vlines.copy()
    hl, vl = findCornerLines(hl, vl)
    corners = intersections(hl, vl)

    if pInter:
        plt.imshow(org_img, cmap='gray')
        for point in dots:
            x, y = point
            plt.plot(x, y, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
        plt.show()

    if gCorner:
        plt.imshow(image, cmap='gray')
        for x, y in corners:
            plt.plot(x, y, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
        plt.show()

    lines = hlines + vlines
    point_slope_form = []
    for len, ang in lines:
        if plot: plt.imshow(org_img, cmap='gray')
        point = len * np.array([np.cos(ang), np.sin(ang)])
        slope = np.tan(ang + (np.pi / 2))
        if plot:
            plt.axline(point, slope=slope)
        point_slope_form.append([point, slope])
    if plot: plt.show()

    return corners, dots


# Finds all "Corners" from gridline intersections
'''
def findCorners(dots, img=None):
    corners = sorted(dots.tolist())[:-9]
    corners = sorted(corners, key=lambda p: p[1])[:-8]
    if img is not None:
        plt.imshow(greyScale(img), cmap='gray')
        for x, y in corners:
            plt.plot(x, y, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
        plt.show()
    return corners
'''

# For each corner it computes the it neighbor points i.e the point directly to its right and directly below it
def getNeighboorPoints(corners, dots, margin=20, img=None):
    point_to_neigh = []

    def findClosest(point, points):
        distances = []
        indexs = {}
        for i in range(len(points)):
            dist = math.dist(point, points[i])
            distances.append(dist)
            indexs[dist] = i
        return points[indexs[min(distances)]]

    for i in range(len(corners)):
        x_points = []
        y_points = []
        for p2 in dots:
            if p2[0] > corners[i][0] + margin:
                x_points.append(p2)
            if p2[1] > corners[i][1] + margin:
                y_points.append(p2)
        point_to_neigh.append([corners[i], findClosest(corners[i], x_points), findClosest(corners[i], y_points)])

    point_to_neigh = orderPeiceImg(point_to_neigh)
    if img is not None:
        for i in range(0, 10):
            # print(point_to_neigh[i])
            plt.imshow(greyScale(img), cmap='gray')
            plt.plot(point_to_neigh[i][0][0], point_to_neigh[i][0][1], marker="o", markersize=5, markeredgecolor="black", markerfacecolor="black")
            plt.plot(point_to_neigh[i][1][0], point_to_neigh[i][1][1], marker="o", markersize=5, markeredgecolor="green", markerfacecolor="green")
            plt.plot(point_to_neigh[i][2][0], point_to_neigh[i][2][1], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")
            plt.show()
            print(i)
            time.sleep(0.2)

    return point_to_neigh

# Using the neighbor information this finds the window for peice and creates an array of them. This is what will ultimately be feed into the CNN
def getPieceImgs(neigh_arr, image, window_factor=0.5):
    img_arr = []
    image = greyScale(image)
    for i in range(len(neigh_arr)):
        c, rc, bc, = neigh_arr[i]
        largest_x = max([c[1], rc[1], bc[1]])
        largest_y = max([c[0], rc[0], bc[0]])
        final_x_range = largest_x - c[1]
        y_range = largest_y - c[0]
        scale_addition = y_range * window_factor
        if c[0] - scale_addition < 0: # Make sure we don't get an outbounds error by trying to pixel outside of the image bounds
            scale_addition = c[0] - 1

        starting_pixel = [int(math.floor(c[1] - scale_addition)), int(math.floor(c[0]))]
        if starting_pixel[1] + y_range + scale_addition >= 800: # Make sure we don't get an outbounds error by trying to pixel outside of the image bounds
            y_range = 800 - starting_pixel[1] - scale_addition - 1
        window = np.zeros((int(math.floor(final_x_range)), int(math.floor(y_range + scale_addition))))

        for x in range(window.shape[0]):
            for y in range(window.shape[1]):
                window[x][y] = image[starting_pixel[0] + x][starting_pixel[1] + y]

        img_arr.append(pieceTransform(window))

    return img_arr


# Succinctly combine all the above functions and return the image arr that the CNN will use
# Image should be in PIL form
def fullyProcess(image):
    if pOrgin:
        plt.imshow(image)
        plt.show()
    if pProImg:
        plt.imshow(greyScale(image), cmap='gray')
        plt.show()
    nb = image if gNeigh else None
    corners, dots = getGridLines(image, plot=gGrid)
    neigh = getNeighboorPoints(corners, dots, margin=marg, img=nb)
    res = getPieceImgs(neigh, image, window_factor=wf)
    if pDecompPeices:
        f, ax = plt.subplots(8, 8)
        track_ind = 0
        for i in range(8):
            for j in range(8):
                ax[i, j].imshow(res[track_ind], cmap='gray')
                ax[i, j].set_axis_off()
                track_ind += 1
        plt.show()

    return res

'''
already_have = []
for file in os.listdir('/Users/rainjuhl/PycharmProjects/pythonProject/decompTest'):
    already_have.append(str(file[0:-4]))

for image in os.listdir('/Users/rainjuhl/PycharmProjects/pythonProject/test'):
    if image == '_annotations.createml.json' or str(image) in already_have:
        continue
    name = image
    print(image)
    image = Image.open(os.path.join('/Users/rainjuhl/PycharmProjects/pythonProject/test', image))
    img_arr = fullyProcess(image)
    np.save(os.path.join('/Users/rainjuhl/PycharmProjects/pythonProject/decompTest', name), np.asarray(img_arr))
'''
