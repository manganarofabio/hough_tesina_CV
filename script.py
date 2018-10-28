#script definitivo per il calcolo delle linee e visualizzazione finale

import argparse
import os
import cv2
import numpy as np
import ntpath


parser = argparse.ArgumentParser(description="Script per il calcolo delle linee e visualizzazione finale")
parser.add_argument("original_image_path", type=str, help="original image path")
parser.add_argument("segmented_image_path", type=str, help="segmented image path")
parser.add_argument("--dir", type=str, dest="dir", help="dir where save results")



def get_mask_original_image(original, mask_segmented_road):

    #img = cv2.imread('start.jpg')
    h, w , d = mask_segmented_road.shape

    tmp = np.zeros((h, w, d), dtype=np.uint8)
    zero = np.array([0, 0, 0], dtype=np.uint8)


    # for r in range(h):
    #     for c in range(w):
    #         if not np.array_equal(mask_segmented_road[r, c], zero):
    #             tmp[r, c] = original[r, c]

    # ToDo test
    tmp[mask_segmented_road != 0] = original[mask_segmented_road != 0]

    cv2.imwrite("mask_original.png", tmp)
    return tmp


#funzioni per hough

def hough_lines(image):
    """
    `image` should be the output of a Canny transform.

    Returns hough lines (not the image with lines)
    """
    #30 20 20

    #best 40, 20, 300
    #fin test 40, 20, 50
    return cv2.HoughLinesP(image, rho=1, theta=np.pi / 90, threshold=40, minLineLength=20, maxLineGap=50)


def average_slope_intercept(lines):
    left_lines = []  # (slope, intercept)
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1 or np.abs(y2 -y1) < 100:
                continue  # ignore a vertical line
            slope = (y2 - y1) / (x2 - x1)
            # if slope >= 0. and slope <= 1.:
            #     continue
            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            if slope < 0:  # y is reversed in image
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))

    # add more weight to longer lines
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None

    return left_lane, right_lane  # (slope, intercept), (slope, intercept)

def average_slope_intercept_modified(lines):
    left_lines = []  # (slope, intercept)
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)

    for line in lines:
        for x1, y1, x2, y2 in line:
            # if x2 == x1 or np.abs(y2 -y1) < 100:
            #     continue  # ignore a vertical line
            if x2 == x1:
                continue
            else:
                slope_abs = np.abs((y2 - y1) / (x2 - x1))
                if slope_abs >= 0. and slope_abs <= 0.5:
                     continue
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
                if slope < 0:  # y is reversed in image
                    left_lines.append((slope, intercept))
                    left_weights.append((length))
                else:
                    right_lines.append((slope, intercept))
                    right_weights.append((length))

    # add more weight to longer lines
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None

    return left_lane, right_lane  # (slope, intercept), (slope, intercept)


def make_line_points(y1, y2, line):
    """
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None

    slope, intercept = line

    # make sure everything is integer as cv2.line requires it
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)

    return ((x1, y1), (x2, y2))



def draw_lines_on_img(lines, img):

    img_post = img.copy()
    for x in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[x]:

            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #if x2 == x1: #or np.abs(y2 - y1) < 100:

                #continue  # ignore a vertical line
            if x2 == x1:
                continue
            else :
                slope = np.abs((y2 - y1) / (x2 - x1))
                if slope >= 0. and slope <= 0.5:
                    continue

            cv2.line(img_post, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite('lines.png', img)
    cv2.imwrite('lines_post.png', img_post)


def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept_modified(lines)  #MODIFIED

    y1 = image.shape[0]  # bottom of the image
    y2 = y1 * 0.6  # slightly lower than the middle 0.6

    left_line = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)

    return left_line, right_line


def draw_lane_lines(image, lines, color=[0, 0, 255], thickness=8):
    # make a separate image to draw lines and combine with the orignal later
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)

    cv2.imwrite('lines_end.png', line_image)
    image = cv2.addWeighted(image, 0.8, line_image, 1.0, 0.0)

    return line_image, image
    # image1 * α + image2 * β + λ
    # image1 and image2 must be the same shape.

    #return cv2.addWeighted(image, 0.8, line_image, 1.0, 0.0), line_image

def overlay(height, width, img, mask):

    # for r in range(height):
    #     for c in range(width):
    #         if not np.array_equal(mask[r, c], [0, 0, 0]):
    #             img[r, c] = mask[r, c]

    img[np.all(mask != [0, 0, 0], axis=-1)] = mask[np.all(mask != [0, 0, 0], axis=-1)]
    return img




if __name__ == '__main__':

    # estrazione parametri
    args = parser.parse_args()

    start_path = sorted(os.listdir(args.original_image_path))
    s_start_path = sorted(os.listdir(args.segmented_image_path))

    assert len(start_path) == len(s_start_path)

    for start, s_start in zip(start_path, s_start_path):

        start = os.path.join(args.original_image_path, start)
        s_start = os.path.join(args.segmented_image_path, s_start)

        #FASE PRE-HOUGH

        #creazione cartella risultati

        if not args.dir:
            newpath = ntpath.basename(args.original_image_path) + '_final'
        else:
            newpath = args.dir


        #creo cartella relativa al file newpath/file_name

        os.makedirs(os.path.join(newpath, os.path.basename(start)[:-4]), exist_ok=True)
        os.chdir(os.path.join(newpath, os.path.basename(start)[:-4]))
        print(os.path.join(newpath, os.path.basename(start)[:-4]))
        print(start, s_start)


        #inizzializiamo le immagini

        img_nseg = cv2.imread(start)
        img_seg = cv2.imread(s_start)

        cv2.imwrite("to_seg.jpg", img_nseg)
        cv2.imwrite("segmented.png", img_seg)

        #prendiamo dimensioni
        height, width, depth = img_seg.shape

        #rgb strada
        pix_road = [128, 64, 128]
        #rgb strisce
        pix_lines = [255, 255, 255]

        #otteniamo maschera strada segmentatata
        mask_segmented_road = np.zeros((height, width, depth))
        # print(test)



        print("calcolo mask_segmented_road")
        # for r in range(height):
        #     for c in range(width):
        #         if np.array_equal(img_seg[r, c], pix_road) or np.array_equal(img_seg[r, c], pix_lines):
        #             mask_segmented_road[r, c] = img_seg[r, c]

        # ToDo test   #ERRORE

        mask_segmented_road[np.all(img_seg == pix_lines, axis=-1)] = pix_lines
        mask_segmented_road[np.all(img_seg == pix_road, axis=-1)] = pix_road
        



        cv2.imwrite("mask_segmented_road.png", mask_segmented_road)
        print("calcolo mask_segmented_road: completato")


        #otteniamo maschera strada immagine originale
        print("calcolo mask_original_road")

        mask_original = get_mask_original_image(img_nseg, mask_segmented_road)

        print("calcolo mask_original_road: completato")

        #FASE CALCOLO LINEE HOUGH

        #applichiamo filtri per pulire l'immagine

        # img_g = cv2.GaussianBlur(img_nseg, (5, 5), 0)  # 9,9
        img_g = cv2.bilateralFilter(img_nseg, 9, 75, 75)

        #in scala di grigi

        img_g = cv2.cvtColor(img_g, cv2.COLOR_BGR2GRAY)

        #applichiamo canny


        c_img = cv2.Canny(img_g, 255 / 3, 255)  # 255/3 255
        cv2.imwrite('canny.png', c_img)


        #elaboro canny per elementi significativi

        # for r in range(height):
        #     for c in range(width):
        #         if np.array_equal(mask_original[r, c], [0, 0, 0]) and c_img[r, c] != 0:
        #             c_img[r, c] = 0

        # ToDo test
        mask_original = cv2.cvtColor(mask_original, cv2.COLOR_BGR2GRAY)
        c_img[mask_original == 0] = 0

        cv2.imwrite("canny_post.png", c_img)
        print("canny completato")


        #calcolo linee hough

        lines = hough_lines(c_img)

        #creiamo immagine linee hough
        img_lines = img_nseg.copy()
        draw_lines_on_img(lines, img_lines)

        #creazione img con linee hough definitive

        #immagine pulita
        img_lines_def = cv2.imread("to_seg.jpg")


        #img_lines_def,
        lines_end, lines_mixed = draw_lane_lines(img_lines_def, lane_lines(img_nseg, lines))

        cv2.imwrite("lines_m.png", lines_mixed)
        #cv2.imwrite("lines_def.png", img_lines_def)

        print("calcolo linee completato")


        #visualizzazione finale



        a = 0.5
        b = 1.

        # img_res = np.zeros_like(img_nseg)
        # for r in range(height):
        #     for c in range(width):
        #         if not np.array_equal(mask_segmented_road[r, c], [0, 0, 0]):
        #             img_res[r, c] = (1 - a) * mask_segmented_road[r, c] + a * img_lines_def[r, c]
        #         else:
        #             img_res[r, c] = img_lines_def[r, c]

        # ToDo test
        img_res = img_lines_def.copy()
        img_res[np.all(mask_segmented_road != 0, axis=-1)] = (1 - a) * mask_segmented_road[np.all(mask_segmented_road != 0, axis=-1)] + a * img_lines_def[np.all(mask_segmented_road != 0, axis=-1)]

        #overlay(height, width, img_res, lines_end)
        img_res[lines_end != 0] = (1 - b) * img_lines_def[lines_end != 0] + b * lines_end[lines_end != 0]

        cv2.imwrite("result.png", img_res)

        print("visualizzazione finale completato")

        os.chdir('../..')

