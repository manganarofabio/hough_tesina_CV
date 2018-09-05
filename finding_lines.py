import cv2
import numpy as np


def to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)


def filter_color(img):


    white = np.uint8([[[255,255,255]]])
    hsv_w = cv2.cvtColor(white, cv2.COLOR_BGR2HSV)
    print(hsv_w[0, 0, 0])
    w_min = np.array(([hsv_w[0, 0, 0] -10, 100, 100 ]))
    w_max = np.array(([hsv_w[0, 0, 0] +10, 255, 255]))
    yellow_min = np.array([65, 80, 80], np.uint8)
    yellow_max = np.array([105, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(img, yellow_min, yellow_max)

    white_min = np.array([0, 0, 0], np.uint8)
    white_max = np.array([0, 0, 255], np.uint8)
    white_mask = cv2.inRange(img, w_min, w_max)

    img = cv2.bitwise_and(img, img, mask=white_mask)
    return img

def canny_post(img, mask):

    h, w = img.shape
    kernel_size = 3
    k = (kernel_size -1) // 2


    for r in range(k, h - k):
        for c in range(k, w - k):
            sum = 0
            for m in range(-k, k + 1):
                for n in range(-k, k + 1):
                    sum = img[r + m, c + n]
            if np.array_equal(mask[r, c], [0, 0, 0]) and img[r, c] != 0 and sum == 0:
                img[r, c] = 0

    return img

def color_filter(image):
    #convert to HLS to mask based on HLS
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    lower = np.array([0,190,0])
    upper = np.array([255,255,255])
    yellower = np.array([10,0,90])
    yelupper = np.array([50,255,255])
    yellowmask = cv2.inRange(hls, yellower, yelupper)
    whitemask = cv2.inRange(hls, lower, upper)
    mask = cv2.bitwise_or(yellowmask, whitemask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    return masked

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

    for x in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[x]:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite('lines.png', img)


def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)

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
    # image1 * α + image2 * β + λ
    # image1 and image2 must be the same shape.
    return cv2.addWeighted(image, 0.8, line_image, 1.0, 0.0)

def draw(img, lines, color=[255, 0, 0], thickness=8):
    imshape = img.shape
    ymin_global = img.shape[0]
    ymax_global = img.shape[0]
    all_left_grad = []
    all_left_y = []
    all_left_x = []
    all_right_grad = []
    all_right_y = []
    all_right_x = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            gradient = np.polyfit((x1, x2), (y1, y2), 1)
            ymin_global = min(min(y1, y2), ymin_global)

            if (np.all(gradient > 0)):
                all_left_grad += [gradient]
                all_left_y += [y1, y2]
                all_left_x += [x1, x2]
            else:
                all_right_grad += [gradient]
                all_right_y += [y1, y2]
                all_right_x += [x1, x2]
    left_mean_grad = np.mean(all_left_grad)
    left_y_mean = np.mean(all_left_y)
    left_x_mean = np.mean(all_left_x)
    left_intercept = left_y_mean - (left_mean_grad * left_x_mean)
    right_mean_grad = np.mean(all_right_grad)
    right_y_mean = np.mean(all_right_y)
    right_x_mean = np.mean(all_right_x)
    right_intercept = right_y_mean - (right_mean_grad * right_x_mean)
    if ((len(all_left_grad) > 0) and (len(all_right_grad) > 0)):
        print("dentro")
        upper_left_x = int((ymin_global - left_intercept) / left_mean_grad)
        lower_left_x = int((ymax_global - left_intercept) / left_mean_grad)
        upper_right_x = int((ymin_global - right_intercept) / right_mean_grad)
        lower_right_x = int((ymax_global - right_intercept) / right_mean_grad)
        cv2.line(img, (upper_left_x, ymin_global),
                 (lower_left_x, ymax_global), color, thickness)
        cv2.line(img, (upper_right_x, ymin_global),
                 (lower_right_x, ymax_global), color, thickness)

def main():

    img_to_seg =  cv2.imread("start.jpg")
    img_on = img_to_seg
    #img = cv2.imread("tmp_original.png")

    #conversione hls e

    #img_g = to_hsv(img_to_seg)
    img_g = img_to_seg
    #cv2.imwrite("pre_filter.png", img_g)

    # gaussian blur e bilateral filter

    img_g = cv2.GaussianBlur(img_g, (5, 5), 0)  # 9,9

    img_g = cv2.bilateralFilter(img_g, 9, 75, 75)

    #selezione linee


    #img_hsv = filter_color(img_g)
    #img_g = color_filter(img_to_seg)
    #cv2.imwrite("hsv.png", img_hsv)

    img_g = cv2.cvtColor(img_g, cv2.COLOR_BGR2GRAY)

    h, w , d = img_to_seg.shape


    #canny transform on image
    c_img = cv2.Canny(img_g, 255/3, 255)  #255/3 255
    cv2.imwrite('canny.png', c_img)

    #da canny prendo solo le parti presenti in tmp_original
    origin_mask = cv2.imread("tmp_original.png")

    #c_img = canny_post(c_img, origin_mask)
    for r in range(h):
        for c in range(w):
            if np.array_equal(origin_mask[r, c],[0, 0, 0]) and c_img[r, c] != 0:
                c_img[r, c] = 0

    cv2.imwrite("canny_post.png", c_img)

    # c_img = canny_post(c_img, origin_mask)
    #
    #
    #
    # cv2.imwrite("blurred.png", c_img)
    #cv2.imshow('canny', c_img)

    lines = hough_lines(c_img)

    #salva immagine

    #left_lane , right_lane = average_slope_intercept(lines)

    #lane_lines(c_img, lines)


    img_lines = img_to_seg
    draw_lines_on_img(lines, img_lines)

    img_fin = cv2.imread('start.jpg')
    #draw(img_fin, lines)
    img_fin = draw_lane_lines(img_fin, lane_lines(img_to_seg, lines))
    cv2.imwrite("fin_w.png", img_fin)

    #otteniamo una maschera che applico all'img originale



    # h , w, d = img_on.shape
    #
    # for r in range(h):
    #     for c in range(w):
    #         if not np.array_equal(mask[r, c], [0, 0, 0]):
    #             img_on[r, c] = mask[r, c]
    #
    #
    # cv2.imwrite('finale.png', img_on)



    #visualizzazione finale

    img_test = cv2.imread("test.png")
    img_vis_fin = np.zeros_like(img_test)

    a = 0.5
    for r in range(h):
        for c in range(w):
            if not np.array_equal(img_test[r,c], [0, 0, 0]):
                img_vis_fin[r, c] = (1 - a) * img_test[r, c] + a * img_fin[r, c]
            else:
                img_vis_fin[r, c] = img_fin[r, c]


    cv2.imwrite("visualizzazione.png", img_vis_fin)







if __name__ == '__main__':

    main()