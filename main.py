import cv2 as cv2
import numpy as np

def extract():
    img_seg = cv2.imread('s_start.png')
    img_nseg = cv2.imread('start.jpg')

    height, width, depth = img_seg.shape

    #print(img_seg)

    print("strada")
    pix_road =  [128, 64, 128]

    print("strisce")

    pix_lines = [255, 255, 255]
    print(pix_lines)

    #creo immagine solo con strada e strisce


    print(height)
    print(width)


    test = np.zeros((height, width, depth))
    #print(test)

    count = 0
    for r in range(height):
        for c in range(width):
            if np.array_equal(img_seg[r, c], pix_road) or np.array_equal(img_seg[r, c], pix_lines):
                test[r, c] = img_seg[r, c]
                count += 1

    #print("fine for")
    #print("count {}".format(count))
    #print(test)

    cv2.imwrite("test.png", test)

    #cv2.imshow("test", test)
    print('img salvata')
    return test


def from_original(test):

    img = cv2.imread('start.jpg')
    h, w , d = img.shape

    #print(test)
    tmp = np.zeros((h, w, d))
    zero = np.array([0, 0, 0])
    #print(np.zeros((1, 3)))
    count = 0
    for r in range(h):
        for c in range(w):
            if not np.array_equal(test[r, c], zero):
                #print("dentro {}, {}".format(r, c) )
                #print(img[r,c])
                tmp[r, c] = img[r, c]
                count += 1
    #print('valore')
    #print(tmp[990, 2])
    #print("count {}".format(count))

    #print("fine for")
    cv2.imwrite("tmp_original.png", tmp)




def main():

    #pass
    test = extract()
    test = cv2.imread("test.png")
    from_original(test)
    #calculate_hough()


if __name__ == '__main__':

    main()

