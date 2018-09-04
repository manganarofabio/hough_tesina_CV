import numpy as np
import cv2


img_t = cv2.imread("start.jpg")
img_s = cv2.imread("s_start.png")


img_t = img_t.astype(np.float)
img_s = img_s.astype(np.float)

a = 0.5

img_f = (1 - a)*img_t + a*img_s

cv2.imwrite("filtered.png", img_f)






