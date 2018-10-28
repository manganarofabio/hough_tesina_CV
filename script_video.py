#script definitivo per il calcolo delle linee e visualizzazione finale

import argparse
import os
import cv2
import numpy as np
import ntpath
import glob

parser = argparse.ArgumentParser(description="Script per il calcolo delle linee e visualizzazione finale")
parser.add_argument("dir", type=str, help="dir where results are")


if __name__ == '__main__':

    # estrazione parametri
    args = parser.parse_args()

    images = sorted(glob.glob(os.path.join(args.dir, '*/result.png')))

    height, width, layers = cv2.imread(images[0]).shape

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video = cv2.VideoWriter('video.avi', fourcc, 2.5, frameSize=(width, height))

    for image in images:
        print(image)
        video.write(cv2.imread(image))

    video.release()
