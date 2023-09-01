#!/bin/env python
import numpy as np
import cv2
import imageio
import subprocess
from optparse import OptionParser


# from https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
# internalRect: returns the intersection between two rectangles
#
#  p1 ---------------- p2
#   |                  |
#   |                  |
#   |                  |
#  p4 ---------------- p3
def internalRect(r1, r2):
    x = 0
    y = 1
    w = 2
    h = 3

    rect1_pt1 = [r1[x], r1[y]]
    rect1_pt2 = [r1[x] + r1[w], r1[y]]
    rect1_pt3 = [r1[x] + r1[w], r1[y] + r1[h]]
    rect1_pt4 = [r1[x], r1[y] + r1[h]]

    rect2_pt1 = [r2[x], r2[y]]
    rect2_pt2 = [r2[x] + r2[w], r2[y]]
    rect2_pt3 = [r2[x] + r2[w], r2[y] + r2[h]]
    rect2_pt4 = [r2[x], r2[y] + r2[h]]

    int_pt1 = [max(rect1_pt1[x], rect2_pt1[x]), max(rect1_pt1[y], rect2_pt1[y])]
    int_pt2 = [min(rect1_pt2[x], rect2_pt2[x]), max(rect1_pt2[y], rect2_pt2[y])]
    int_pt3 = [min(rect1_pt3[x], rect2_pt3[x]), min(rect1_pt3[y], rect2_pt3[y])]
    int_pt4 = [max(rect1_pt4[x], rect2_pt4[x]), min(rect1_pt4[y], rect2_pt4[y])]

    rect = [int_pt1[x], int_pt1[y], int_pt2[x] - int_pt1[x], int_pt4[y] - int_pt1[y]]
    return rect


# align_image: use src1 as the reference image to transform src2
def align_image(simg1, simg2, src1, src2, scale, warp_mode=cv2.MOTION_TRANSLATION):
    # convert images to grayscale
    # use the smaller image for alignment
    img1_gray = cv2.cvtColor(simg1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(simg2, cv2.COLOR_BGR2GRAY)

    # define 2x3 or 3x3 matrices and initialize it to a identity matrix
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # number of iterations:
    num_iters = 1000

    # specify the threshold of the increment in the correlation coefficient between two iterations
    termination_eps = 1e-8

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_iters, termination_eps)

    print("findTransformECC() may take a while...")

    # perform ECC: use the selected model to calculate the transformation required to align src2 with src1. The resulting transformation matrix is stored in warp_matrix:
    (cc, warp_matrix) = cv2.findTransformECC(img1_gray, img2_gray, warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=1)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        img2_aligned = cv2.warpPerspective(src2, warp_matrix, (src1.shape[1], src1.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        # use warpAffine() for: translation, euclidean and affine models
        img2_aligned = cv2.warpAffine(src2, warp_matrix, (src1.shape[1], src1.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # compute the cropping area to remove the black bars from the transformed image
    x = 0
    y = 0
    w = src1.shape[1]
    h = src1.shape[0]

    if warp_matrix[0][2] < 0:
        x = warp_matrix[0][2] * -1
        w -= x

    if warp_matrix[1][2] < 0:
        y = warp_matrix[1][2] * -1
        h -= y

    if warp_matrix[1][2] > 0:
        h -= warp_matrix[1][2]

    matchArea = [int(x), int(y), int(w), int(h)]

    return img2_aligned, matchArea


##########################################################################################


def main(options, args):
    img1 = cv2.imread(args[0])
    img2 = cv2.imread(args[1])

    ###
    # resize images to be the same size as the smallest image for debug purposes
    ###
    max_h = img1.shape[0]
    max_h = max(max_h, img2.shape[0])
    max_w = img1.shape[1]
    max_w = max(max_w, img2.shape[1])
    scale = options.scale

    min_w = min(max_w, 1000)
    auto_scale = min_w / max_w
    min_h = int(min_w / max_w * max_h)

    simg1 = cv2.resize(img1, (min_w, min_h), interpolation=cv2.INTER_AREA)
    simg2 = cv2.resize(img2, (min_w, min_h), interpolation=cv2.INTER_AREA)

    img1_padded = cv2.resize(img1, (int(max_w / scale), int(max_h / scale)), interpolation=cv2.INTER_AREA)
    img2_padded = cv2.resize(img2, (int(max_w / scale), int(max_h / scale)), interpolation=cv2.INTER_AREA)

    ###
    # perform image alignment
    ###

    # specify the motion model
    # warp_mode = cv2.MOTION_EUCLIDEAN   # cv2.MOTION_TRANSLATION, cv2.MOTION_EUCLIDEAN, cv2.MOTION_AFFINE, cv2.MOTION_HOMOGRAPHY
    warp_mode = cv2.MOTION_AFFINE  # cv2.MOTION_TRANSLATION, cv2.MOTION_EUCLIDEAN, cv2.MOTION_AFFINE, cv2.MOTION_HOMOGRAPHY

    # for testing purposes: img2 will be the reference image
    img1_aligned, matchArea1 = align_image(simg1, simg2, img2_padded, img1_padded, warp_mode)
    img1_aligned_cpy = img1_aligned.copy()
    cv2.rectangle(img1_aligned_cpy, (matchArea1[0], matchArea1[1]), (matchArea1[0] + matchArea1[2], matchArea1[1] + matchArea1[3]), (0, 255, 0), 2)

    print("\n###############################################\n")

    # compute the crop area in the reference image and draw a red rectangle
    # img2_eq_cpy = img2.copy()
    img2_eq_cpy = img2_padded.copy()
    cv2.rectangle(img2_eq_cpy, (matchArea1[0], matchArea1[1]), (matchArea1[0] + matchArea1[2], matchArea1[1] + matchArea1[3]), (0, 0, 255), 2)

    print("\n###############################################\n")

    # crop images to the smallest internal area between them
    img1_aligned_cropped = img1_aligned[matchArea1[1] : matchArea1[1] + matchArea1[3], matchArea1[0] : matchArea1[0] + matchArea1[2]]
    img2_eq_cropped = img2_padded[matchArea1[1] : matchArea1[1] + matchArea1[3], matchArea1[0] : matchArea1[0] + matchArea1[2]]

    cv2.imwrite(f"{options.output}img1.jpg", img1_aligned_cropped)
    cv2.imwrite(f"{options.output}img2.jpg", img2_eq_cropped)

    command = f"convert -delay 1 -loop 0 {options.output}img1.jpg {options.output}img2.jpg {options.output}_animated.gif"
    subprocess.run(command.split(" "))


if __name__ == "__main__":
    usage = "usage: %prog [options] arg1 arg2"
    parser = OptionParser(usage=usage)
    parser.add_option("-o", "--output", dest="output", help="output_file name FILE", metavar="FILE", default="output")
    parser.add_option("-s", "--scale", dest="scale", help="scale the picture", default=2, type="int")

    (options, args) = parser.parse_args()
    print(args)
    if len(args) != 2:
        parser.error("incorrect number of arguments")
    main(options, args)
