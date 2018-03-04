import cv2
import os
# from pymouse import *
# DATA_DIR = "/Volumes/DATA/train"
DATA_DIR = "D:/dataAnnotation/20171105_quarterfinals"

# IM_ROWS = 5106
IM_ROWS = 4000
IM_COLS = 15106
ROI_SIZE = 512
import numpy as np
def on_mouse(event, x, y, flags, params):
    img, points = params['img'], params['points']
    if event == cv2.EVENT_FLAG_LBUTTON:
        points.append((x, y))

    if event == cv2.EVENT_FLAG_RBUTTON:
        points.pop()

    temp = img.copy()
    if len(points) > 2:
        cv2.fillPoly(temp, [np.array(points)], (0, 0, 255))

    for i in range(len(points)):
        cv2.circle(temp, points[i], 1, (0, 0, 255))

    cv2.circle(temp, (x, y), 1, (0, 255, 0))
    cv2.imshow('img', temp)

def label_img(img,img1,label,out, label_name):
    c = 'x'
    tiny = np.zeros(img.shape)
    while c != 'n':
        cv2.namedWindow('img', 0)
        cv2.namedWindow('img_2015', 0)
        cv2.namedWindow('img_label', 0)
        cv2.namedWindow('img_out', 0)
        temp = img.copy()
        points = []
        cv2.setMouseCallback('img', on_mouse, {'img': temp, 'points': points})
        cv2.imshow('img', img)
        cv2.imshow('img_2015', img1)
        cv2.imshow('img_label', label)
        cv2.imshow('img_out', out)
        c = chr(cv2.waitKey(0))

        if c == 's':

            if len(points) > 0:
                cv2.fillPoly(img, [np.array(points)], (0, 0, 255))

                cv2.fillPoly(tiny, [np.array(points)], (255, 255, 255))

        if c == 'x':
            cv2.imwrite(label_name, tiny)
            break
        # if c == 'l':
        #
        #     move(x, y)

    print(label_name)

    return

if __name__ == '__main__':


    for i in range(int(IM_ROWS // ROI_SIZE)+1):
    # for i in range(10,12):
        for j in range(int(IM_COLS // ROI_SIZE)):
            ss1_2017 = '{}/cut_512/2017_{}_{}_{}_.jpg'.format(DATA_DIR, i, j, ROI_SIZE)
            ss1_2015 = '{}/cut_512/2015_{}_{}_{}_.jpg'.format(DATA_DIR, i, j, ROI_SIZE)
            label = '{}/mylabel/{}_{}_{}_.jpg'.format(DATA_DIR, i, j, ROI_SIZE)
            out = '{}/out2/label_{}_{}_{}_.jpg'.format(DATA_DIR, i, j, ROI_SIZE)
            ss2 = '{}/mylabel_mod_11_12_1/mod_{}_{}_{}_.jpg'.format(DATA_DIR, i, j, ROI_SIZE)
            if os.path.exists(ss2):
                continue
            src = cv2.imread(ss1_2017, cv2.IMREAD_UNCHANGED)
            src2 = cv2.imread(ss1_2015, cv2.IMREAD_UNCHANGED)
            src3 = cv2.imread(label, cv2.IMREAD_UNCHANGED)
            src4 = cv2.imread(out, cv2.IMREAD_UNCHANGED)
            label_img(src,src2,src3,src4, ss2)