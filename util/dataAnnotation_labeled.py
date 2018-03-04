import cv2
import os

# DATA_DIR = "/Volumes/DATA/train"
DATA_DIR = "D:/dataAnnotation"

# IM_ROWS = 5106
IM_ROWS = 5106
IM_COLS = 15106
ROI_SIZE = 256
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

def label_img(img,img1,tiny1,cata,label, label_name):
    c = 'x'
    tiny = np.zeros(img.shape)
    while c != 'n':
        cv2.namedWindow('img', 0)
        cv2.namedWindow('img_2015', 0)
        cv2.namedWindow('img_tiny', 0)
        cv2.namedWindow('img_cata', 0)
        cv2.namedWindow('img_label', 0)
        temp = img.copy()
        points = []
        cv2.setMouseCallback('img', on_mouse, {'img': temp, 'points': points})
        cv2.imshow('img', img)
        cv2.imshow('img_2015', img1)
        cv2.imshow('img_tiny', tiny1)
        cv2.imshow('img_cata', cata)
        cv2.imshow('img_label', label)
        c = chr(cv2.waitKey(0))

        if c == 's':

            if len(points) > 0:
                cv2.fillPoly(img, [np.array(points)], (0, 0, 255))
                cv2.fillPoly(tiny, [np.array(points)], (255, 255, 255))
    print(label_name)
    # cv2.imwrite(label_name, tiny)
    # cv2.imencode(label_name, tiny)[1].tofile(_path)
    return

if __name__ == '__main__':

    # path_file = "D:\\数据\\天池\\数据标注\\2017_0_0_256_.jpg"
    # img = cv2.imdecode(np.fromfile(path_file, dtype=np.uint8), -1)

    for i in range(int(IM_ROWS // ROI_SIZE)+1):
    # for i in range(10,12):
        for j in range(int(IM_COLS // ROI_SIZE)):
            # ss1 = '{}/2017/{}_{}_{}_.jpg'.format(DATA_DIR, i, j, ROI_SIZE)
            ss1_2017 = '{}/origindata/2017_{}_{}_{}_.jpg'.format(DATA_DIR, i, j, ROI_SIZE)
            ss1_2015 = '{}/origindata/2015_{}_{}_{}_.jpg'.format(DATA_DIR, i, j, ROI_SIZE)
            ss1_tiny = '{}/origindata/tiny_{}_{}_{}_.jpg'.format(DATA_DIR, i, j, ROI_SIZE)
            ss1_cada = '{}/origindata/cada_{}_{}_{}_.jpg'.format(DATA_DIR, i, j, ROI_SIZE)
            ss1_label = '{}/origindata/label_{}_{}_{}_.jpg'.format(DATA_DIR, i, j, ROI_SIZE)

            # print(ss1)
            ss2 = '{}/mylabel_2017_2/{}_{}_{}_.jpg'.format(DATA_DIR, i, j, ROI_SIZE)
            if os.path.exists(ss2):
                continue
            src = cv2.imread(ss1_2017, cv2.IMREAD_UNCHANGED)
            src2 = cv2.imread(ss1_2015, cv2.IMREAD_UNCHANGED)
            src3 = cv2.imread(ss1_tiny, cv2.IMREAD_UNCHANGED)
            src4 = cv2.imread(ss1_cada, cv2.IMREAD_UNCHANGED)
            src5 = cv2.imread(ss1_label, cv2.IMREAD_UNCHANGED)
            # src = cv2.imdecode(np.fromfile(ss1, dtype=np.uint8), -1)
            label_img(src,src2,src3,src4,src5, ss2)