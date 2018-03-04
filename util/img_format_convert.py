import numpy as np
import cv2
import tifffile as tiff

def jpg2tiff(read_path,save_path):

    img = cv2.imread(read_path,cv2.IMREAD_GRAYSCALE)
    tiff.imsave(save_path,img)

def tiff2jpg(read_path,save_path):

    a = tiff.imread(read_path)
    a = a.astype(np.uint8)
    a[a != 0] = 255
    cv2.imwrite(save_path, a)

def npy2tiff(read_path,save_path):

    out = np.load(read_path)
    print(np.shape(out))
    print(out.sum())
    result = np.array(out, dtype=np.uint8)
    tiff.imsave(save_path, result)

def tiff2npy(read_path,save_path):

    img = tiff.imread(read_path)
    np.save(save_path,img)