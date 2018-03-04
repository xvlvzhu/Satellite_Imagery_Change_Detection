import cv2
import numpy as np
import tifffile as tiff

def scale_percentile(matrix):
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0) - mins
    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    # print(matrix)
    return matrix

def tiff2ImgSclice(tiff_file1,tiff_file2,img_size=512):
    FILE_1 = 'D:/数据/天池/20171105_quarterfinals/quarterfinals_2015.tif'
    FILE_2 = 'D:/数据/天池/20171105_quarterfinals/quarterfinals_2017.tif'
    # FILE_cadastral2015 = 'C:\\Users\\zxl\\Desktop\\20170905_preliminary\\20170907_hint\\cadastral2015.tif'
    # FILE_tinysample = 'C:\\Users\\zxl\\Desktop\\20170905_preliminary\\20170907_hint\\tinysample.tif'
    # FILE_label = 'C:\\Users\\zxl\\Desktop\\20170905_preliminary\\20171026_初赛第一阶段的建筑物变化标记(new)\\answer_complete.tif'
    # FILE_label = 'D:/数据/天池/tif/out_11_10_D_LP.tiff'

    im_1 = tiff.imread(FILE_1).transpose([1, 2, 0])
    im_2 = tiff.imread(FILE_2).transpose([1, 2, 0])
    # im_tiny = tiff.imread(FILE_tinysample)
    # im_cada = tiff.imread(FILE_cadastral2015)
    # im_label = tiff.imread(FILE_label)

    print(im_1.shape)
    print(im_2.shape)
    # img_size = 512 # 15106/ 256 =59...2  5106/256=19..284

    for i in range(int(len(im_1)/img_size) + 1 ): # last 284
        for j in range(int(len(im_1[0])/img_size) ): #last 2 too small, drop one
            im_name = str(i)+'_'+str(j)+'_'+str(img_size)+'_.jpg'
            cv2.imwrite("2017_"+im_name,scale_percentile(im_2[i*img_size:i*img_size+img_size, j*img_size:j*img_size+img_size, :3])*255)
            cv2.imwrite("2015_"+im_name,scale_percentile(im_1[i*img_size:i*img_size+img_size, j*img_size:j*img_size+img_size, :3])*255)
        # cv2.imshow('img',scale_percentile(im_2[i*img_size:i*img_size+img_size, j*img_size:j*img_size+img_size, :3])*255)
        # cv2.imwrite("cada_"+im_name,im_cada[i*img_size:i*img_size+img_size, j*img_size:j*img_size+img_size]*255)
        # cv2.imwrite("tiny_"+im_name,im_tiny[i*img_size:i*img_size+img_size, j*img_size:j*img_size+img_size]*255)
        # cv2.imwrite("label_" + im_name,im_label[i * img_size:i * img_size + img_size, j * img_size:j * img_size + img_size] * 255)

if __name__ == '__main__':
    tiff2ImgSclice('D:/数据/天池/20171105_quarterfinals/quarterfinals_2015.tif','D:/数据/天池/20171105_quarterfinals/quarterfinals_2017.tif')