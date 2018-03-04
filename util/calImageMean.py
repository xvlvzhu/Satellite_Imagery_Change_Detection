import numpy as np
import tifffile as tiff
import cv2
# FILE_2015 = 'C:\\Users\\zxl\\Desktop\\20170905_preliminary\\preliminary\\quickbird2015.tif'
# FILE_2017 = 'C:\\Users\\zxl\\Desktop\\20170905_preliminary\\preliminary\\quickbird2017.tif'
#
# im_2015 = tiff.imread(FILE_2015).transpose([1, 2, 0])
# im_2017 = tiff.imread(FILE_2017).transpose([1, 2, 0])
#
# def scale_percentile(matrix):
#     w, h, d = matrix.shape
#     matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
#     # Get 2nd and 98th percentile
#     mins = np.percentile(matrix, 1, axis=0)
#     maxs = np.percentile(matrix, 99, axis=0) - mins
#     matrix = (matrix - mins[None, :]) / maxs[None, :]
#     matrix = np.reshape(matrix, [w, h, d])
#     matrix = matrix.clip(0, 1)
#     # print(matrix)
#     return matrix
#
# print(tiff.imread(FILE_2015).shape)
# im_2015_3 = scale_percentile(im_2015[:,:, :3])*255
# im_2017_3 = scale_percentile(im_2017[:,:, :3])*255
# im_2015_mean = np.mean(im_2015_3,axis=(0,1),dtype=np.float32)
# print(im_2015_mean)
# print(im_2015[0])

img_size = 960 # 15106/ 256 =59...2  5106/256=19..284

im_tiny_2017 = np.zeros([5106,15106,3],np.uint8)
for i in range(int(len(im_tiny_2017)/img_size) + 1 ): # last 284
    for j in range(int(len(im_tiny_2017[0])/img_size) ): #last 2 too small, drop one
        im_name = str(i)+'_'+str(j)+'_'+str(img_size)+'_.jpg'
        img_2017 = cv2.imread("D:\\dataAnnotation\\preliminary_1\\originalData_960\\2017_"+im_name)

        im_tiny_2017[i*img_size:i*img_size+img_size, j*img_size:j*img_size+img_size,:3] = img_2017

print(np.mean(im_tiny_2017,axis=(0,1)))