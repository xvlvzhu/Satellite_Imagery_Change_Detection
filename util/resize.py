import cv2
import numpy as np
import tifffile as tiff
img_size = 960 # 15106/ 256 =59...2  5106/256=19..284

im_tiny_2015 = np.zeros([5106,15106,1],np.uint8)
im_tiny_2017 = np.zeros([5106,15106,1],np.uint8)
for i in range(int(len(im_tiny_2017)/img_size) + 1 ): # last 284
    for j in range(int(len(im_tiny_2017[0])/img_size) ): #last 2 too small, drop one
        im_name = str(i)+'_'+str(j)+'_'+str(img_size)+'_.jpg'
        img_2015 = cv2.imread("D:\\dataAnnotation\\preliminary_1\\2015\\"+im_name,0)
        img_2017 = cv2.imread("D:\\dataAnnotation\\preliminary_1\\2017\\"+im_name,0)

        im_tiny_2015[i*img_size:i*img_size+img_size, j*img_size:j*img_size+img_size,0] = img_2015
        im_tiny_2017[i*img_size:i*img_size+img_size, j*img_size:j*img_size+img_size,0] = img_2017

im_2015 = im_tiny_2015[:,:,0]>0
im_2015 = np.array(im_2015,np.uint8)
im_2017 = im_tiny_2017[:,:,0]>0
im_2017 = np.array(im_2017,np.uint8)
# tiff.imsave('out_11_3_origin_2017.tiff',a)
# print(a.shape)
# print(im_tiny.shape)
# print(int(len(im_tiny[0])))

division_size = 256

for i in range(int(len(im_tiny_2017)/division_size) + 1 ): # last 284
    for j in range(int(len(im_tiny_2017[0])/division_size) ): #last 2 too small, drop one
        im_name = str(i)+'_'+str(j)+'_'+str(division_size)+'_.jpg'
        cv2.imwrite("2017_"+im_name,im_2017[i*division_size:i*division_size+division_size, j*division_size:j*division_size+division_size]*255)
        cv2.imwrite("2015_"+im_name,im_2015[i*division_size:i*division_size+division_size, j*division_size:j*division_size+division_size]*255)