import cv2
import numpy as np
import tifffile as tiff
img_size = 512 # 15106/ 256 =59...2  5106/256=19..284
im_tiny = np.zeros([4000,15106,1],np.uint8)
for i in range(int(len(im_tiny)/img_size) + 1 ): # last 284
    for j in range(int(len(im_tiny[0])/img_size) ): #last 2 too small, drop one
        im_name = 'mod_'+str(i)+'_'+str(j)+'_'+str(img_size)+'_.jpg'
        img = cv2.imread("D:/dataAnnotation/20171105_quarterfinals/mylabel_mod_11_12/"+im_name,0)
        # if img != None:
        print(i)
        _,img = cv2.threshold(img,10,255,cv2.THRESH_BINARY)
        im_tiny[i*img_size:i*img_size+img_size, j*img_size:j*img_size+img_size,0] = img

a = im_tiny[:,:,0]>0
a = np.array(a,np.uint8)
tiff.imsave('out_11_12_mod.tiff',a)