import numpy as np
import tifffile as tiff

iter=100
threshold=10

def label_prop(iter=100,threshold=10):
    y = tiff.imread("D:/数据/天池/tif/res_11_10_L.tiff")
    w,h = y.shape
    # y1 = np.zeros(y.shape) #4000*15106
    # # for m in range(np.sum(y)):
    # #初始化标签
    # m=1
    # for i in range(w):
    #     print(i)
    #     for j in range(h):
    #         if y[i,j]==1:
    #             y1[i,j]=m
    #             m = m+1
    y1 = np.arange(w*h,dtype=np.uint32).reshape([w,h])*y

    #标签传播
    y_iter = y1.astype(dtype=np.uint32)
    for i in range(iter):
        print(i)
        temp = np.zeros([w+2,h+2,5]) #4000*15106*5
        temp[1:w+1,1:h+1,0] = y_iter
        temp[0:w,1:h+1,1] = y_iter
        temp[2:w+2,1:h+1,2] = y_iter
        temp[1:w+1,2:h+2,3] = y_iter
        temp[1:w+1,0:h,4] = y_iter

        temp = np.max(temp,axis=2)
        temp_mask = temp[1:w+1,1:h+1]*y
        # print(temp_mask)
        # print(temp_mask.shape)
        if (temp_mask==y_iter).all():
            break
        else:
            # y_iter[y==1] = temp[1:w+1,1:h+1][y==1]
            # print(np.sqrt(np.sum(np.square(temp_mask-y_iter))))
            y_iter = temp_mask


    # print(temp.shape)

    #删去连通数小于阈值的区域
    unique, counts = np.unique(y_iter,return_counts=True)
    delete_id = unique[counts<=threshold]
    for i in delete_id:
        y_iter[y_iter==i]=0

    y_iter[y_iter>0]==1
    return y_iter.astype(np.uint8)

if __name__ == '__main__':
    res = label_prop()
    tiff.imsave("D:/数据/天池/tif/res_11_10_LP1.tiff",res)