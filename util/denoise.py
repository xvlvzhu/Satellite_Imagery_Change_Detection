import numpy as np
import tifffile as tiff

mask = np.array([[0.5,1,0.5],[1,0,1],[0.5,1,0.5]])
iter=100
threshold=40

def label_prop(img,iter=10000,threshold=50):

    y=img
    w,h = y.shape
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
        if (temp_mask==y_iter).all():
            break
        else:
            y_iter = temp_mask

    #删去连通数小于阈值的区域
    unique, counts = np.unique(y_iter,return_counts=True)
    delete_id = unique[counts<=threshold]
    for i in delete_id:
        y_iter[y_iter==i]=0
    y_iter[y_iter>0]=1
    return y_iter.astype(np.uint8)

#1.计算邻居8个点
def cal_neighbor(i,j):
    # global a
    neighbor =  a[i-1:i+2,j-1:j+2]
    if a[i,j]==1:
        if np.sum(neighbor*mask)<=2:
            return 0
    # else:
    #     if np.sum(neighbor*mask)>=3.5:
    #         return 1
    return a[i,j]

def label_pruning(img):
    n, m = img.shape
    for i in range(n):
        for j in range(m):
            if img[i, j] == 1:
                print(i, j)
                q = [(i, j)]
                p = 0
                f = True
                while p < len(q):
                    x, y = q[p]
                    p += 1
                    if x + 1 < n and (x + 1, y) not in q and img[x + 1, y] == 1:
                        q.append((x + 1, y))
                    if x - 1 >= 0 and (x - 1, y) not in q and img[x - 1, y] == 1:
                        q.append((x - 1, y))
                    if y + 1 < m and (x, y + 1) not in q and img[x, y + 1] == 1:
                        q.append((x, y + 1))
                    if y - 1 >= 0 and (x, y - 1) not in q and img[x, y - 1] == 1:
                        q.append((x, y - 1))
                    # if x + 1 < n and y + 1 < m and (x + 1, y + 1) not in q and img[x + 1, y + 1] == 1:
                    #     q.append((x + 1, y + 1))
                    # if x + 1 < n and y - 1 >= 0 and (x + 1, y - 1) not in q and img[x + 1, y - 1] == 1:
                    #     q.append((x + 1, y - 1))
                    # if x - 1 >= 0 and y + 1 < m and (x - 1, y + 1) not in q and img[x - 1, y + 1] == 1:
                    #     q.append((x - 1, y + 1))
                    # if x - 1 >= 0 and y - 1 >= 0 and (x - 1, y - 1) not in q and img[x - 1, y - 1] == 1:
                    #     q.append((x - 1, y - 1))
                    if len(q) > 5:
                        f = False
                        break
                if f:
                    for x, y in q:
                        img[x, y] = 0
    return img

if __name__ == '__main__':

    res = label_prop(tiff.imread("D:/数据/天池/tif/out_11_10_drop.tiff"))
    tiff.imsave("D:/数据/天池/tif/out_11_11_LP_1.tiff",res)

    y = res
    w = np.shape(y)[0]
    h = np.shape(y)[1]
    print([w+2,h+2])
    # global a
    a = np.zeros([w+2,h+2])
    b = np.zeros([w+2,h+2])

    a[1:w+1,1:h+1] = y
    for m in range(10):
        for i in range(1,w+1):
            print(i)
            for j in range(1,h+1):
                b[i,j] = cal_neighbor(i,j)

    out = b[1:w+1,1:h+1]
    print(out.shape)
    tiff.imsave("D:\\数据\\天池\\tif\\out_11_11_D_LP.tiff",out.astype(np.uint8))