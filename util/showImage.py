import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff

# res = tiff.imread("C:\\Users\\zxl\\Desktop\\20170905_preliminary\\20171026_初赛第一阶段的建筑物变化标记(new)\\answer_complete.tif")
res = tiff.imread("D:/数据/天池/tif/res_11_10_LP1.tiff")
res[res!=0]=1
tiff.imsave("D:/数据/天池/tif/res_11_10_LP1.tiff",res)
# res = np.load("out_10_31.npy")
print(res.shape)
plt.imshow(res)
# plt.imshow(res1)
plt.show()
print(np.sum(np.sum(res)))
print(np.max(np.max(res)))
print(np.min(res,axis=(0,1)))
print(4000*15106-np.sum(np.sum(res)))
