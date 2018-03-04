import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff

# res = tiff.imread("C:\\Users\\zxl\\Desktop\\20170905_preliminary\\20171026_初赛第一阶段的建筑物变化标记(new)\\answer_complete.tif")
res = tiff.imread("D:/数据/天池/tif/res_11_10_L.tiff")
res1 = tiff.imread("D:/数据/天池/tif/out_11_8_h.tiff")

print(np.sum(res1*res)/np.sum(res1))