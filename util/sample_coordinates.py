# -*- coding: utf8 -*-

# 提取正负样本坐标

import numpy as np
import tifffile as tiff

label_path = "./data/out_11_8_h.tiff"

# 标签数据，二维数组形式，第一维4000，第二维15106
label = tiff.imread(label_path)
label_dimension = label.shape
print(label_dimension)
print(np.unique(label))


# 取样本数据坐标
def get_sample_c():
    positive_c = []
    negative_c = []

    for i in range(label_dimension[0]):
        for j in range(label_dimension[1]):
            if label[i][j] == 1:
                positive_c.append((i, j))
            elif label[i][j] == 0:
                negative_c.append((i, j))

        print("* processing line %s" % i)

    return {"positive": np.array(positive_c), "negative": np.array(negative_c)}


c = get_sample_c()
print(c["positive"].shape)
print(c["negative"].shape)
np.save("./npy/positive_c", c["positive"])
np.save("./npy/negative_c", c["negative"])
