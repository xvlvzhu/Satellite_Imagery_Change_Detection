import numpy as np

q = np.load("train_y_tif8.npy")
# np.load("train_x_tif8")
print(q.shape)
print(q.dtype)
print(q.max())
print(q.min())