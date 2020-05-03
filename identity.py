import numpy as np
import matplotlib.pyplot as plt

# Load 'T' image  (200, 150, 1) rebuild  image.

img_source = plt.imread("source.png")

# 1.question 1

affine_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
img_target = np.zeros(img_source.shape)

# mapping source to target.
for x in range(img_source.shape[0]):  # h
    for y in range(img_source.shape[1]):  # w
        original = np.array([x, y, 1])
        x_new, y_new, _ = np.matmul(original, affine_matrix)
        img_target[int(x_new), int(y_new)] = img_source[x, y]

# draw
plt.rcParams["figure.figsize"] = (8, 4)
plt.subplot(1, 2, 1)
plt.title("original")
plt.imshow(img_source, cmap="gray")
plt.subplot(1, 2, 2)
plt.title("Identity")
plt.imshow(img_target, cmap="gray")
plt.show()
