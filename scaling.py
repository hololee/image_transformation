import numpy as np
import matplotlib.pyplot as plt

# Load 'T' image  (200, 150, 1) rebuild  image.

img_source = plt.imread("source.png")

def scaling(image, scale, method):
    affine_matrix_identity = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])

    img_target = np.zeros([image.shape[0] * scale, image.shape[1] * scale])

    for x in range(image.shape[0] - 1):  # h
        for y in range(image.shape[1] - 1):  # w
            # Left Top
            map_lt = np.array([x, y, 1])
            x_lt, y_lt, _ = np.matmul(map_lt, affine_matrix_identity)

            for step_x in range(scale + 1):
                for step_y in range(scale + 1):
                    if method == "nearest_neighbor":
                        if step_x < (scale + 1) // 2 and step_y < (scale + 1) // 2:
                            img_target[x_lt + step_x, y_lt + step_y] = image[x, y]
                        elif step_x < (scale + 1) // 2 and step_y >= (scale + 1) // 2:
                            img_target[x_lt + step_x, y_lt + step_y] = image[x, y + 1]
                        elif step_x >= (scale + 1) // 2 and step_y < (scale + 1) // 2:
                            img_target[x_lt + step_x, y_lt + step_y] = image[x + 1, y]
                        elif step_x >= (scale + 1) // 2 and step_y >= (scale + 1) // 2:
                            img_target[x_lt + step_x, y_lt + step_y] = image[x + 1, y + 1]

                    elif method == "bilinear":
                        for step_x in range(scale + 1):
                            for step_y in range(scale + 1):
                                temp1 = image[x, y] * ((scale - step_x) / scale) + image[x, y + 1] * (step_x / scale)
                                temp2 = image[x + 1, y] * ((scale - step_x) / scale) + image[x + 1, y + 1] * (step_x / scale)
                                img_target[x_lt + step_x, y_lt + step_y] = temp1 * ((scale - step_y) / scale) + temp2 * (step_y / scale)

                    elif method == "bicubic":
                        try:
                            f00 = image[x, y]
                        except:
                            f00 = 0
                        try:
                            f01 = image[x, y + 1]
                        except:
                            f01 = 0
                        try:
                            f10 = image[x + 1, y]
                        except:
                            f10 = 0
                        try:
                            f11 = image[x + 1, y + 1]
                        except:
                            f11 = 0
                        try:
                            fx00 = (image[x + 1, y] - image[x - 1, y]) / 2
                        except:
                            fx00 = 0
                        try:
                            fx10 = (image[x + 2, y] - image[x, y]) / 2
                        except:
                            fx10 = 0
                        try:
                            fx01 = (image[x + 1, y + 1] - image[x - 1, y + 1]) / 2
                        except:
                            fx01 = 0
                        try:
                            fx11 = (image[x + 2, y + 1] - image[x, y + 1]) / 2
                        except:
                            fx11 = 0
                        try:
                            fy00 = (image[x, y + 1] - image[x, y - 1]) / 2
                        except:
                            fy00 = 0
                        try:
                            fy10 = (image[x, y + 2] - image[x, y]) / 2
                        except:
                            fy10 = 0
                        try:
                            fy01 = (image[x + 1, y + 1] - image[x + 1, y - 1]) / 2
                        except:
                            fy01 = 0
                        try:
                            fy11 = (image[x + 1, y + 2] - image[x + 1, y]) / 2
                        except:
                            fy11 = 0
                        try:
                            fxy00 = (image[x - 1, y - 1] - image[x - 1, y + 1] - image[x + 1, y - 1] + image[x + 1, y + 1]) / 4
                        except:
                            fxy00 = 0
                        try:
                            fxy10 = (image[x, y - 1] - image[x, y + 1] - image[x + 2, y - 1] + image[x + 2, y + 1]) / 4
                        except:
                            fxy10 = 0
                        try:
                            fxy01 = (image[x - 1, y] - image[x - 1, y + 2] - image[x + 1, y] + image[x + 1, y + 2]) / 4
                        except:
                            fxy01 = 0
                        try:
                            fxy11 = (image[x, y] - image[x, y + 2] - image[x + 2, y] + image[x + 2, y + 2]) / 4
                        except:
                            fxy11 = 0

                        for step_x in range(scale + 1):
                            for step_y in range(scale + 1):
                                temp1 = np.array([1, step_x, np.power(step_x, 2), np.power(step_x, 3)])
                                temp2 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [-3, 3, -2, -1], [2, -2, 1, 1]])
                                temp3 = np.array(
                                    [[f00, f01, fy00, fy01], [f10, f11, fy10, fy11], [fx00, fx01, fxy00, fxy01], [fx10, fx11, fxy10, fxy11]])
                                temp4 = np.array([[1, 0, -3, 2], [0, 0, 3, -2], [0, 1, -2, 1], [0, 0, -1, 1]])
                                temp5 = np.transpose(np.array([1, step_y, np.power(step_y, 2), np.power(step_y, 3)]))

                                img_target[x_lt + step_x, y_lt + step_y] = np.dot(np.dot(np.dot(np.dot(temp1, temp2), temp3), temp4), temp5)
    return img_target


scale_factor = 2

result_nearest_neighbor = scaling(img_source, scale_factor, "nearest_neighbor")
result_bilinear = scaling(img_source, scale_factor, "bilinear")
result_bicubic = scaling(img_source, scale_factor, "bicubic")

# draw
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey=True)
ax[0, 0].set_title("original")
ax[0, 0].imshow(img_source, cmap='gray')
ax[0, 1].set_title("scaling(nearest neighbor):x{}".format(scale_factor))
ax[0, 1].imshow(result_nearest_neighbor, cmap='gray')
ax[1, 0].set_title("scaling(bilinear):x{}".format(scale_factor))
ax[1, 0].imshow(result_bilinear, cmap='gray')
ax[1, 1].set_title("scaling(bicubic):x{}".format(scale_factor))
ax[1, 1].imshow(result_bicubic, cmap='gray')
plt.show()
