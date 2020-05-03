import numpy as np
import matplotlib.pyplot as plt

img_source = plt.imread("source.png")


def shear_vertical(image, vertical, method):
    img_target = np.zeros(shape=image.shape)

    affine_matrix = np.array([[1, 0, 0], [vertical, 1, 0], [0, 0, 1]])

    # target(rotated coordinate) (x, y)
    for x in range(image.shape[0] - 1):  # h
        for y in range(image.shape[1] - 1):  # w
            map = np.array([x, y, 1])

            # original image coordinate (v, w)
            v, w, _ = np.matmul(map, np.linalg.inv(affine_matrix))

            x_lt = int(v)
            y_lt = int(w)

            x_rt = int(v)
            y_rt = int(w + 1)

            x_lb = int(v + 1)
            y_lb = int(w)

            x_rb = int(v + 1)
            y_rb = int(w + 1)

            if method == "nearest_neighbor":
                try:
                    img_target[x, y] = image[x_lt, y_lt]
                except:
                    img_target[x, y] = 0

            elif method == "bilinear":

                try:
                    img_lt = image[x_lt, y_lt]
                except:
                    img_lt = 0

                try:
                    img_rt = image[x_rt, y_rt]
                except:
                    img_rt = 0

                try:
                    img_lb = image[x_lb, y_lb]
                except:
                    img_lb = 0

                try:
                    img_rb = image[x_rb, y_rb]
                except:
                    img_rb = 0

                temp1 = img_lt * (y_rt - w) + img_rt * (1 - (y_rt - w))
                temp2 = img_lb * (y_rt - w) + img_rb * (1 - (y_rt - w))
                img_target[x, y] = temp1 * (x_lb - v) + temp2 * (1 - (x_lb - v))

            elif method == "bicubic":
                try:
                    f00 = image[x_lt, y_lt]
                except:
                    f00 = 0
                try:
                    f01 = image[x_lt, y_lt + 1]
                except:
                    f01 = 0
                try:
                    f10 = image[x_lt + 1, y_lt]
                except:
                    f10 = 0
                try:
                    f11 = image[x_lt + 1, y_lt + 1]
                except:
                    f11 = 0
                try:
                    fx00 = (image[x_lt + 1, y_lt] - image[x_lt - 1, y_lt]) / 2
                except:
                    fx00 = 0
                try:
                    fx10 = (image[x_lt + 2, y_lt] - image[x_lt, y_lt]) / 2
                except:
                    fx10 = 0
                try:
                    fx01 = (image[x_lt + 1, y_lt + 1] - image[x_lt - 1, y_lt + 1]) / 2
                except:
                    fx01 = 0
                try:
                    fx11 = (image[x_lt + 2, y_lt + 1] - image[x_lt, y_lt + 1]) / 2
                except:
                    fx11 = 0
                try:
                    fy00 = (image[x_lt, y_lt + 1] - image[x_lt, y_lt - 1]) / 2
                except:
                    fy00 = 0
                try:
                    fy10 = (image[x_lt, y_lt + 2] - image[x_lt, y_lt]) / 2
                except:
                    fy10 = 0
                try:
                    fy01 = (image[x_lt + 1, y_lt + 1] - image[x_lt + 1, y_lt - 1]) / 2
                except:
                    fy01 = 0
                try:
                    fy11 = (image[x_lt + 1, y_lt + 2] - image[x_lt + 1, y_lt]) / 2
                except:
                    fy11 = 0
                try:
                    fxy00 = (image[x_lt - 1, y_lt - 1] - image[x_lt - 1, y_lt + 1] - image[x_lt + 1, y_lt - 1] + image[x_lt + 1, y_lt + 1]) / 4
                except:
                    fxy00 = 0
                try:
                    fxy10 = (image[x_lt, y_lt - 1] - image[x_lt, y_lt + 1] - image[x_lt + 2, y_lt - 1] + image[x_lt + 2, y_lt + 1]) / 4
                except:
                    fxy10 = 0
                try:
                    fxy01 = (image[x_lt - 1, y_lt] - image[x_lt - 1, y_lt + 2] - image[x_lt + 1, y_lt] + image[x_lt + 1, y_lt + 2]) / 4
                except:
                    fxy01 = 0
                try:
                    fxy11 = (image[x_lt, y_lt] - image[x_lt, y_lt + 2] - image[x_lt + 2, y_lt] + image[x_lt + 2, y_lt + 2]) / 4
                except:
                    fxy11 = 0
                # ----------------------distance coordinates --------distance coordinates------------distance coordinates-------
                temp1 = np.array([1, (1 - (x_lb - v)), np.power((1 - (x_lb - v)), 2), np.power((1 - (x_lb - v)), 3)])
                temp2 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [-3, 3, -2, -1], [2, -2, 1, 1]])
                temp3 = np.array(
                    [[f00, f01, fy00, fy01], [f10, f11, fy10, fy11], [fx00, fx01, fxy00, fxy01], [fx10, fx11, fxy10, fxy11]])
                temp4 = np.array([[1, 0, -3, 2], [0, 0, 3, -2], [0, 1, -2, 1], [0, 0, -1, 1]])
                temp5 = np.transpose(np.array([1, (1 - (y_rt - w)), np.power((1 - (y_rt - w)), 2), np.power((1 - (y_rt - w)), 3)]))

                img_target[x, y] = np.dot(np.dot(np.dot(np.dot(temp1, temp2), temp3), temp4), temp5)

    return img_target


vertical = 0.2

result_nearest_neighbor = shear_vertical(img_source, vertical, method="nearest_neighbor")
result_bilinear = shear_vertical(img_source, vertical, method="bilinear")
result_bicubic = shear_vertical(img_source, vertical, method="bicubic")

# draw
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey=True)
ax[0, 0].set_title("original")
ax[0, 0].imshow(img_source, cmap='gray')
ax[0, 1].set_title("shear_vertical(nearest neighbor):{}'".format(np.abs(vertical)))
ax[0, 1].imshow(result_nearest_neighbor, cmap='gray')
ax[1, 0].set_title("shear_vertical(bilinear):{}'".format(np.abs(vertical)))
ax[1, 0].imshow(result_bilinear, cmap='gray')
ax[1, 1].set_title("shear_vertical(bicubic):{}".format(np.abs(vertical)))
ax[1, 1].imshow(result_bicubic, cmap='gray')
plt.show()
