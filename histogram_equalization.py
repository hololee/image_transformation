import numpy as np
import matplotlib.pyplot as plt

img1 = 255 * plt.imread("histogram1.png")
img2 = 255 * plt.imread("histogram2.png")

img1 = img1.astype(int)
img2 = img2.astype(int)


def get_histogram(img):
    H = np.zeros(256)

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            H[img[x, y]] += 1

    return H


def image_equalization(img):
    H = get_histogram(img)

    g_min = np.min(H)
    H_min = np.where(H == g_min)

    Hc = np.zeros(shape=H.shape)
    for index, H_val in enumerate(H):
        if index == 0:
            Hc[index] = H_val

        else:
            Hc[index] = Hc[index - 1] + H[index]

    T = np.zeros(shape=H.shape)
    for index, H_val in enumerate(H):
        T[index] = np.round(((Hc[index] - H_min[0]) * 255 / (img.shape[0] * img.shape[1] - H_min[0])))

    img_result = np.zeros(shape=img.shape)
    for x in range(img_result.shape[0]):
        for y in range(img_result.shape[1]):
            img_result[x, y] = T[img[x, y]]

    return img_result


img_result = image_equlization(img1)

plt.imshow(img1, cmap="gray")
plt.show()
plt.imshow(img_result, cmap="gray")
plt.show()
