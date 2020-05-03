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
            H[int(img[x, y])] += 1

    return H


def image_equalization(img, img_text):
    H = get_histogram(img)

    g_min = np.min(H)

    Hc = np.zeros(shape=H.shape)
    for index, H_val in enumerate(H):
        if index == 0:
            Hc[index] = H_val

        else:
            Hc[index] = Hc[index - 1] + H[index]

    H_min = Hc[int(g_min)]

    # draw transformation.
    draw_histogram(Hc, "cumulative(transformation) histogram {}".format(img_text))

    T = np.zeros(shape=H.shape)
    for index, H_val in enumerate(H):
        T[index] = np.round(((Hc[index] - H_min) * 255 / (img.shape[0] * img.shape[1] - H_min)))

    img_result = np.zeros(shape=img.shape)
    for x in range(img_result.shape[0]):
        for y in range(img_result.shape[1]):
            img_result[x, y] = T[img[x, y]]

    return img_result


def draw_histogram(histogram, text):
    plt.title(text)
    plt.bar(range(256), histogram)
    plt.savefig("C:/Users/jh_work/PycharmProjects/image_transformation/results2/{}.png".format(text))
    plt.show()


# # img1
img_result = image_equalization(img1, "img1")

draw_histogram(get_histogram(img1), "histogram img1")
draw_histogram(get_histogram(img_result), "histogram img1_result")

plt.imshow(img1, cmap="gray")
plt.title("img1")
plt.savefig("C:/Users/jh_work/PycharmProjects/image_transformation/results2/{}.png".format("img1"))
plt.show()

plt.imshow(img_result, cmap="gray")
plt.title("img1_result")
plt.savefig("C:/Users/jh_work/PycharmProjects/image_transformation/results2/{}.png".format("img1_result"))
plt.show()

# # img2
img_result = image_equalization(img2, "img2")

draw_histogram(get_histogram(img2), "histogram img2")
draw_histogram(get_histogram(img_result), "histogram img2_result")

plt.imshow(img2, cmap="gray")
plt.title("img2")
plt.savefig("C:/Users/jh_work/PycharmProjects/image_transformation/results2/{}.png".format("img2"))
plt.show()
plt.imshow(img_result, cmap="gray")
plt.title("img2_result")
plt.savefig("C:/Users/jh_work/PycharmProjects/image_transformation/results2/{}.png".format("img2_result"))
plt.show()
