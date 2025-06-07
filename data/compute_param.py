import cv2
import os
from os import listdir
from os.path import isfile, join
from PIL import Image as Image
import numpy as np

from scipy.optimize import curve_fit


def relit(x, a, b):
    return np.array((a * x / 255.0 + b) * 255.0,dtype=float)


from matplotlib import pyplot as plt


def plshow(im, title='MINE'):
    if len(im.shape) > 2:
        #      plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        plt.imshow(im)
    else:
        plt.imshow(im, cmap='gray')
    plt.title(title)
    plt.rcParams["figure.figsize"] = (80, 12)
    plt.show()


sd_path = r'../datasets/UAV-SC/shadow'
mask_path = r'../datasets/UAV-SC/mask'
sdfree_path = r'../datasets/UAV-SC/shadowfree'

out = r'../datasets'

if not os.path.exists(out):
    os.makedirs(out)

im_list = [f for f in listdir(sd_path) if isfile(join(sd_path, f)) and f.endswith('tif')]
print(len(im_list), im_list[0])
kernel = np.ones((5, 5), np.uint8)


def im_relit(Rpopt, Gpopt, Bpopt, dump):
    # some weird bugs with python
    sdim = dump.copy()
    sdim.setflags(write=1)
    sdim = np.array(sdim,dtype=float)
    sdim[:, :, 0] = (sdim[:, :, 0] / 255) * Rpopt[0] + Rpopt[1]
    sdim[:, :, 1] = (sdim[:, :, 1] / 255) * Gpopt[0] + Gpopt[1]
    sdim[:, :, 2] = (sdim[:, :, 2] / 255) * Bpopt[0] + Bpopt[1]
    sdim = sdim * 255
    # print(np.amin(sdim),np.amax(sdim))
    return sdim


errors = []

for im in im_list:
    sd = np.asarray(Image.open(join(sd_path, im)))
    mean_sdim = np.mean(sd, axis=2)

    mask_ori = np.asarray(Image.open(join(mask_path, im)))
    mask = cv2.erode(mask_ori, kernel, iterations=2)

    # sdfree = (1 - mask/255) * (sd)
    sdfree = np.asarray(Image.open(join(sdfree_path, im)))
    mean_sdfreeim = np.mean(sdfree, axis=2)

    # Ensure mask is single-channel
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]  # Extract the first channel

    # Perform the operation
    i, j = np.where(
        np.logical_and(
            np.logical_and(
                np.logical_and(mask >= 1, mean_sdim > 5),
                mean_sdfreeim < 230
            ),
            np.abs(mean_sdim - mean_sdfreeim) > 10
        )
    )
    q = len(i)
    if len(i) == 0:
        continue

    # i, j = np.where(np.logical_and(mask >= 1, np.abs(mean_sdim - mean_sdfreeim) > 10))
    else:
        source = sd * 0
        source[tuple([i, j])] = sd[tuple([i, j])]
        target = sd * 0
        target[tuple([i, j])] = sdfree[tuple([i, j])]

        R_s = source[:, :, 0][tuple([i, j])]
        G_s = source[:, :, 1][tuple([i, j])]
        B_s = source[:, :, 2][tuple([i, j])]

        R_t = target[:, :, 0][tuple([i, j])]
        G_t = target[:, :, 1][tuple([i, j])]
        B_t = target[:, :, 2][tuple([i, j])]

        c_bounds = [[1, -0.1], [10, 0.5]]

        Rpopt, pcov = curve_fit(relit, R_s, R_t, bounds=c_bounds)
        Gpopt, pcov = curve_fit(relit, G_s, G_t, bounds=c_bounds)
        Bpopt, pcov = curve_fit(relit, B_s, B_t, bounds=c_bounds)

        relitim = im_relit(Rpopt, Gpopt, Bpopt, sd)

        final = sd.copy()
        final[tuple([i, j])] = relitim[tuple([i, j])]
        final[final > 255] = 255
        final[final < 0] = 0

        # plshow(final)

        error = np.mean(np.array(np.abs(np.array(relitim[tuple([i, j])] - sdfree[tuple([i, j])]),dtype=float)))
        print(error, Rpopt, Gpopt, Bpopt)
        f = open(join(out, 'parameters.txt'), "a")
        WR = 1 /Rpopt[0]
        BR = Rpopt[1]/Rpopt[0] *(-1)
        WG = 1 /Gpopt[0]
        BG = Gpopt[1]/Gpopt[0] *(-1)
        WB = 1 /Bpopt[0]
        BB = Bpopt[1]/Bpopt[0] *(-1)
        f.write("%f %f %f %f %f %f \n" % (BR, WR, BG, WG, BB, WB))
        # f.write("%f %f %f %f %f %f \n" % (Rpopt[1], Rpopt[0], Gpopt[1], Gpopt[0], Bpopt[1], Bpopt[0]))
        f.close()

        #  print(error)
        errors.append(error)

# print(np.mean(errors))




