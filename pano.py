import numpy as np
from scipy import io
from Quaternion import *
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
import cv2
#from findDimensions import findDimensions
import math
file_num = 8
mydata = 0 # flag decides whether to use vicon or my filtered data

if mydata:
    fileName = "filtered" + str(file_num) + ".npy"
    fileNamet = "time" + str(file_num) + ".npy"
    q = np.load(fileName)
    q_t = np.load(fileNamet)
else:
    v = io.loadmat('vicon/viconRot' + str(file_num) + '.mat')
    q_t = np.reshape(v['ts'], (-1, 1))

# load cam data
camdata = io.loadmat('cam/cam' + str(file_num) + '.mat')
pics = camdata['cam']  # dimensions (240, 320, 3, 1685)
pics_t = camdata['ts']  # dimensions (1, 1685)]
im_size = tuple(np.shape(pics[:,:,:,0]))
h_pix = im_size[0]
w_pix = im_size[1]

# set up pano canvas
pano_size = (1000,1000,3)
pano = np.zeros(pano_size, dtype=np.uint8)


# pre-calculate arrays to be used for pixel rotations
h_fov = np.radians(45)
w_fov = np.radians(60)

y_u = np.linspace(0, h_pix-1, h_pix).astype(int)
x_v = np.linspace(0, w_pix-1, w_pix).astype(int)
yv, xv = np.meshgrid(y_u, x_v, sparse=False, indexing='ij')

lam = -(y_u - h_pix/2) / h_pix * h_fov
phi = -(x_v - w_pix/2) / w_pix * w_fov

x = np.outer(np.cos(lam), np.cos(phi))
y = np.outer(np.cos(lam), np.sin(phi))
z = np.tile(np.sin(lam).reshape(-1,1),(1,w_pix))
cart = np.dstack((x,y,z))

T = np.shape(pics_t)[1]
for t in range(0, T,20):
    i = np.argmin(q_t <= pics_t[0, t])
    if mydata:
        R = q[i].to_rotation()
    else:
        R = v['rots'][:, :, i]

    frame = pics[:,:,:,t]

    r_cart = np.einsum('pr,mnr->mnp', R, cart)

    X = r_cart[:, :, 0]
    Y = r_cart[:, :, 1]
    Z = r_cart[:, :, 2]

    pitch = np.arccos(Z / np.sqrt(np.power(X,2) + np.power(Y,2) + np.power(Z,2)))
    yaw = np.arctan2(X,Y) + np.pi
    h = (pitch / math.pi * pano_size[0]).astype(int)
    theta = (yaw / math.pi / 2 * pano_size[1]).astype(int)

    ################ blending
    new_frame = np.zeros(pano_size, dtype=np.uint8)
    new_frame[h.tolist(), theta.tolist(), :] = frame

    empty_pano = np.zeros(pano_size, dtype=np.uint8)

    # get the intersecting region
    ret1, mask1 = cv2.threshold(cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
    ret2, mask2 = cv2.threshold(cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)

    # empty_pano has old pano on top
    empty_pano = cv2.add(empty_pano,new_frame, mask=(cv2.bitwise_and(mask1,cv2.bitwise_not(mask2))))
    empty_pano = cv2.add(empty_pano,pano)

    # pano has the newest frame on top
    pano[h.tolist(), theta.tolist(), :] = frame

    pano = cv2.addWeighted(empty_pano, 0.5, pano, 0.5, 0)


# ip = interp2d(x, y, z); zi = ip(xi, yi)
#pano = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)
plt.imshow(pano)
plt.show()

################################################################################### OLD

# for i in range(0, h_pix):
#     for j in range(0, w_pix):
#         r_coords = np.dot(R, np.array([x[i,j], y[i,j], z[i,j]]))
#
#         X = r_coords[0]
#         Y = r_coords[1]
#         Z = r_coords[2]
#
#         lam2 = np.arccos(Z / np.sqrt(X**2 + Y**2 + Z**2))
#         phi2 = np.arctan(Y/X)
#
#         #theta = np.mod(int((np.arctan(Y / X) - np.pi) / (2 * np.pi) * res_theta + res_theta / 2), res_theta)
#         #h = np.mod(int(Z / np.sqrt(X ** 2 + Y ** 2) * res_h * 10), res_h)
#         theta = int(lam2 / math.pi / 2 * pano_size[1])
#         h = int((phi2+math.pi/2) / math.pi * pano_size[0])
#         pano[h, theta, :] = frame[i, j, :]



# for y_u in range(1, 240):
#     for x_v in range(1, 320):
#
#         lam = (y_u) / h_pix * h_fov
#         phi = (x_v) / w_pix * w_fov
#
#         # the order of x,y,z and phi, lam is incorrect
#         x = np.cos(lam)*np.cos(phi)
#         y = np.cos(lam)*np.sin(phi)
#         z = np.sin(lam)
#
#         r_coords = np.dot(R, np.array([x, y, z]))
#
#         X = r_coords[0]
#         Y = r_coords[1]
#         Z = r_coords[2]
#         # mistakes end here
#
#         lam2 = np.arccos(Z / np.sqrt(X**2 + Y**2 + Z**2))
#         phi2 = np.arctan(Y/X)
#
#         #theta = np.mod(int((np.arctan(Y / X) - np.pi) / (2 * np.pi) * res_theta + res_theta / 2), res_theta)
#         #h = np.mod(int(Z / np.sqrt(X ** 2 + Y ** 2) * res_h * 10), res_h)
#         theta = int(lam2 / math.pi / 2 * pano_size[1])
#         h = int(phi2 / math.pi * pano_size[0])
#         pano[h, theta, :] = frame[y_u, x_v, :]
# count += 1

        #next_img_warp = cv2.warpPerspective(pics[:, :, :, t], np.eye(3), (pano_size[1], pano_size[0]))

        # get the location of the new image

    # H = R_first * np.linalg.inv(R)
    # (min_x, min_y, max_x, max_y) = findDimensions(pics[:, :, :, t], H)
    # print((min_x, min_y, max_x, max_y))
    #
    # # new dimensions of pano
    # max_x = max(max_x, result.shape[1])
    # max_y = max(max_y, result.shape[0])
    #
    # move_h = np.matrix(np.identity(3), np.float32)
    #
    # if (min_x < 0):
    #     move_h[0, 2] += -min_x
    #     max_x += -min_x
    #
    # if (min_y < 0):
    #     move_h[1, 2] += -min_y
    #     max_y += -min_y
    #
    # img_w = int(math.ceil(max_x))
    # img_h = int(math.ceil(max_y))
    #
    # base_img_warp = cv2.warpPerspective(result, move_h, (img_w, img_h))
    # next_img_warp = cv2.warpPerspective(pics[:, :, :, t], move_h * H, (img_w, img_h))

    # plt.imshow(next_img_warp)
    # plt.show()
    #print(t)

    #result = cv2.add(temp, result)

    # (ret, data_map) = cv2.threshold(cv2.cvtColor(next_img_warp, cv2.COLOR_BGR2GRAY),0, 255, cv2.THRESH_BINARY)
    #
    # enlarged_base_img = np.zeros((img_h, img_w,3), np.uint8)
    # enlarged_base_img = cv2.add(enlarged_base_img, base_img_warp, mask=np.bitwise_not(data_map), dtype=cv2.CV_8U)
    # result = cv2.add(enlarged_base_img, next_img_warp, dtype=cv2.CV_8U)
    # plt.imshow(result)
    # plt.show()


    # # get first masked value (foreground)
    # fg = cv2.bitwise_or(next_img_warp, next_img_warp, mask=mask)
    #
    # # get second masked value (background) mask must be inverted
    # mask = cv2.bitwise_not(mask)
    # #background = np.full(result.shape, 0, dtype=np.uint8)
    # bk = cv2.bitwise_or(result, result, mask=mask)
    #
    # # combine foreground+background
    # result = cv2.bitwise_or(fg, bk)

    # plt.imshow(result)
    # plt.show()


# plt.imshow(result)
# plt.show()
#
# print(pano)
# plt.imshow(pano)
# plt.show()
#








