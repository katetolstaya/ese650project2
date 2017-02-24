import numpy as np
from scipy import io
from Quaternion import *
import matplotlib.pyplot as plt
import cv2
from findDimensions import findDimensions
import math
file_num = 1
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
pano_size = (500,1500,3)
#pano_size = im_size
result = np.zeros(pano_size, np.uint8)

# pano details
res_theta = 1000
res_h = 300
pano = np.zeros((res_h, res_theta, 3))


#x = np.linspace(0, 1 , 240)
#y = np.linspace(0, 1,320)
#xv, yv = np.meshgrid(y, x)
#xv = np.reshape(xv,(np.shape(xv)[0], np.shape(xv)[1],1))
#yv = np.reshape(yv,(np.shape(yv)[0], np.shape(yv)[1],1))

f = 1 # focal point??
#zv = np.tile(f, (np.shape(xv)[0], np.shape(xv)[1],1))
#coords = np.concatenate((xv,yv,zv), axis=2)


T = 500#np.shape(pics_t)[1]
#count = 0
for t in range(0, T,20):
    i = np.argmin(q_t <= pics_t[0, t])
    if mydata:
        R = q[i].to_rotation()
    else:
        R = v['rots'][:, :, i]

    if t == 0:
        R_first = R #np.linalg.inv(R)
        R = np.eye(3)
        #next_img_warp = cv2.warpPerspective(pics[:, :, :, t], np.eye(3), (pano_size[1], pano_size[0]))

        # get the location of the new image

    H = R_first * np.linalg.inv(R)
    (min_x, min_y, max_x, max_y) = findDimensions(pics[:, :, :, t], H)
    print((min_x, min_y, max_x, max_y))

    # new dimensions of pano
    max_x = max(max_x, result.shape[1])
    max_y = max(max_y, result.shape[0])

    move_h = np.matrix(np.identity(3), np.float32)

    if (min_x < 0):
        move_h[0, 2] += -min_x
        max_x += -min_x

    if (min_y < 0):
        move_h[1, 2] += -min_y
        max_y += -min_y

    img_w = int(math.ceil(max_x))
    img_h = int(math.ceil(max_y))

    base_img_warp = cv2.warpPerspective(result, move_h, (img_w, img_h))
    next_img_warp = cv2.warpPerspective(pics[:, :, :, t], move_h * H, (img_w, img_h))

    # plt.imshow(next_img_warp)
    # plt.show()
    #print(t)

    #result = cv2.add(temp, result)

    (ret, data_map) = cv2.threshold(cv2.cvtColor(next_img_warp, cv2.COLOR_BGR2GRAY),0, 255, cv2.THRESH_BINARY)

    enlarged_base_img = np.zeros((img_h, img_w,3), np.uint8)
    enlarged_base_img = cv2.add(enlarged_base_img, base_img_warp, mask=np.bitwise_not(data_map), dtype=cv2.CV_8U)
    result = cv2.add(enlarged_base_img, next_img_warp, dtype=cv2.CV_8U)
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
    # for x in range(1,240):
    #     for y in range(1,320):
    #         r_coords = np.dot(R,np.array([x,y,f]))
    #         #print(r_coords)
    #         X = r_coords[0]
    #         Y = r_coords[1]
    #         Z = r_coords[2]
    #
    #         theta = np.mod(int((np.arctan(Y/X) - np.pi) / (2*np.pi) * res_theta + res_theta/2),res_theta)
    #         h = np.mod(int(Z/np.sqrt(X**2 + Y**2) * res_h*10),res_h)
    #         pano[h,theta,:] = pics[x,y,:,t]
    #         count+=1
    #
    #        # print(h)
    #        # print(theta)

plt.imshow(result)
plt.show()
#
# print(pano)
# plt.imshow(pano)
# plt.show()
#








