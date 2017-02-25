import numpy as np
from scipy import io
from Quaternion import *
import matplotlib.pyplot as plt
import cv2, math

file_num = 9
mydata = 1 # flag decides whether to use vicon or my filtered data

if mydata:
    fileName = "filtered/filtered" + str(file_num) + ".npy"
    fileNamet = "filtered/time" + str(file_num) + ".npy"
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
dT = 10 # use 1 out of every 10 images
for t in range(0, T , dT):
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

    # Blend new frame and the rest of the pano
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

    # blend the two
    pano = cv2.addWeighted(empty_pano, 0.5, pano, 0.5, 0)

# Crop the final pano
# source: http://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy
ret, mask = cv2.threshold(cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
pano = pano[np.ix_(mask.any(1),mask.any(0))]

# ip = interp2d(x, y, z); zi = ip(xi, yi) #TODO
plt.imshow(pano)
plt.show()
