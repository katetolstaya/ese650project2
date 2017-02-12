import numpy as np
from scipy import io
from Quaternion import *
import matplotlib.pyplot as plt

file_num = 1
fileName = "filtered" + str(file_num) + ".npy"
fileNamet = "time" + str(file_num) + ".npy"
camdata = io.loadmat('cam/cam' + str(file_num) + '.mat')
q = np.load(fileName)
q_t = np.load(fileNamet)

pics = camdata['cam']  # dimensions (240, 320, 3, 1685)
pics_t = camdata['ts']  # dimensions (1, 1685)

res_theta = 1000
res_h = 300

pano = np.zeros((res_h,res_theta, 3))

#x = np.linspace(0, 1 , 240)
#y = np.linspace(0, 1,320)
#xv, yv = np.meshgrid(y, x)
#xv = np.reshape(xv,(np.shape(xv)[0], np.shape(xv)[1],1))
#yv = np.reshape(yv,(np.shape(yv)[0], np.shape(yv)[1],1))

f = 1 # focal point??
#zv = np.tile(f, (np.shape(xv)[0], np.shape(xv)[1],1))
#coords = np.concatenate((xv,yv,zv), axis=2)


T = 5 #np.shape(pics_t)[1]
count =0
for t in range(0, T):
    i = np.argmin(q_t <= pics_t[0, t])
    R = q[i].to_rotation()
    print (t)
    for x in range(1,240):
        for y in range(1,320):
            r_coords = np.dot(R,np.array([x,y,f]))
            #print(r_coords)
            X = r_coords[0]
            Y = r_coords[1]
            Z = r_coords[2]

            theta = np.mod(int((np.arctan(Y/X) - np.pi) / (2*np.pi) * res_theta + res_theta/2),res_theta)
            h = np.mod(int(Z/np.sqrt(X**2 + Y**2) * res_h*10),res_h)
            pano[h,theta,:] = pics[x,y,:,t]
            count+=1

           # print(h)
           # print(theta)


print(count)
print(pano)
plt.imshow(pano)
plt.show()









