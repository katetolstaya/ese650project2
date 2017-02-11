from scipy import io
import numpy as np
from Quaternion import *

x = io.loadmat('imu/imuRaw1.mat')
v = io.loadmat('vicon/viconRot1.mat')

# ‘cams’, ‘vals’, ‘rots’, and ‘ts’
# shape of vals = (6, 5645)
# shape of ts = (1, 5645)

# vicon - rots shape (3, 3, 5561)


Vref = float(3300)  # mV

gyro_sens1 = 0.83 * 180 / np.pi  # mv / (deg/s) to radians
gyro_sens4 = 3.33 * 180 / np.pi  # mv / (deg/s) to radians
gyro_bias = 1023 * (1.23 / 3.3)  # Vref/2 /1023 # mV zero rate level
gyro_scale = Vref / 1023 / gyro_sens4

accel_sens = 330  # mV/g
accel_bias = 1023 / 2  # Vref / 2  #mV
accel_scale = Vref / 1023 / accel_sens

accel_raw = (x['vals']).astype('float')[0:3, :]  # x,y,z
gyro_raw = (x['vals']).astype('float')[3:6, :]  # z,x,y

accel_val = (accel_raw - accel_bias) * accel_scale
gyro_val = (gyro_raw - gyro_bias) * gyro_scale

# print(gyro_val)
# print(accel_val)


# gyro orientation
dt = np.mean(np.ediff1d(x['ts']))

# paper equations 9-11
d_angle = np.linalg.norm(gyro_val, axis=0) * dt
d_axis = gyro_val / np.tile(np.linalg.norm(gyro_val, axis=0), (3, 1))  # z,x,y
d_q = np.empty([4, np.shape(d_axis)[1]])

d_q[0, :] = np.cos(d_angle / 2)
d_q[1, :] = d_axis[1, :] * np.sin(d_angle / 2)  # x
d_q[2, :] = d_axis[2, :] * np.sin(d_angle / 2)  # y
d_q[3, :] = d_axis[0, :] * np.sin(d_angle / 2)  # z

# q1 = Quaternion.from_vector([1, 1, 2, 3])
# q2 = Quaternion.from_vector([4, 4, 5, 6])
# q3 = Quaternion.from_vector([1, 1, 1, 1])


q1 = Quaternion.from_euler(np.pi * np.array([1, 0, 0]))
q2 = Quaternion.from_euler(np.pi * np.array([1, 0, 0]))



#print((q1*q2).to_angles())
#print(np.array([1,0,0]) * np.array([1,0,0]))
arrq = [q1, q2]
arrw = [0.5, 0.5]
avg = Quaternion.average(arrq,arrw )
print(avg*avg)
print(q2*q1)


# print(d_axis)


# orientation = np.cumsum(gyro_val*ts, axis=1) # integrating gyro values # z,x,y

# accel orientation
