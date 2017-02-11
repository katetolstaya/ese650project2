from scipy import io
import numpy as np
from Quaternion import *
from rotplot import rotplot
import matplotlib.pyplot as plt

from matplotlib.pyplot import *
import math

# https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

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

accel_sens = 330  # 304  # mV/g
accel_bias = 1023 / 2  # Vref / 2  #mV
accel_scale = Vref / 1023 / accel_sens

accel_raw = (x['vals']).astype('float')[0:3, :]  # x,y,z
gyro_raw = (x['vals']).astype('float')[3:6, :]  # z,x,y

accel_val = (accel_raw - accel_bias) * accel_scale * np.tile(np.reshape(np.array([-1,-1,1]),(-1,1)),(1,np.shape(accel_raw)[1]))

gyro_val = (gyro_raw - gyro_bias) * gyro_scale
temp = np.copy(gyro_val[0, :])
gyro_val[0, :] = gyro_val[1, :]
gyro_val[1, :] = gyro_val[2, :]
gyro_val[2, :] = temp

# subtract the gyro drift (avg of gyro)
gyro_drift = np.reshape(np.mean(gyro_val,axis=1),(-1,1))
gyro_val = np.add(gyro_val, -1 * np.tile(gyro_drift,(1,np.shape(gyro_val)[1])))

#########################
dt = np.mean(np.ediff1d(x['ts']))

# GYRO - equations 9-11
d_angle = np.linalg.norm(gyro_val, axis=0) * dt
d_axis = gyro_val / np.tile(d_angle, (3, 1))
gyro_orientation = np.cumsum(gyro_val * dt, axis=1)

# ACCEL
accel_orientation = np.zeros(np.shape(accel_val)) # yaw = 0
accel_orientation[0, :] = np.arctan2(accel_val[1, :], accel_val[2, :])  # roll
accel_orientation[1, :] = np.arctan2(-accel_val[0, :], np.sqrt(
    accel_val[1, :] * accel_val[1, :] + accel_val[2, :] * accel_val[2, :]))  # pitch

##############################################################################
############# benchmarking vs vicon ##########################################
vicon_orientation = np.zeros((np.shape(v['rots'])[0], np.shape(v['rots'])[2]))
for i in range(0, np.shape(v['rots'])[2]):
    vicon_orientation[:, i] = rotationMatrixToEulerAngles(v['rots'][:, :, i])

t = np.reshape(x['ts'], (-1, 1))
tv = np.reshape(v['ts'], (-1, 1))

# plt.figure(1)
# plt.subplot(311)
# plt.plot(t, accel_orientation[0, :], 'r', t, gyro_orientation[0, :], 'b', tv, vicon_orientation[0, :], 'g')
# plt.subplot(312)
# plt.plot(t, accel_orientation[1, :], 'r', t, gyro_orientation[1, :], 'b', tv, vicon_orientation[1, :], 'g')
# plt.subplot(313)
# plt.plot(t, accel_orientation[2, :], 'r', t, gyro_orientation[2, :], 'b', tv,vicon_orientation[2, :], 'g')
# plt.show()

##############################################################################
############# FILTER and real time plots #####################################

# process noise parameters
q_w = Quaternion.from_euler(dt * np.array([0.00001, 0.00001, 0.00001]))  # orientation noise
w_w = dt * np.array([0, 0, 0])  # np.array([0.001,0.001,0.001]) # rotation noise

# initial conditions
w_k = np.array([0, 0, 0])
q_k = Quaternion.from_euler(accel_orientation[:, 0])

s = np.eye(3)  # cov

fig = plt.figure()
ax = fig.gca(projection='3d')
plt.ion()
plt.show()
for t in range(0, np.shape(accel_val)[1]):
    # Process model
    q_d = Quaternion.from_euler(gyro_val[:, t] * dt)
    w_k_1 = w_k + w_w
    q_k_1 = q_k * q_w * q_d

    # measurement model
    q_a = Quaternion.from_euler(accel_orientation[:, t])  # yaw - N/A
    w_a = np.array([0, 0, 0])  # N/A
    q_k = q_k_1


    R_a = q_a.to_rotation()
    rotplot(R_a, 'Blue','Red', ax)
    RTurn = q_k.to_rotation()
    rotplot(RTurn,  'Green', 'Yellow', ax)

    plt.draw()
    plt.pause(0.000001)
    plt.cla()

show()

