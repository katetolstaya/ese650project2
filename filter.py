from scipy import io
import math, os, scipy
from Quaternion import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from util import rotationMatrixToEulerAngles, isRotationMatrix

######## Change file num and file name params ###########
file_num = 9
x = io.loadmat('imu/imuRaw' + str(file_num) + '.mat')
v = io.loadmat('vicon/viconRot' + str(file_num) + '.mat')
fileName = "filtered/filtered"+str(file_num)
fileNamet = "filtered/time"+str(file_num)
########################################################

####### Data cleaning ##################################
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

accel_val = (accel_raw - accel_bias) * accel_scale * np.tile(np.reshape(np.array([-1, -1, 1]), (-1, 1)),
                                                             (1, np.shape(accel_raw)[1]))

gyro_val = (gyro_raw - gyro_bias) * gyro_scale
temp = np.copy(gyro_val[0, :])

# swap gyro axes according to spec
gyro_val[0, :] = gyro_val[1, :]
gyro_val[1, :] = gyro_val[2, :]
gyro_val[2, :] = temp

# subtract the gyro drift (avg of gyro)
gyro_drift = np.reshape(np.mean(gyro_val[:,1:100], axis=1), (-1, 1))
gyro_val = np.add(gyro_val, -1 * np.tile(gyro_drift, (1, np.shape(gyro_val)[1])))

##############################################################################
############# benchmarking vs vicon and basic averaging ######################
dt = np.mean(np.ediff1d(x['ts']))

# GYRO - equations 9-11
d_angle = np.linalg.norm(gyro_val, axis=0) * dt
d_axis = gyro_val / np.tile(d_angle, (3, 1))
gyro_orientation = np.cumsum(gyro_val * dt, axis=1)

# ACCEL
accel_orientation = np.zeros(np.shape(accel_val))  # yaw = 0
accel_orientation[0, :] = np.arctan2(accel_val[1, :], accel_val[2, :])  # roll
accel_orientation[1, :] = np.arctan2(-accel_val[0, :], np.sqrt(
    accel_val[1, :] * accel_val[1, :] + accel_val[2, :] * accel_val[2, :]))  # pitch

vicon_orientation = np.zeros((np.shape(v['rots'])[0], np.shape(v['rots'])[2]))
for i in range(0, np.shape(v['rots'])[2]):
    vicon_orientation[:, i] = rotationMatrixToEulerAngles(v['rots'][:, :, i])

t_plot = np.reshape(x['ts'], (-1, 1))
tv_plot = np.reshape(v['ts'], (-1, 1))

# plt.figure(1)
# plt.subplot(311)
# plt.plot(t_plot, accel_orientation[0, :], 'r', t_plot, gyro_orientation[0, :], 'b', tv_plot, vicon_orientation[0, :], 'g')
# plt.subplot(312)
# plt.plot(t_plot, accel_orientation[1, :], 'r', t_plot, gyro_orientation[1, :], 'b', tv_plot, vicon_orientation[1, :], 'g')
# plt.subplot(313)
# plt.plot(t_plot, accel_orientation[2, :], 'r', t_plot, gyro_orientation[2, :], 'b', tv_plot,vicon_orientation[2, :], 'g')
# plt.show()

##############################################################################
############# FILTER and real time plots #####################################
n = 3
g = Quaternion(0, np.array([0, 0, 1]))

# initial conditions
q_k = Quaternion.from_euler(accel_orientation[:, 0])  # mean
# TODO tune
P_k = 1e-6 * np.eye(3)  # cov
# noise parameters
Q = 1e-8 * np.eye(3)  # cov
R = 5e-2 * np.eye(3)  # cov

# animated plot
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# plt.ion()
# plt.show()

q_accum = np.empty(np.shape(accel_val)[1], dtype=Quaternion)
predict_orientation = np.zeros(np.shape(accel_val))  # yaw = 0

for t in range(0, np.shape(accel_val)[1]):

    predict_orientation[:, t] = q_k.to_euler()
    q_accum[t] = q_k

    # Sigma Points #####################################################################################################
    tempW =  np.sqrt(2*n) * scipy.linalg.cholesky(P_k + Q)
    W = np.append(tempW, -tempW, axis=1)

    X = np.empty(2 * n, dtype=Quaternion)
    for i in range(0, 2 * n):
        X[i] = q_k * Quaternion.from_angle(W[:, i])

    # Process model ####################################################################################################
    q_d = np.tile(Quaternion.from_euler(gyro_val[:, t] * dt), (2 * n,))
    Y = X * q_d

    Yavg = Quaternion.average(Y, 1 / (2 * n) * (np.ones((2 * n))))

    P_k_ = np.zeros(P_k.shape)
    for i in range(0, 2 * n):
        w = (Y[i] * Yavg.inverse()).to_angle()
        P_k_ += np.outer(w, w)
    P_k_ /= (4 * n)

    # Measurement model ################################################################################################
    a_val = -accel_val[:, i]

    Z = np.empty((3, 2 * n))
    for i in range(0, 2 * n):
        Z[:, i] = (Y[i] * g * Y[i].inverse()).to_angle()

    z_k = np.mean(Z, axis=1)

    P_zz = np.zeros(P_k.shape)  # Measure estimate covariance
    P_xz = np.zeros(P_k.shape)  # Cross correlation matrix
    for i in range(0, 2 * n):
        w = (Y[i] * Yavg.inverse()).to_angle()
        temp = Z[:, i] - z_k
        P_zz += np.outer(temp, temp)
        P_xz += np.outer(w, temp)
    P_zz /= (4 * n)
    P_xz /= (4 * n)

    P_vv = P_zz + R  # covariance of the innovation
    K = np.dot(P_xz, np.linalg.inv(P_vv))  # Kalman gain
    v = z_k - a_val

    # Update ###########################################################################################################
    q_k = Yavg * Quaternion.from_angle(np.dot(K, v))
    P_k = P_k_ - np.dot(np.dot(K, P_vv), np.matrix.transpose(K))

# # more animated plot code
#     R_a = q_a.to_rotation()
#     rotplot(R_a, 'Blue', 'Red', ax)
#     RTurn = q_k.to_rotation()
#     rotplot(RTurn, 'Green', 'Yellow', ax)
#
#     plt.draw()
#     plt.pause(0.001)
#     plt.cla()
# show()

np.save(fileName, q_accum)  # X is an array
np.save(fileNamet, t_plot)

plt.figure(1)
plt.subplot(311)
plt.plot(t_plot, predict_orientation[0, :], 'b', tv_plot, vicon_orientation[0, :], 'g')
plt.subplot(312)
plt.plot(t_plot, predict_orientation[1, :], 'b', tv_plot, vicon_orientation[1, :], 'g')
plt.subplot(313)
plt.plot(t_plot, predict_orientation[2, :], 'b', tv_plot, vicon_orientation[2, :], 'g')
plt.show()
