import numpy as np


class Quaternion:
    def __init__(self, s, v):
        self.v = v.astype('float')
        self.s = float(s)

    @classmethod
    def from_vector(cls, v):
        return Quaternion(v[0], v[1:])

    def to_vector(self):
        return np.array([self.s, self.v[0], self.v[1], self.v[2]])

    @classmethod
    def from_euler(cls, v):
        roll = v[0]
        pitch = v[1]
        yaw = v[2]

        # roll, pitch, yaw
        t0 = np.cos(yaw * 0.5);
        t1 = np.sin(yaw * 0.5);
        t2 = np.cos(roll * 0.5);
        t3 = np.sin(roll * 0.5);
        t4 = np.cos(pitch * 0.5);
        t5 = np.sin(pitch * 0.5);

        w = t0 * t2 * t4 + t1 * t3 * t5;
        x = t0 * t3 * t4 - t1 * t2 * t5;
        y = t0 * t2 * t5 + t1 * t3 * t4;
        z = t1 * t2 * t4 - t0 * t3 * t5;
        return Quaternion(w, np.array([x, y, z]))

    @classmethod
    def from_angle(cls, v):
        if np.linalg.norm(v) == 0:
            return Quaternion(0, np.array([0, 0, 0]))
        d_angle = np.linalg.norm(v)
        d_axis = v / np.linalg.norm(v)
        return Quaternion(np.cos(d_angle / 2), d_axis * np.sin(d_angle / 2))

    @classmethod
    def from_rotation(cls, v):
        # from http://www.cg.info.hiroshima-cu.ac.jp/~miyazaki/knowledge/teche52.html
        r11 = v[0, 0]
        r12 = v[0, 1]
        r13 = v[0, 2]
        r21 = v[1, 0]
        r22 = v[1, 1]
        r23 = v[1, 2]
        r31 = v[2, 0]
        r32 = v[2, 1]
        r33 = v[2, 2]

        q0 = (r11 + r22 + r33 + 1.0) / 4.0
        q1 = (r11 - r22 - r33 + 1.0) / 4.0
        q2 = (-r11 + r22 - r33 + 1.0) / 4.0
        q3 = (-r11 - r22 + r33 + 1.0) / 4.0
        if (q0 < 0.0):
            q0 = 0.0
        if (q1 < 0.0):
            q1 = 0.0
        if (q2 < 0.0):
            q2 = 0.0
        if (q3 < 0.0):
            q3 = 0.0
        q0 = np.sqrt(q0)
        q1 = np.sqrt(q1)
        q2 = np.sqrt(q2)
        q3 = np.sqrt(q3)
        if q0 >= q1 and q0 >= q2 and q0 >= q3:
            q0 *= +1.0
            q1 *= np.sign(r32 - r23)
            q2 *= np.sign(r13 - r31)
            q3 *= np.sign(r21 - r12)
        elif q1 >= q0 and q1 >= q2 and q1 >= q3:
            q0 *= np.sign(r32 - r23)
            q1 *= +1.0
            q2 *= np.sign(r21 + r12)
            q3 *= np.sign(r13 + r31)
        elif q2 >= q0 and q2 >= q1 and q2 >= q3:
            q0 *= np.sign(r13 - r31)
            q1 *= np.sign(r21 + r12)
            q2 *= +1.0
            q3 *= np.sign(r32 + r23)
        else:  # if (q3 >= q0 & & q3 >= q1 & & q3 >= q2)
            q0 *= np.sign(r21 - r12)
            q1 *= np.sign(r31 + r13)
            q2 *= np.sign(r32 + r23)
            q3 *= +1.0

        r = np.linalg.norm([q0, q1, q2, q3])
        q0 /= r
        q1 /= r
        q2 /= r
        q3 /= r
        return Quaternion(q0, [q1, q2, q3])

    def __add__(self, other):
        #if (type(other) == Quaternion):
        return Quaternion(self.s + other.s, self.v + other.v)
        # else:
        #     return Quaternion(other[0] + self.s, other[1:] + self.v)

    def __sub__(self, other):
        # if (type(other) == Quaternion):
        return Quaternion(self.s - other.s, self.v - other.v)
        # else:
        #     return Quaternion(other[0] + self.s, other[1:] + self.v)

    def __mul__(self, other):
        if (type(other) == Quaternion):
            return Quaternion(self.s * other.s - np.dot(self.v, other.v),
                              self.s * other.v + self.v * other.s + np.cross(self.v, other.v))
        else:
            return Quaternion(other * self.s, other * self.v)

    def __rmul__(self, other):
        return Quaternion(other * self.s, other * self.v)

    def __div__(self, other):
        return Quaternion(self.scalar / other, self.vector / other)

    def inverse(self):
        c = self.conjugate()
        p = self.norm() ** 2
        if (np.linalg.norm(self.v) == 0):
            return self
        return Quaternion(c.s / p, c.v / p)

    def conjugate(self):
        return Quaternion(self.s, -self.v)

    def __str__(self):
        return "[" + str(self.s) + ",[" + str(self.v[0]) + "," + str(self.v[1]) + "," + str(self.v[2]) + "]]"

    def norm(self):
        return np.linalg.norm([self.s, self.v[0], self.v[1], self.v[2]])

    def normalize(self):
        n = self.norm()
        return Quaternion(self.s/n, self.v/n)

    def log(self):
        n_q = self.norm()
        n_v = np.linalg.norm(self.v)
        return Quaternion(np.log(n_q), (self.v / n_v * np.arccos(self.s / n_q)))

    def exp(self):
        n = np.linalg.norm(self.v)
        c = np.cos(n)
        s = np.sin(n)
        e = np.exp(self.s)
        return e * Quaternion(c, s / n * self.v)

    def to_rotation(self):
        # from www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/
        x = self.v[0]
        y = self.v[1]
        z = self.v[2]
        w = self.s

        m = np.empty((3, 3))
        m[0, 0] = 1 - 2 * (y * y + z * z)
        m[0, 1] = 2 * (x * y - z * w)
        m[0, 2] = 2 * (x * z + y * w)

        m[1, 0] = 2 * (x * y + z * w)
        m[1, 1] = 1 - 2 * (x * x + z * z)
        m[1, 2] = 2 * (y * z - x * w)

        m[2, 0] = 2 * (x * z - y * w)
        m[2, 1] = 2 * (y * z + x * w)
        m[2, 2] = 1 - 2 * (x * x + y * y)

        return m

    def to_euler(self):

        x = self.v[0]
        y = self.v[1]
        z = self.v[2]
        w = self.s
        ysqr = y * y

        # roll = x axis rotation
        t0 = 2 * (w * x + y * z)
        t1 = 1 - 2 * (x * x + ysqr)
        roll = np.arctan2(t0, t1)

        # pitch = y axis rotation
        t2 = 2 * (w * y - z * x)
        if t2 > 1:
            t2 = 1
        elif t2 < -1:
            t2 = -1

        pitch = np.arcsin(t2)

        # yaw = z axis rotation
        t3 = 2 * (w * z + x * y)
        t4 = 1 - 2 * (ysqr + z * z)
        yaw = np.arctan2(t3, t4)
        return np.array([roll, pitch, yaw])

    def to_angle(self):
        if np.isclose(np.linalg.norm(self.v), 0,1e-8):
            return np.array([0, 0, 0])
        else:
            theta = 2 * np.arccos(self.s)
            u = self.v / np.sin(theta / 2)
        return theta * u

    @staticmethod
    def average(q_arr, w_arr):
        max_iter = 100
        eps = 0.01
        n = len(q_arr)

        q = q_arr[0]

        for t in range(0, max_iter):
            e_v = np.zeros(3)

            for i in range(0, n):
                if np.isclose(np.linalg.norm((q_arr[i] * q.inverse()).v), 0, atol=1e-15):
                    e_vi = np.zeros(3)
                else:
                    e_vi = 2 * ((q_arr[i] * q.inverse()).log()).v
                    # print(c)
                    e_vi = (-np.pi + np.mod(np.linalg.norm(e_vi) + np.pi, 2 * np.pi)) / np.linalg.norm(e_vi) * e_vi
                    # print("e_vi=" + str(e_vi))

                e_v = np.add(e_v, w_arr[i] * e_vi)
                # print("e_v="+str(e_v))

            if np.linalg.norm(e_v) < eps:
                return q
            else:
                q = Quaternion(0, e_v / 2).exp() * q

        return q
