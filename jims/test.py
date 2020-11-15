import numpy as np
from numpy import cos, sin, arctan2, arcsin, pi, rad2deg

# def quat_to_Rot():
#     q0, q1, q2, q3 = [0.0, 0.25881904510252074, 0.0, 0.9659258262890683]
#     roll = arctan2(2*(q0*q1 + q2*q3), 1-2*(q1*q1 + q2*q2))
#     pitch = arcsin(2*(q0*q2 - q3*q1))
#     yaw = arctan2(2*(q0*q3 + q1*q2), 1-2*(q2*q2 + q3*q3))
#     return roll, pitch, yaw

# # print("Rad:", quat_to_Rot())
# # print("Deg:", np.rad2deg(quat_to_Rot()))

# def rot_to_quat():
#     roll, pitch, yaw = [0, pi*30/180, 0]
#     q0 = cos((roll-yaw)/2) * sin(pitch/2)
#     q1 = sin((roll-yaw)/2) * sin(pitch/2)
#     q2 = sin((roll+yaw)/2) * cos(pitch/2)
#     q3 = cos((roll+yaw)/2) * cos(pitch/2)
#     return q0, q1, q2, q3


# def rot_to_Quat():
#     r, p, y = [0, pi*30/180, 0]
#     q0 = (sin(r/2)*cos(p/2)*cos(y/2)) - (cos(r/2)*sin(p/2)*sin(y/2))
#     q1 = (cos(r/2)*sin(p/2)*cos(y/2)) + (sin(r/2)*cos(p/2)*sin(y/2))
#     q2 = (cos(r/2)*cos(p/2)*sin(y/2)) - (sin(r/2)*sin(p/2)*cos(y/2))
#     q3 = (cos(r/2)*cos(p/2)*cos(y/2)) + (sin(r/2)*sin(p/2)*sin(y/2))
#     return q0, q1, q2, q3


# print(quat_to_Rot())
# print(rot_to_Quat())
# def quaternion_to_euler(x, y, z, w):

#         import math
#         t0 = +2.0 * (w * x + y * z)
#         t1 = +1.0 - 2.0 * (x * x + y * y)
#         X = math.degrees(math.atan2(t0, t1))

#         t2 = +2.0 * (w * y - z * x)
#         t2 = +1.0 if t2 > +1.0 else t2
#         t2 = -1.0 if t2 < -1.0 else t2
#         Y = math.degrees(math.asin(t2))

#         t3 = +2.0 * (w * z + x * y)
#         t4 = +1.0 - 2.0 * (y * y + z * z)
#         Z = math.degrees(math.atan2(t3, t4))

#         return X, Y, Z

# x, y, z, w = rot_to_Quat()
# print("Quat back to Euler:", quaternion_to_euler(x, y, z, w))

# x = 90/180
# print(x)

# def get_body_Rot():
#     x, y, z, w = [0.0, 0.25881904510252074, 0.0, 0.9659258262890683]
#     roll = rad2deg(arctan2(2.0 * (w*x + y*z), 1 -2*(x*x + y*y)))

#     t2 = +2.0 * (w * y - z * x)
#     t2 = +1.0 if t2 > +1.0 else t2
#     t2 = -1.0 if t2 < -1.0 else t2
#     pitch = math.degrees(math.asin(t2))

#     t3 = +2.0 * (w * z + x * y)
#     t4 = +1.0 - 2.0 * (y * y + z * z)
#     yaw = math.degrees(math.atan2(t3, t4))

#     return roll, pitch, yaw

# print(np.linalg.norm([], ord=3))
print(1e-5)
print(0.00001)

