import numpy as np


def transform_from_pose(pose):
    x, y, theta = pose
    c = np.cos(theta)
    s = np.sin(theta)

    R = np.array([[c, s], [-s, c]])
    t = -R @ np.array([x, y])

    trans = np.eye(3)
    trans[:2, :2] = R
    trans[:2, 2] = t

    return trans


def inverse_transform_from_pose(pose):
    x, y, theta = pose

    c = np.cos(-theta)
    s = np.sin(-theta)

    R = np.array([[c, s], [-s, c]])

    trans = np.eye(3)
    trans[:2, :2] = R
    trans[:2, 2] = (x, y)

    return trans
