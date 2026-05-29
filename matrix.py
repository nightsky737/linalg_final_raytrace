import numpy as np
import math

def make_translate( x, y, z ):
    t = np.zeros((4,4))
    ident(t)
    t[0][3] = x
    t[1][3] = y
    t[2][3] = z
    return np.array(t)

def make_scale( x, y, z ):
    t = np.zeros((4,4))
    ident(t)
    t[0][0] = x
    t[1][1] = y
    t[2][2] = z
    return np.array(t)

def make_rotX(theta):
    t = np.zeros((4,4))
    ident(t)
    t[1][1] =  math.cos(theta)
    t[1][2] = -math.sin(theta)  # row 1, col 2
    t[2][1] =  math.sin(theta)  # row 2, col 1
    t[2][2] =  math.cos(theta)
    return np.array(t)

def make_rotY(theta):
    t = np.zeros((4,4))
    ident(t)
    t[0][0] =  math.cos(theta)
    t[0][2] =  math.sin(theta)  # row 0, col 2
    t[2][0] = -math.sin(theta)  # row 2, col 0
    t[2][2] =  math.cos(theta)
    return np.array(t)

def make_rotZ(theta):
    t = np.zeros((4,4))
    ident(t)
    t[0][0] =  math.cos(theta)
    t[0][1] = -math.sin(theta)  # row 0, col 1
    t[1][0] =  math.sin(theta)  # row 1, col 0
    t[1][1] =  math.cos(theta)
    return np.array(t)


def ident( matrix ):
    for r in range( len( matrix[0] ) ):
        for c in range( len(matrix) ):
            if r == c:
                matrix[c][r] = 1


