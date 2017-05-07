#!/usr/bin/env python2

import sys
import copy
import random
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

epsilon = 0.0000000001
tries = 10
N = 10


determinant = parse_expr("-a1*beta1**2*e1*l**4 - a1*beta1**2*f2*l**3 - 2*a1*beta1*beta2*e1*l**3 - 2*a1*beta1*beta2*f2*l**2 + a1*beta1*d1*gamma1*l**4 + a1*beta1*d1*gamma2*l**3 + a1*beta1*e1*gamma1*l**4 + a1*beta1*e1*gamma2*l**3 + 2*a1*beta1*e2*gamma1*l**3 + 2*a1*beta1*e2*gamma2*l**2 - a1*beta2**2*e1*l**2 - a1*beta2**2*f2*l + a1*beta2*d1*gamma1*l**3 + a1*beta2*d1*gamma2*l**2 + a1*beta2*e1*gamma1*l**3 + a1*beta2*e1*gamma2*l**2 + 2*a1*beta2*e2*gamma1*l**2 + 2*a1*beta2*e2*gamma2*l - a1*d1*e2*fi1*l**3 - a1*d1*e2*fi2*l**2 + a1*d1*f2*fi1*l**3 + a1*d1*f2*fi2*l**2 - a1*d1*gamma1**2*l**4 - 2*a1*d1*gamma1*gamma2*l**3 - a1*d1*gamma2**2*l**2 + a1*d2*e1*fi1*l**3 + a1*d2*e1*fi2*l**2 + a1*d2*f2*fi1*l**2 + a1*d2*f2*fi2*l - a1*d2*gamma1**2*l**3 - 2*a1*d2*gamma1*gamma2*l**2 - a1*d2*gamma2**2*l - a1*e1*e2*fi1*l**3 - a1*e1*e2*fi2*l**2 - a1*e2**2*fi1*l**2 - a1*e2**2*fi2*l - a2*beta1**2*e1*l**3 - a2*beta1**2*f2*l**2 - 2*a2*beta1*beta2*e1*l**2 - 2*a2*beta1*beta2*f2*l + a2*beta1*d1*gamma1*l**3 + a2*beta1*d1*gamma2*l**2 + a2*beta1*e1*gamma1*l**3 + a2*beta1*e1*gamma2*l**2 + 2*a2*beta1*e2*gamma1*l**2 + 2*a2*beta1*e2*gamma2*l - a2*beta2**2*e1*l - a2*beta2**2*f2 + a2*beta2*d1*gamma1*l**2 + a2*beta2*d1*gamma2*l + a2*beta2*e1*gamma1*l**2 + a2*beta2*e1*gamma2*l + 2*a2*beta2*e2*gamma1*l + 2*a2*beta2*e2*gamma2 - a2*d1*e2*fi1*l**2 - a2*d1*e2*fi2*l + a2*d1*f2*fi1*l**2 + a2*d1*f2*fi2*l - a2*d1*gamma1**2*l**3 - 2*a2*d1*gamma1*gamma2*l**2 - a2*d1*gamma2**2*l + a2*d2*e1*fi1*l**2 + a2*d2*e1*fi2*l + a2*d2*f2*fi1*l + a2*d2*f2*fi2 - a2*d2*gamma1**2*l**2 - 2*a2*d2*gamma1*gamma2*l - a2*d2*gamma2**2 - a2*e1*e2*fi1*l**2 - a2*e1*e2*fi2*l - a2*e2**2*fi1*l - a2*e2**2*fi2 + alpha1**2*d1*e2*l**3 - alpha1**2*d1*f2*l**3 - alpha1**2*d2*e1*l**3 - alpha1**2*d2*f2*l**2 + alpha1**2*e1*e2*l**3 + alpha1**2*e2**2*l**2 + 2*alpha1*alpha2*d1*e2*l**2 - 2*alpha1*alpha2*d1*f2*l**2 - 2*alpha1*alpha2*d2*e1*l**2 - 2*alpha1*alpha2*d2*f2*l + 2*alpha1*alpha2*e1*e2*l**2 + 2*alpha1*alpha2*e2**2*l + 2*alpha1*b1*beta1*e1*l**4 + 2*alpha1*b1*beta1*f2*l**3 + 2*alpha1*b1*beta2*e1*l**3 + 2*alpha1*b1*beta2*f2*l**2 - alpha1*b1*d1*gamma1*l**4 - alpha1*b1*d1*gamma2*l**3 - alpha1*b1*e1*gamma1*l**4 - alpha1*b1*e1*gamma2*l**3 - 2*alpha1*b1*e2*gamma1*l**3 - 2*alpha1*b1*e2*gamma2*l**2 + 2*alpha1*b2*beta1*e1*l**3 + 2*alpha1*b2*beta1*f2*l**2 + 2*alpha1*b2*beta2*e1*l**2 + 2*alpha1*b2*beta2*f2*l - alpha1*b2*d1*gamma1*l**3 - alpha1*b2*d1*gamma2*l**2 - alpha1*b2*e1*gamma1*l**3 - alpha1*b2*e1*gamma2*l**2 - 2*alpha1*b2*e2*gamma1*l**2 - 2*alpha1*b2*e2*gamma2*l - alpha1*beta1*c1*d1*l**4 - alpha1*beta1*c1*e1*l**4 - 2*alpha1*beta1*c1*e2*l**3 - alpha1*beta1*c2*d1*l**3 - alpha1*beta1*c2*e1*l**3 - 2*alpha1*beta1*c2*e2*l**2 - alpha1*beta2*c1*d1*l**3 - alpha1*beta2*c1*e1*l**3 - 2*alpha1*beta2*c1*e2*l**2 - alpha1*beta2*c2*d1*l**2 - alpha1*beta2*c2*e1*l**2 - 2*alpha1*beta2*c2*e2*l + 2*alpha1*c1*d1*gamma1*l**4 + 2*alpha1*c1*d1*gamma2*l**3 + 2*alpha1*c1*d2*gamma1*l**3 + 2*alpha1*c1*d2*gamma2*l**2 + 2*alpha1*c2*d1*gamma1*l**3 + 2*alpha1*c2*d1*gamma2*l**2 + 2*alpha1*c2*d2*gamma1*l**2 + 2*alpha1*c2*d2*gamma2*l + alpha2**2*d1*e2*l - alpha2**2*d1*f2*l - alpha2**2*d2*e1*l - alpha2**2*d2*f2 + alpha2**2*e1*e2*l + alpha2**2*e2**2 + 2*alpha2*b1*beta1*e1*l**3 + 2*alpha2*b1*beta1*f2*l**2 + 2*alpha2*b1*beta2*e1*l**2 + 2*alpha2*b1*beta2*f2*l - alpha2*b1*d1*gamma1*l**3 - alpha2*b1*d1*gamma2*l**2 - alpha2*b1*e1*gamma1*l**3 - alpha2*b1*e1*gamma2*l**2 - 2*alpha2*b1*e2*gamma1*l**2 - 2*alpha2*b1*e2*gamma2*l + 2*alpha2*b2*beta1*e1*l**2 + 2*alpha2*b2*beta1*f2*l + 2*alpha2*b2*beta2*e1*l + 2*alpha2*b2*beta2*f2 - alpha2*b2*d1*gamma1*l**2 - alpha2*b2*d1*gamma2*l - alpha2*b2*e1*gamma1*l**2 - alpha2*b2*e1*gamma2*l - 2*alpha2*b2*e2*gamma1*l - 2*alpha2*b2*e2*gamma2 - alpha2*beta1*c1*d1*l**3 - alpha2*beta1*c1*e1*l**3 - 2*alpha2*beta1*c1*e2*l**2 - alpha2*beta1*c2*d1*l**2 - alpha2*beta1*c2*e1*l**2 - 2*alpha2*beta1*c2*e2*l - alpha2*beta2*c1*d1*l**2 - alpha2*beta2*c1*e1*l**2 - 2*alpha2*beta2*c1*e2*l - alpha2*beta2*c2*d1*l - alpha2*beta2*c2*e1*l - 2*alpha2*beta2*c2*e2 + 2*alpha2*c1*d1*gamma1*l**3 + 2*alpha2*c1*d1*gamma2*l**2 + 2*alpha2*c1*d2*gamma1*l**2 + 2*alpha2*c1*d2*gamma2*l + 2*alpha2*c2*d1*gamma1*l**2 + 2*alpha2*c2*d1*gamma2*l + 2*alpha2*c2*d2*gamma1*l + 2*alpha2*c2*d2*gamma2 - b1**2*e1*fi1*l**4 - b1**2*e1*fi2*l**3 - b1**2*f2*fi1*l**3 - b1**2*f2*fi2*l**2 + b1**2*gamma1**2*l**4 + 2*b1**2*gamma1*gamma2*l**3 + b1**2*gamma2**2*l**2 - 2*b1*b2*e1*fi1*l**3 - 2*b1*b2*e1*fi2*l**2 - 2*b1*b2*f2*fi1*l**2 - 2*b1*b2*f2*fi2*l + 2*b1*b2*gamma1**2*l**3 + 4*b1*b2*gamma1*gamma2*l**2 + 2*b1*b2*gamma2**2*l - 2*b1*beta1*c1*gamma1*l**4 - 2*b1*beta1*c1*gamma2*l**3 - 2*b1*beta1*c2*gamma1*l**3 - 2*b1*beta1*c2*gamma2*l**2 - 2*b1*beta2*c1*gamma1*l**3 - 2*b1*beta2*c1*gamma2*l**2 - 2*b1*beta2*c2*gamma1*l**2 - 2*b1*beta2*c2*gamma2*l + b1*c1*d1*fi1*l**4 + b1*c1*d1*fi2*l**3 + b1*c1*e1*fi1*l**4 + b1*c1*e1*fi2*l**3 + 2*b1*c1*e2*fi1*l**3 + 2*b1*c1*e2*fi2*l**2 + b1*c2*d1*fi1*l**3 + b1*c2*d1*fi2*l**2 + b1*c2*e1*fi1*l**3 + b1*c2*e1*fi2*l**2 + 2*b1*c2*e2*fi1*l**2 + 2*b1*c2*e2*fi2*l - b2**2*e1*fi1*l**2 - b2**2*e1*fi2*l - b2**2*f2*fi1*l - b2**2*f2*fi2 + b2**2*gamma1**2*l**2 + 2*b2**2*gamma1*gamma2*l + b2**2*gamma2**2 - 2*b2*beta1*c1*gamma1*l**3 - 2*b2*beta1*c1*gamma2*l**2 - 2*b2*beta1*c2*gamma1*l**2 - 2*b2*beta1*c2*gamma2*l - 2*b2*beta2*c1*gamma1*l**2 - 2*b2*beta2*c1*gamma2*l - 2*b2*beta2*c2*gamma1*l - 2*b2*beta2*c2*gamma2 + b2*c1*d1*fi1*l**3 + b2*c1*d1*fi2*l**2 + b2*c1*e1*fi1*l**3 + b2*c1*e1*fi2*l**2 + 2*b2*c1*e2*fi1*l**2 + 2*b2*c1*e2*fi2*l + b2*c2*d1*fi1*l**2 + b2*c2*d1*fi2*l + b2*c2*e1*fi1*l**2 + b2*c2*e1*fi2*l + 2*b2*c2*e2*fi1*l + 2*b2*c2*e2*fi2 + beta1**2*c1**2*l**4 + 2*beta1**2*c1*c2*l**3 + beta1**2*c2**2*l**2 + 2*beta1*beta2*c1**2*l**3 + 4*beta1*beta2*c1*c2*l**2 + 2*beta1*beta2*c2**2*l + beta2**2*c1**2*l**2 + 2*beta2**2*c1*c2*l + beta2**2*c2**2 - c1**2*d1*fi1*l**4 - c1**2*d1*fi2*l**3 - c1**2*d2*fi1*l**3 - c1**2*d2*fi2*l**2 - 2*c1*c2*d1*fi1*l**3 - 2*c1*c2*d1*fi2*l**2 - 2*c1*c2*d2*fi1*l**2 - 2*c1*c2*d2*fi2*l - c2**2*d1*fi1*l**2 - c2**2*d1*fi2*l - c2**2*d2*fi1*l - c2**2*d2*fi2")


class cell():
    def __init__(self, lx, ly, lz):
       self.lx = lx
       self.ly = ly
       self.lz = lz

class ellipsoid():
    def __init__(self, center=None, u=None, v=None, w=None):
        self.center = center # center
        self.u = u # vectors along main axes
        self.v = v
        self.w = w

    def getMatrix(self):
        """getting matrix A: xT*A*x=0 - ellipse equation"""
        center = self.center
        u = self.u
        v = self.v
        w = self.w
        lu = (u[0]**2 + u[1]**2 + u[2]**2)**0.5
        lv = (v[0]**2 + v[1]**2 + v[2]**2)**0.5
        lw = (w[0]**2 + w[1]**2 + w[2]**2)**0.5
        a = (u[0]**2/lu**2 + v[0]**2/lv**2 + w[0]**2/lw**2)
        b = (u[0]*u[1]/lu**2 + v[0]*v[1]/lv**2 + w[0]*w[1]/lw**2)
        c = (u[0]*u[2]/lu**2 + v[0]*v[2]/lv**2 + w[0]*w[2]/lw**2)
        d = (u[1]**2/lu**2 + v[1]**2/lv**2 + w[1]**2/lw**2)
        e = (u[1]*u[2]/lu**2 + v[1]*v[2]/lv**2 + w[1]*w[2]/lw**2)
        f = (u[2]**2/lu**2 + v[2]**2/lv**2 + w[2]**2/lw**2)
        fi = (a*center[0]**2 + d*center[1]**2 + f*center[2]**2 + 
              2*b*center[0]*center[1] + 2*c*center[0]*center[2] +
              2*e*center[1]*center[2] - 1)
        alpha = -a*center[0] - b*center[1] - c*center[2]
        beta = -b*center[0] -d*center[1] - e*center[2]
        gamma = -c*center[0] - e*center[1] - f*center[2]
        coeffs = [a, b, c, alpha, d, e, beta, f, gamma, fi]
        return coeffs

class ifEllipsoidsIntersect():
    def __init__(self, ellipsoid1, ellipsoid2):
        self.ellipsoid1 = ellipsoid1
        self.ellipsoid2 = ellipsoid2

    def check(self):
        coeffs1 = self.ellipsoid1.getMatrix()
        a1 = sp.Symbol('a1')
        b1 = sp.Symbol('b1')
        c1 = sp.Symbol('c1')
        alpha1 = sp.Symbol('alpha1')
        d1 = sp.Symbol('d1')
        e1 = sp.Symbol('e1')
        beta1 = sp.Symbol('beta1')
        f1 = sp.Symbol('f1')
        gamma1 = sp.Symbol('gamma1')
        fi1 = sp.Symbol('fi1')
        coeffs2 = self.ellipsoid2.getMatrix()
        a2 = sp.Symbol('a2')
        b2 = sp.Symbol('b2')
        c2 = sp.Symbol('c2')
        alpha2 = sp.Symbol('alpha2')
        d2 = sp.Symbol('d2')
        e2 = sp.Symbol('e2')
        beta2 = sp.Symbol('beta2')
        f2 = sp.Symbol('f2')
        gamma2 = sp.Symbol('gamma2')
        fi2 = sp.Symbol('fi2')

        lam = sp.Symbol('l')

        npMatrix = np.array(
         [
          [a2+lam*a1, b2+lam*b1, c2+lam*c1, alpha2+lam*alpha1],
          [b2+lam*b1, d2+lam*d1, e2+lam*d1, beta2+lam*beta1],
          [c2+lam*c1, e2+lam*e1, f2+lam*e1, gamma2+lam*gamma1],
          [alpha2+lam*alpha1, beta2+lam*beta1, gamma2+lam*gamma1, fi2+lam*fi1]
         ])
        spMatrix = sp.Matrix(npMatrix)
        det = copy.deepcopy(determinant)
        det_lam = det.subs({a1:coeffs1[0],
                        b1:coeffs1[1],
                        c1:coeffs1[2],
                        alpha1:coeffs1[3],
                        d1:coeffs1[4],
                        e1:coeffs1[5],
                        beta1:coeffs1[6],
                        f1:coeffs1[7],
                        gamma1:coeffs1[8],
                        fi1:coeffs1[9],
                        a2:coeffs2[0],
                        b2:coeffs2[1],
                        c2:coeffs2[2],
                        alpha2:coeffs2[3],
                        d2:coeffs2[4],
                        e2:coeffs2[5],
                        beta2:coeffs2[6],
                        f2:coeffs2[7],
                        gamma2:coeffs2[8],
                        fi2:coeffs2[9]})
        e = sp.Eq(det_lam)
        print(e)
        roots = sp.solveset(e)
        print(roots)
        realRoots = []
        for root in roots:
            if root.is_real is True:
                realRoots.append(float(root))
        if (len(realRoots) < 2):
            return True
        realRoots.sort()
        #print('ROOTS, ', realRoots)
        if (realRoots[-1] > 0 and realRoots[-2] > 0 and
            abs(realRoots[-1] - realRoots[-2]) > epsilon):
            return False
        return True

def addEllipsoid(ellipsoids, l, r, h):
    for i in range(tries):
        center1 = random.random()  * (l - r - h)
        center2 = random.random() * (l - r - h)
        center3 = random.random() * (l - r - h)
        center = [center1, center2, center3]
        print('try ', i, center)
        (u, v, w) = calculateAxis(r, h)
        newEllipsoid = ellipsoid(center, u, v, w)
        flagIntersection = False
        for j in range(len(ellipsoids)):
            print('intersecting with ellipsoid ', j)
            iei = ifEllipsoidsIntersect(ellipsoids[j], newEllipsoid)
            flagIntersection = iei.check()
            if flagIntersection == True:
                print('Intersect')
                break
            print('Separate')
        if flagIntersection == True:
            newEllipsoid = None
            continue
    #print('add ', newEllipsoid.center)
    return newEllipsoid


def calculateAxis(r, h):
    u1 = 2 * (random.random() - 0.5) * r
    u2 = 2 * (random.random() - 0.5) * (r**2 - u1**2)**0.5
    u3 = (r**2 - u1**2 - u2**2)**0.5
    v2 = (r**2 / (1 + (u2/u1)**2))**0.5
    v1 = (r**2 - v2**2)**0.5
    v3 = 0
    w2 = h / (((u1/u2)**2 + 1 + ((u1**2+u2**2)/(u2*u3)**2)**2)**0.5)
    w1 = w2*u1/u2
    w3 = -w2*((u1**2+u2**2)/(u2*u3))
    u = [u1, u2, u3]
    v = [v1, v2, v3]
    w = [w1, u2, w3]
    return (u, v, w)


def main(ellipsoids, l, r, h, depth=0):
    ellipsoids = []
    center = [random.random() * l,
              random.random() * l,
              random.random() * l]
    (u, v, w) = calculateAxis(r, h)
    newEllipsoid = ellipsoid(center, u, v, w)
    ellipsoids.append(newEllipsoid)
    while len(ellipsoids) < 5:
        for oldEllipsoid in ellipsoids:
            print(oldEllipsoid.center)
            print(oldEllipsoid.u)
            print(oldEllipsoid.v)
            print(oldEllipsoid.w)
        for j in range(tries + 1):
            if j == tries:
                print('cannot append ellipsod')
                return ellipsoids
            intersectionFlag = False
            center = [random.random() * l,
                      random.random() * l,
                      random.random() * l]
            (u, v, w) = calculateAxis(r, h)
            newEllipsoid = ellipsoid(center, u, v, w)
            print((str(len(ellipsoids) + 1) + 'th ellipsoid, try ' + str(j)))
            print(newEllipsoid.center)
            print(newEllipsoid.u)
            print(newEllipsoid.v)
            print(newEllipsoid.w)
            for oldEllipsoid in ellipsoids:
                iei = ifEllipsoidsIntersect(oldEllipsoid, newEllipsoid)
                if iei.check():
                    intersectionFlag = True
                    break
            if intersectionFlag == False:
                ellipsoids.append(newEllipsoid)
                break
            else:
                continue


def tester(l=10, r=0.5, h=0.1):
    for i in range(10):
        c1 = [random.random() * l,
              random.random() * l,
              random.random() * l]
        (u1, v1, w1) = calculateAxis(r, h)
        c2 = [random.random() * l,
              random.random() * l,
              random.random() * l]
        (u2, v2, w2) = calculateAxis(r, h)
        el1 = ellipsoid(c1, u1, v1, w1)
        el2 = ellipsoid(c2, u2, v2, w2)
        print('el1')
        print(el1.center[0], el1.center[1], el1.center[2])
        print(el1.u[0], el1.u[2], el1.u[2])
        print(el1.v[0], el1.v[2], el1.v[2])
        print(el1.w[0], el1.w[2], el1.w[2])
        print('el2')
        print(el2.center[0], el2.center[1], el2.center[2])
        print(el2.u[0], el2.u[2], el2.u[2])
        print(el2.v[0], el2.v[2], el2.v[2])
        print(el2.w[0], el2.w[2], el2.w[2])
        iei = ifEllipsoidsIntersect(el1, el2)
        flag = iei.check()
        print(flag)
        print('')

tester()
#els = main([], 10000, 5, 1)
#for el in els:
#    print(el.center, el.u, el.v, el.w)
