# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 18:46:04 2021

@author: willi
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
pio.renderers.default='svg'


class EmissionCurve:
    
    d = 1e-3
    me = 9.109383e-31
    q = -1.602e-19

    def __init__(self, r, E, angles=None):
        self.r = r
        self.E = E
        
        if angles is None:
            self.angles = np.linspace(0,np.arctan(self.d/r), 53)[1:-1]
        else:
            self.angles = angles * np.pi / 180

        self.curve = self.generate_curve(r=r, E=E, angles=self.angles)
            
    def generate_curve(self, r=None, E=None, angles=None):
        if r is None:
            r = self.r
        if E is None:
            E = self.E
        if angles is None:
            angles = self.angles
        return (self.q*E*r**2) / (4*((np.cos(angles))**2) * (self.d - np.tan(angles)*r))

    def plot_curve(self, angles=None, curve=None):
        if angles is None:
            angles = self.angles
        if curve is None:
            curve = self.curve
        plt.plot(angles*180/np.pi, np.log(curve))
        plt.xlabel("Angle (deg)")
        plt.ylabel("Emission energy")
        plt.show()

class EmissionCurveZero:

    d = 1e-3
    me = 9.109383e-31
    q = -1.602e-19

    def __init__(self, r0, r1, dE):
        self.r0 = r0
        self.r1 = r1
        self.dE = dE
        self.angle = np.arctan(self.d/r0)
        self.data = pd.DataFrame(columns=["x","y","t","dE","r0","t1","E0"])

        self.co = np.cos(self.angle)
        self.si = np.sin(self.angle)

        self.E0 = self.get_E0()
        self.t1 = self.get_t_collision()
        self.directory = f"D:\\OneDrive - Danmarks Tekniske Universitet\\OneDrive\Dokumenter\\DTU\\Terahertz\\PaperExperiments\\measurements\\EmissionCurves\\{r0}_{r1}_{dE}.html"
        # self.display(dEs=[2*dE,dE,0,-dE,-2*dE], E0s=np.linspace(self.E0 - self.E0*0.8, self.E0 + self.E0*0.8, 17), directory=self.directory)

    def x(self, t, E0=None, angle=None):
        if E0 == None:
            E0 = self.E0
        if angle is None:
            co = self.co
        else:
            co = np.cos(angle)
        return co * np.sqrt(2*E0/self.me) * t
    
    def y(self, t, E0=None, angle=None, dE=None):
        if E0 == None:
            E0 = self.E0
        if angle is None:
            si = self.si
        else:
            si = np.sin(angle)
        if dE is None:
            dE = self.dE

        return (si * np.sqrt(2*E0/self.me) * t + (1/2) * self.q * dE * t**2 / self.me)
            

    def get_t_collision(self, E0=None, angle=None, d=None, dE=None):
        if E0 == None:
            E0 = self.E0
        if angle is None:
            si = self.si
            co = self.co
        else:
            si = np.sin(angle)
            co = np.cos(angle)
        if d is None:
            d = self.d
        if dE is None:
            dE = self.dE
            
        if dE == 0:
            return (d / si) * np.sqrt(self.me / 2 / E0)
            
            
        inner = self.me*((-E0 * co**2)
                          + (dE * d * self.q)
                          + E0)
        
        if inner < 0:
            inner = self.me*((-E0 * co**2)
                          + (abs(dE) * d * self.q)
                          + E0)
        
        t1 = (np.sqrt(2) * (
                (-si * self.me * np.sqrt(E0/self.me))
                + (np.sqrt(inner
                            )
                    )
                )
            / (
                self.q * dE
                )
        )
        return abs(t1)

    def generate_positions(self, dE=None, E0=None):
        if dE is None:
            dE = self.dE
        if E0 is None:
            E0 = self.E0

        t1 = self.get_t_collision(dE=dE, E0=E0)
            
        times = np.linspace(0,1,101) * t1
        print(t1)
        ys = self.y(times, dE=dE, E0=E0)
        xs = self.x(times, E0=E0)
        self.data = self.data.append(pd.DataFrame([[x,y,t,dE,self.r0,t1, E0] for x,y,t in zip(xs,ys,times)],
                                  columns=["x","y","t","dE","r0","t1","E0"]))

    def __add__(self, other):
        return self.data.append(other.data)


    def get_E0(self):
        return abs((self.q*self.dE*self.r1**2) / (4*(self.co*self.si*self.r1 + self.d*self.si**2 - self.d)))

    def display(self, dEs=[], E0s=[], directory=None):
        
        for dE in dEs:
            for E0 in E0s:
                self.generate_positions(dE=dE, E0=E0)

        self.fig = px.scatter(self.data, x="x", y="y", color="dE", animation_frame="E0")
        self.fig.show()
        
        if directory is not None:
            self.fig.write_html(directory)
        
        return self.fig

    
class RangeSurface:
    d = 1e-3
    me = 9.109383e-31
    q = -1.602e-19

    def __init__(self, E, es=None, angles=None):
        self.E = E
        self.es = es
        self.angles=angles
        
    def get_r(self, E=None, es=None, angles=None):
        if es is None and self.es is None:
            es = np.linspace(1e-30,1e-35,100)
        elif es is None:
            es = self.es

        if angles is None and self.angles is None:
            angles = np.linspace(-np.pi/2,np.pi/2,45)
        elif es is None:
            angles= self.angles
        
        if E is None:
            E = self.E
        
        es, angles = np.meshgrid(es, angles)

        return -2*(es*np.sin(angles) - np.sqrt(es**2 * (np.sin(angles))**2 + self.q*E*es*self.d)) * np.cos(angles) / (self.q*E)

from scipy.optimize import fmin, minimize_scalar

def get_boundary(x):
    rmax, E, theta = [2e-2, -200, np.pi/16/4/2]
    range_curve = RangeSurface(E=E)
    dr = 1
    e = x
    r = range_curve.get_r(es=np.array([e]), angles=np.array([theta]))
    dr = abs(rmax - r)
    # while dr > 1e-5:
    #     if r > rmax:
    #         break
    #     e *= 10
    #     r = range_curve.get_r(es=np.array([e]), angles=np.array([theta]))
    #     dr = abs(rmax - r)
    #     print(e, dr, r)
    
    # while dr > 1e-5:
    #     e *= 0.99
    #     r = range_curve.get_r(es=np.array([e]), angles=np.array([theta]))
    #     dr = abs(rmax - r)
    #     print(e, dr, r)
    
    return dr


class Electron:
    
    d = 1e-3
    q = -1.602e-19
    
    def __init__(self, E=None, theta=None):
        
        if theta is None:
            self.theta = np.pi*9/20
        else:
            self.theta = theta
            
        if E is None:
            self.E = 0
        else:
            self.E = E
            
    def __str__(self):
        return f"Angle : {self.theta}\nEnergy : {self.E}\n"

    def get_r(self, field):        
        return -2*(self.E*np.sin(self.theta) - np.sqrt(self.E**2 * (np.sin(self.theta))**2  \
                    + self.q*field*self.E*self.d)) * np.cos(self.theta) / (self.q*field)


class ElectronCluster:
    
    def __init__(self, N=None, E=None, theta=None):

        self.electrons = self.make_electrons(N, E, theta)
        self.distribution = pd.DataFrame(columns=["range", "energy", "theta", "field"])


    def make_electrons(self, N=None, E=None, theta=None):
        if E is None and theta is None:
            return [Electron(E=None, theta=None) for N in range(N)]
        elif N is None:
            return [Electron(E=e, theta=t) for e,t in zip(E,theta)]

    def get_distribution(self, field):
        self.distribution[field] = self.distribution.append(pd.DataFrame([[e.get_r(field), e.E, e.theta, field] for e in self.electrons], 
                                                columns=["range", "energy", "theta", "field"]), ignore_index=True)
        self.distribution.loc[self.distribution.field == field] /= np.max(self.distribution[self.distribution.field == field])
        return self.distribution.loc[self.distribution.field == field]

    def plot_distribution(self, field):
        if field in list(self.distribution):
            dist = self.distribution[field]
        else:
            dist = self.get_distribution(field)
    
        
cluster = ElectronCluster(N=1000)


# distances = np.linspace(0.01,1,100)
# true_shape = abs(np.sin(distances*np.pi*2))/(distances+0.1)  + 0.2
# true_shape /= np.max(true_shape)

# plt.figure()
# plt.plot(distances, true_shape)
# plt.xlabel("Distance")
# plt.ylabel("Norm. Counts")
# plt.title("True Shape")


# def guess()



if __name__ == '__main__':
    # curve = EmissionCurve(r0=1e-3, r1=1e-3-1e-4, dE=-1e2)
    # for energy in np.logspace(-25, -18, 11):
    #     ranges = [RangeSurface(E=E) for E in np.linspace(-200,200,41)]
    #     ran = [r.get_r(es=np.array([energy]), angles=np.linspace(-np.pi/4,np.pi/2-(np.pi*3/4/21),21)) for r in ranges]
    #     plt.figure()
    #     im = plt.imshow(np.log(ran))
    #     plt.title(str(energy))
    #     plt.colorbar()
    #     plt.show()
    # print(minimize_scalar(get_boundary))
    xs = np.logspace(-25,-17, 201)
    rs = get_boundary(xs).reshape(201,)
    # plt.plot(xs, rs)
    
    # plt.figure()
    xs = np.logspace(-35,-25, 201)
    rs = get_boundary(xs).reshape(201,)
    # plt.plot(xs, rs)
    # print(get_boundary(1e-40))
    