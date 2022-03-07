import numpy as np
from emission_curves import EmissionCurve
import matplotlib.pyplot as plt
from matplotlib import cm

class Map:

    curves = []

    def __init__(self):
        pass

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "This is a great map"

    def __getitem__(self, item):
        return self.curves[item]

    def extract_radius(self):
        return np.array([c.r for c in self.curves])

    def extract_angle(self):
        return np.array([c.angles for c in self.curves])

    def extract_field(self):
        return np.array([c.E for c in self.curves])

    def extract_map(self):
        return np.array([c.curve for c in self.curves])

class EMap(Map):

    def __init__(self, e):
        super().__init__()

        # Define radii and generate emission curves
        self.rs = np.linspace(1e-7,2e-3,301)
        self.curves = [EmissionCurve(E=e,r=r) for r in self.rs]

        # Extract the angles and maps from the curves
        self.angles = self.extract_angle()
        self.map = self.extract_map()
        self.logmap = np.log(self.map)

    def show_3dmap(self):
        # Create figure
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        
        # Make grid of radii
        radii = np.array([self.rs for i in range(len(self.angles))])
        
        # Plot surface
        surf = ax.plot_surface(self.angles*180/np.pi, radii, self.logmap, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.zaxis.set_major_formatter('{x:.02f}')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        
    def show_map(self):
        plt.figure()
        plt.imshow(self.logmap)
        plt.xlabel("Angles")
        plt.ylabel("Ranges")
        angles = self.angles[0]
        plt.xticks([0,
                    int((len(angles)-1)/2),
                    len(angles)-1],
                   [round(angles[0]*180/np.pi,2), 
                    round((angles[0]*180/np.pi+angles[-1]*180/np.pi)/2,2), 
                    round(angles[-1]*180/np.pi,2)])
        plt.yticks([0,
                    int((len(self.rs)-1)/2),
                    len(self.rs)-1],
                   [self.rs[0],
                    round((self.rs[-1]+self.rs[0])/2, 1+int(-np.log10((self.rs[-1]+self.rs[0])/2))),
                    self.rs[-1]])
        plt.colorbar()
        plt.show()

class RMap(Map):

    def __init__(self, r):
        pass

class ThetaMap(Map):

    def __init__(self, theta):
        pass