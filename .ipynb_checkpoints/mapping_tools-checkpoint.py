import numpy as np
from emission_curves import EmissionCurve
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import interp2d

class MapBook:
    
    def __init__(self, Es):
        self.Es = Es
        self.maps = {}
        self.generate_maps(Es)
    
    def generate_maps(self, Es):
        for e in Es:
            self.maps[e] = EMap(e)
            print(f"Map {e} generated")

    def plot_map(self, e):
        self.maps[e].show_3dmap()

    def __getitem__(self, item):
        return self.maps[item]

    def __str__(self):
        return f"MapBook containing {len(self.maps)} maps"

    def __repr__(self):
        return self.__str__()

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

        self.e = e
        # Define radii and generate emission curves
        self.rs = np.linspace(1e-7,2e-3,51)
        self.curves = [EmissionCurve(E=e,r=r) for r in self.rs]

        # Extract the angles and maps from the curves
        self.angles = self.extract_angle()
        self.map = self.extract_map()
        self.logmap = np.log10(self.map)

        self.radii = np.array([self.rs for i in range(len(self.angles))]).T
        self.intermap = interp2d(self.radii, self.angles, self.map)
        self.rangemap = interp2d(self.map, self.angles, self.radii)
        
    def get_value(self, r, theta):
        return self.intermap(r,theta)

    def show_3dmap(self):
        # Create figure
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        
        # Make grid of radii
        radii = np.array([self.rs for i in range(len(self.angles))])
        
        # Plot surface
        surf = ax.plot_surface(self.angles.T*180/np.pi, radii, self.logmap, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.zaxis.set_major_formatter('{x:.02f}')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    def show_3dmap_range(self):
        # Create figure
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        
        # Make grid of radii
        radii = np.array([self.rs for i in range(len(self.angles))])
        
        # Plot surface
        surf = ax.plot_surface(self.logmap, self.angles.T*180/np.pi, radii, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.zaxis.set_major_formatter('{x:.02f}')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    def get_range_distribution(self, distribution):
        ranges = []
        for values in distribution.values():
            for v in values:
                ranges.append(self.rangemap(v[1],v[0])[0])
        return ranges

    def get_allowed_angles(self, r):
        if self.e < 0:
            angles = np.linspace(np.arctan(1e-3/r), np.pi/2, 53)[1:-1]
        else:
            angles = np.linspace(0,np.arctan(1e-3/r), 53)[1:-1]
        return min(angles), max(angles)
        
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