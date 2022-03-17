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

        # Define radii and generate emission curves
        self.rs = np.linspace(1e-7,2e-3,51)
        self.curves = [EmissionCurve(E=e,r=r) for r in self.rs]

        # Extract the angles and maps from the curves
        self.angles = self.extract_angle()
        self.map = self.extract_map()
        self.logmap = np.log(self.map)

        radii = np.array([self.rs for i in range(len(self.angles))]).T
        self.intermap = interp2d(radii, self.angles, self.map)
        
    def get_value(self, r, theta):
        return self.intermap(r,theta)

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

    def get_allowed_angles(self, r):
        
        lower_limit = np.min(self.angles)
        upper_limit = np.max(self.angles)
        angs = np.linspace(lower_limit, upper_limit, 201)
        for ang in angs:
            if self.intermap(r,ang) != 0:
                lower_limit = ang
                break
        for ang in reversed(angs):
             if self.intermap(r,ang) != 0:
                upper_limit = ang
                break
        return lower_limit, upper_limit
        
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