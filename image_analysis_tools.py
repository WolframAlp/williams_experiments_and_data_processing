import numpy as np
from importPicture import *
import os
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import find_peaks

from scipy.ndimage import gaussian_filter, median_filter


class ImageSeries:

    '''Class holding capable of loading and holding image information on
    a series of images'''

    def __init__(self, main=None, others=None, center=None):

        # Get working directory for returning later
        cdir = os.getcwd()

        if main is not None:
            # Change directory to main and get list of content.
            # If content satisfies being an image and being alone it is loaded
            if main.endswith('.png') or main.endswith('.tif'):
                self.main = Image.open(main)
                self.main_data = np.asarray(self.main)
                main_name = main.split("\\")[-1]
                print(f"Main file: {main_name}, loaded")
            else:
                os.chdir(main)
                main_content = os.listdir()
                if len(main_content) == 0:
                    os.chdir(cdir)
                    raise Exception("No image in main folder")
                if len(main_content) > 1:
                    os.chdir(cdir)
                    raise Exception("Too many files in main folder")

                # Load the image and save the numpy array
                self.main = Image.open(main_content[0])
                self.main_data = np.asarray(self.main)

                print(f"Main file: {main_content[0]}, loaded")

            # Finds the center spot using import pictures function or uses provided center
            if center is None:
                self.center = findCenterSpot(self.main_data)
                print("Center coordinates:", self.center)
            else:
                self.center = np.array(center)
                print("Using provided center value: ", center)

        if others is not None:
            # Change directory to other set of images and check if they exist
            os.chdir(others)
            others_content = os.listdir()
            if len(others_content) == 0:
                os.chdir(cdir)
                raise Exception("No images in others folder")

            ## Get image names
            # Creates a splitting of each name 
            undersplit = [name.split("_") for name in others_content] # TODO reevaluate how to split and select names [-18:-4]
            
            self.num_images = {}
            self.positions = []
            self.voltages = []
            self.others_names = []
            
            for split in undersplit:
                imag = None
                p = 0
                v = 0
                for i,part in enumerate(split):
                    if part == 'img':
                        imag = split[i+1]
                    elif part == 'pos':
                        if split[i+1] == "0.0":
                            p = 0
                            os.rename(f"img_{split[1]}_pos_{split[3]}_{split[4]}", f"img_{split[1]}_pos_0_{split[4]}")
                        else:
                            p = int(float(split[i+1])) # Does not work with int for energy considerations
                        if p not in self.positions:
                            self.positions.append(p)
                    elif part.startswith("volt"):
                        v = part.split(".")
                        if len(v) > 2:
                            os.rename(f"img_{split[1]}_pos_{split[3]}_{split[4]}", f"img_{split[1]}_pos_{split[3]}_{v[0]}.{v[-1]}")
                        v = int(v[0][4:])
                        if v not in self.voltages:
                            self.voltages.append(v)
                im_name = f"pos_{p}_volt{v}"
                if imag is not None:
                    if im_name in self.num_images:
                        self.num_images[im_name].append(imag)
                    else:
                        self.num_images[im_name] = [imag]
                if im_name not in self.others_names:
                    self.others_names.append(im_name)
            
            # sort the voltages and the names before creating dict
            self.voltages, self.others_names = self._sort_volts_names(self.voltages, self.others_names)
            
            # Runds though all image names 
            self.others_data = {}
            self.num_settings = len(self.others_names)
            print("Number of settings: ", self.num_settings)

            for n,name in enumerate(self.others_names):

                if name in self.num_images:
                    num_images = self.num_images[name]
                    for i,num in enumerate(num_images):
                        full_name = f"img_{num}_{name}.png"
                        if i == 0:
                            image = np.array(Image.open(full_name), dtype=float)
                        else:
                            image += np.array(Image.open(full_name), dtype=float)
                    self.others_data[name] = image / len(num_images)
                else:
                    num_images = None
                    full_name = f"{name}.png"
                    self.others_data[name] = np.array(Image.open(full_name), dtype=float)

            # Saves a CImage object for each averaged image
            self.others = {im: CImage(self.others_data[im], im) for im in self.others_names}

            print(f"Loaded {len(self.others)} other images")

        # Return to original directory
        os.chdir(cdir)


    def __str__(self):
        tex = f"Main image center spot: {self.center}\n"
        tex += f"Number of other images: {len(self.others)}"
        return tex

    def __repr__(self):
        return self.__str__()

    def _sort_volts_names(self, volts, names):
        '''Sorts the voltages and others_names to be in correct order'''
        sort_names = []
        volts = sorted(volts)
        for volt in volts:
            for name in names:
                if str(volt) in name:
                    sort_names.append(name)
        return volts, sort_names
    
    def apply_gaussian_filter(self, save=False, sigma=3):
        '''Applies a gaussian filter to the image with a width of sigma
        if save is set it will update the data in class,
        else the data will be returned'''
        if save:
            for im, cim in self.others.items():
                cim.update_data(gaussian_filter(cim.data,sigma))
        else:
            return [gaussian_filter(cim.data,sigma) for im,cim in self.others.items()]
    
    def apply_median_filter(self, save=False, size=3):
        '''Applies a gaussian filter to the image with a width of sigma
        if save is set it will update the data in class,
        else the data will be returned'''
        if save:
            for im, cim in self.others.items():
                cim.update_data(median_filter(cim.data,size=size))
        else:
            return [median_filter(cim.data,size=size) for im,cim in self.others.items()]
    
    def save_reduced_set(self, path):
        '''Saves the averaged images to a path with names given by voltage setting'''
        if not os.path.exists(path):
            os.mkdir(path)
        for m in self.others_names:
            image = Image.fromarray(self.others_data[m])
            image = image.convert("L")
            try:
                image.save(path + f"\\{m}.png")
            except:
                image.save(path + f"\\{m}.tif")
        print("Averaged images saved to folder: ", path)
    
    def get_image_by_potential(self, V):
        '''Takes potential as input and outputs image as array'''

        if type(V) is int or type(V) is float:
            V = str(V)

        if V in self.others_names:
            return self.others_data[V]
        else:
            print("Image was not found")
            return None

    def subtract_background_all(self, background):
        '''Subtracts the provided background from all the other images'''
        for im, cim in self.others.items():
            cim.update_data(cim.data - background)

    def subtract_backgrounds(self, backgrounds, fronts):
        '''Subtracts the provided background from all the other images'''
        for im, background in zip(fronts.items(), backgrounds.items()):
            im[1].update_data(im[1].data - background[1].data)
            dat = im[1].data
            dat[dat < 0] = 0
            im[1].update_data(dat)
        return fronts
        
    def slice_images(self, center=None, angle=5, unit="degrees", assignment="binary"):
        '''Slices all the images with the selected parameters'''

        if center is None:
            center = self.center

        for name, image in self.others.items():
            image.slice_image(center=center, angle=angle, unit=unit, assignment=assignment)


class CImage:
    '''Holds more detailed information on each image and can do different manipulations'''

    def __init__(self, data, name):
        self.data = data
        self.name = name
        self.image = Image.fromarray(self.data)
        self.slices = {}
        self.angles = []
        self.slice_intensities = []
        self.binned = data
        self.logged = np.log(data)

    def update_data(self, data):
        '''Sets the data to the new provided values'''
        self.data = data
        self.image = Image.fromarray(self.data)

    def apply_mask(self, mask):
        self.data[mask] = 0
        self.update_data(self.data)

    def show(self):
        '''Returns the image object of the data'''
        self.image = self.image.convert("L")
        return self.image

    def show_binned(self):
        '''Shows the binned data using matplotlib'''
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
        fig.suptitle(self.name, fontsize=16)
        
        im1 = ax1.imshow(self.data, aspect='auto')
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="10%", pad=0.05)
        cbar1 = plt.colorbar(im1, cax=cax1)
        
        im2 = ax2.imshow(self.binned, aspect='auto')
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="10%", pad=0.05)
        cbar2 = plt.colorbar(im1, cax=cax2)
        
        plt.subplots_adjust(top=0.85)
        plt.show()

    def bin_logarithmic(self, bins=10):
        '''Bins logarithmic dataset'''
        self.bin_data(bins=bins, domain="log")

    def reset_log_data(self):
        '''Resets the values of the log data to the real data'''
        self.logged = np.log(data)

    def bin_data(self, bins=10, domain="Linear"):
        '''Bins intensity to specific number of values, down from 255
        domain set to linear will bin the linear data, while any other
        domain setting will bin the logarithmic data'''
        if domain.lower() == "linear":
            data = self.data
        else:
            data = self.logged

        mi, ma = np.min(data), np.max(data)
        step = (ma - mi) / bins
        hstep = step / 2
        bin_vals = np.round([mi + step/2] + [mi + i * step + step/2  for i in range(1,bins-2)] + [ma - step/2],2)

        for b in bin_vals:
            m1, m2 = self.data > (b-hstep), self.data < (b+hstep)
            mask = m1 & m2
            if domain.lower() == "linear":
                self.binned[mask] = int(round(b))
            else:
                self.logged[mask] = np.log(b)

    def __str__(self):
        return f"Image: {self.name}\nSize: {self.data.shape}"

    def __repr__(self):
        return self.__str__()

    def slice_image(self, center, angle=5, unit="degrees", assignment="binary"):
        '''Slices image into a number of slices given by 360/angle or 2pi / angle if radians
        The pixels inside each slice is assigned to the slice for summation and radial plots.
        Assignments supported are binary (either one slice or the other) and
        split (area inside each slice is calculated). split is the smoother option but higher comp time.'''
        if unit.lower() == "radians":
            angle = angle * 180 / np.pi
        
        if assignment.lower() not in ["binary", "weighted"]:
            raise Exception(f"Assignment method {assignment}, not supported. Supported assignments are : [binary, weighted]")
        
        if 360 % angle !=0:
            raise Exception("Anlge is does not add to 360 degrees when multiplied by integer number")
        
        num_slices = int(360 / angle)

        self.slices = {}
        for a in range(num_slices):
            self.angles.append(a*angle)
            self.slices[a*angle] = AngleSlice(data=self.data, center=center, lower=a*angle, step=angle, assignment=assignment)

    def get_phi_dependence(self):
        '''Gets the sum of pixels in each slice.
        Showing the total electron intensity at that angle.
        Angles and sums are returned.'''
        self.angles = []
        self.slice_intensities = []
        for angle,sli in self.slices.items():
            self.angles.append(angle)
            self.slice_intensities.append(self.slices[angle].get_total_intensity())
        return self.angles, self.slice_intensities

    def plot_phi_dependence(self):
        '''Plots the phi dependence, so you don't have to.'''
        if len(self.slice_intensities) == 0:
            print("Get phi dependence before plotting")
            return
        plt.figure()
        plt.plot(self.angles,np.array(self.slice_intensities)/np.max(self.slice_intensities))
        plt.xlabel("Angles (deg)")
        plt.ylabel("Nom. Intensity")
        plt.title(self.name)
        plt.grid()
        plt.show()

class AngleSlice:
    '''Holds infomation on pixels in specific area of opening angle
    given in the CImage, set as step'''

    def __init__(self, data, center, lower=0, step=5, assignment="binary"):
        self.step = step
        self.assignment = assignment.lower()
        self.lower = lower
        self.upper = lower + step
        self.lower_rad = lower * np.pi / 180
        self.upper_rad = self.upper * np.pi / 180

        if (self.lower >= 45 and self.lower <= 135) or (self.lower >= 225 and self.lower <= 315):
            self.sweep = 'y'

            # Define start positions and intensity list
            xu, xl = center[0], center[0]
            yval = 0
            self.intensities = []
            self.xls = []
            self.xus = []
            self.ys = []

            # Check if both y positions are within image and xval is within image
            while (yval + center[1] < data.shape[0] and yval + center[1] >= 0):

                #checks assignment method and gets the yl and yu values
                if self.assignment == "binary":
                    xl = int(round(xl,0))
                    xu = int(round(xu,0))
                
                if xl >= data.shape[1]:
                    xl = data.shape[1]
                elif xl < 0:
                    xl = 0
                if xu >= data.shape[1]:
                    xu = data.shape[1]
                elif xu < 0:
                    xu = 0

                if xl < xu:
                    self.xls.append(xl)
                    self.xus.append(xu)
                    self.intensities.append(sum(data[center[1] + yval, xl:xu]) / (abs(xu-xl)))
                elif xl > xu:
                    self.xls.append(xl)
                    self.xus.append(xu)
                    self.intensities.append(sum(data[center[1] + yval, xu:xl]) / (abs(xu-xl)))
                else:
                    self.xls.append(xl)
                    self.xus.append(xu)
                    self.intensities.append(sum(data[center[1] + yval, xl:xu]))

                self.ys.append(yval)
                if self.lower < 180:
                    yval -= 1
                else:
                    yval += 1
                xl, xu = center[0] - (yval) / np.tan(self.lower_rad), center[0] - (yval) / np.tan(self.upper_rad)
            self.xus = np.array(self.xus)
            self.xls = np.array(self.xls)
            self.numY = abs(yval)

        else:
            self.sweep = 'x'

            # Define start positions and intensity list
            yu, yl = center[1], center[1]
            xval = 0
            self.intensities = []
            self.yls = []
            self.yus = []
            self.xs = []

            # Check if both y positions are within image and xval is within image
            while (xval + center[0] < data.shape[1] and xval + center[0] >= 0):

                #checks assignment method and gets the yl and yu values
                if self.assignment == "binary":
                    yl = int(round(yl,0))
                    yu = int(round(yu,0))
                
                if yl >= data.shape[0]:
                    yl = data.shape[0]
                elif yl < 0:
                    yl = 0
                if yu >= data.shape[0]:
                    yu = data.shape[0]
                elif yu < 0:
                    yu = 0

                if yl < yu:
                    self.yls.append(yl)
                    self.yus.append(yu + 1)
                    self.intensities.append(sum(data[yl:yu, center[0] + xval]) / (abs(yu-yl)))
                elif yl > yu:
                    self.yls.append(yl+1)
                    self.yus.append(yu)
                    self.intensities.append(sum(data[yu:yl, center[0] + xval]) / (abs(yu-yl)))
                else:
                    self.yls.append(yl)
                    self.yus.append(yu + 1)
                    self.intensities.append(sum(data[yl:yu, center[0] + xval]))

                self.xs.append(xval)
                if self.lower < 45 or self.lower > 315:
                    xval += 1
                else:
                    xval -= 1
                yl, yu = center[1] - np.tan(self.lower_rad) * (xval), center[1] - np.tan(self.upper_rad) * (xval)
            self.yus = np.array(self.yus)
            self.yls = np.array(self.yls)
            self.numX = abs(xval)
        self.intensities = self.intensities[21:]


    def __str__(self):
        return f"Center : {(self.lower + self.upper)/2} degrees"
    
    def __repr__(self):
        return self.__str__()
    
    def plot(self):
        plt.figure()
        plt.plot(np.arange(len(self.intensities)), self.intensities)
        plt.xlabel("Pixel number -20")
        plt.ylabel("Intensity Average")
        plt.show
    
    def get_peak_centers(self, prominence=1, plot=False, width=5, distance=5):
        
        peaks, specifics = find_peaks(self.intensities, prominence=prominence, width=width, distance=distance)
    
        if plot:
            plt.figure()
            plt.plot(np.arange(len(self.intensities)), self.intensities)
            plt.scatter(peaks, np.array(self.intensities)[peaks])
            plt.show()
    
        return peaks, specifics
    
    def get_total_intensity(self):
        return sum(self.intensities)