# -*- coding: utf-8 -*-
"""
Created on Fri May  7 09:13:11 2021

@author: willi
"""

from lucam import Lucam, API
from importPicture import *

class Camera:
    '''Class for interacting the the Lucam camera'''
    
    def __init__(self, exposure=40):

        # Tries to get the camera
        self.snapshot = API.LUCAM_SNAPSHOT()
        try:
            self.camera = Lucam()
            self.save_snapshot_format()
            self.create_snapshot(exposure=exposure)
            self.connected = True
        except Exception as e:
            # On fail set connected to False
            print("Camera connection error")
            print(e)
            self.connected = False
            return

    def SaveImage(self, image, filename):
        '''Saves image to png file'''

        # Naming conventions are very specific
        # The specific error is therefor outputted
        try:
            self.camera.SaveImage(image, filename)
        except Exception as e:
            print(e)

    def TakeSnapshot(self, snapshot=None):
        '''Take an image'''
        if snapshot is None:
            snapshot = self.snapshot
        return self.camera.TakeSnapshot(snapshot=snapshot)

    def check_positioning(self):
        im = self.TakeSnapshot()
        xsum = make_hist(im, plot=True)
        ysum = make_hist(im, axis=1, plot=True)
        xmax, ymax = xsum.max(), ysum.max()
        val = int(round((xmax + ymax) / 6, 0))
        ellipseFitWithPlot(im, val)
        return im
    
    def save_snapshot_format(self):
        self.snapshotformat = self.camera.GetFormat()[0]
    
    def create_snapshot(self, exposure=500):
        self.snapshot = API.LUCAM_SNAPSHOT()
        try:
            self.snapshot.format = self.camera.GetFormat()[0]
        except:
            self.snapshot.format = self.snapshotformat
        self.snapshot.exposure = exposure
        self.snapshot.gain = 1
        self.snapshot.timeout = 1000.0
        self.snapshot.gainRed = 1.0
        self.snapshot.gainBlue = 1.0
        self.snapshot.gainGrn1 = 1.0
        self.snapshot.gainGrn2 = 1.0
        self.snapshot.useStrobe = False
        self.snapshot.strobeDelay = 0.0
        self.snapshot.useHwTrigger = 0
        self.snapshot.shutterType = 0
        self.snapshot.exposureDelay = 0.0
        self.snapshot.bufferlastframe = 0

