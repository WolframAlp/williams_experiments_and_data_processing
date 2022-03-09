# -*- coding: utf-8 -*-
"""
Created on Fri May  7 09:10:02 2021

@author: willi
"""

import os
import machines
from ctypes import c_char_p
import numpy as np
#from multiprocessing.pool import ThreadPool
import time
import pandas as pd
#from queue import Queue
import threading
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
pio.renderers.default='svg'
#os.chdir("D:\OneDrive - Danmarks Tekniske Universitet\OneDrive\Dokumenter\DTU\Terahertz\PaperExperiments")


class Experiment:
    '''Experiment class creating a basis for single experiments.
    Experiemnts connects to the desired machines and has basic experimental
    functions.
    '''
    
    # Creates Data structure and parameters
    data = pd.DataFrame()
    all_connected = True
    amperemeter_parameters = ['amp_int_time','amp_current_level']
    powersupply_parameters = []
    stage_parameters = []
    camera_parameters = []
    
    # Keeps track of parameters being ready
    parameters_set = False
    
    def __init__(self, cam=None, amp=None, stages=None, volt=None):
        '''Initializes each of the necessary devices
        
        Initializer takes None or True as input per default.
        Other amp instruments can be set by amp=address
        Other stages can be set by inserting serial number as c_type or list of c_type objects
        
        If connection fails, no experiments can be run. 
        In this case rerun setup after restarting kernal.
        
        '''
        
        # Checks if camera should be used
        if cam:
            
            # Gets an instance of camera
            self.cam = self.get_camera()
            
            # Checks if connection was made
            if not self.cam.connected:
                self.all_connected = False
        else:
            self.cam = None

        # Checks if ampere meter should be used
        if amp:
            
            # Checks if the input is a string
            # If input is string, then sends the string as name of instrument
            # Not necessary when using the fA Ampere meter
            if type(amp) == str:
                
                # Gets an instance of ampere meter
                self.amp = self.get_amp(serial=amp)

            else:
                # If no serial was passed, gets standard instrument
                self.amp = self.get_amp()
                
            # Checkcs if connected
            if not self.amp.connected:
                self.all_connected = False
        else:
            self.amp = None

        # Checks if Volt meter should be used
        if volt:
            
            # Gets instance of volt meter
            self.volt = self.get_volt(ip=volt)
            
            # Checks if connected
            if not self.volt.connected:
                self.all_connected = False
        else:
            self.volt = None

        # Checks if stages should be used
        if stages:
            # Checks if multiple stages should be used
            if type(stages) == list:
                # Gets multiple stages
                self.stages = [self.get_stages(serial) for serial in stages]
            else:
                # Gets single stage
                self.stages = self.get_stages(stages)
                
            # Whether the stage is connected does not seem to be supported
            # in the stage api at this moment.
            # It is therefor a good idea to check if it homes at connection.
        else:
            self.stages = None

    
    def get_camera(self):
        '''Gets an instance of Camera'''
        return machines.Camera()
    
    def get_amp(self, serial=None):
        '''Gets an instance of Amperemeter'''
        
        # Uses serial number if given, else uses class default serial
        if serial is None:
            return machines.Amperemeter()
        else:
            return machines.Amperemeter(address=serial)

    def get_stages(self, serial):
        '''Gets an instance of given stage'''
        return machines.PMT_Scanner(serialNum=serial, HomeStages=True) # Note serial should be c_char_p of binary string
    
    def get_volt(self, ip='ws://192.168.1.4:8080'):
        '''Gets an instance of PowerSource'''
        if type(ip) is str:
            return machines.PowerSupply(ip=ip)
        else:
            return machines.PowerSupply()
    
    def set_parameters(self,**kwargs):
        '''Sets the parameters necessary for an expriment and 
        send the appropriate settings to the appropriate devices'''
        
        # Puts parameters into class as parameters
        self.__dict__.update(kwargs)

        # Checks if directory is provided
        try:
            self.data_directory = self.directory
        except:
            raise Exception("No data directory inputted")
            
        print(kwargs,"\n")
            
        # Checks if the right number of channels and voltages are inserted
        if 'volt_per_chan' in kwargs.keys():
            if len(self.volt_per_chan) != len(self.used_channels):
                raise Exception("Number of voltages and number of channels should be equal was: {len(self.volt_per_chan)}, {len(self.used_channels)}")

        # Loops thought the input parameters
        for key in kwargs.keys():
            
            # Checks if key belongs to ampere meter and sends command
            if key in self.amperemeter_parameters and self.amp is not None:
                self.amp.set_parameter(key, kwargs[key])
                
            # Checks if key belongs to volt meter and sends command
            elif key in self.powersupply_parameters and self.volt is not None:
                self.volt.set_parameter(key, kwargs[key])
                
        # Notes that parameters are set
        self.parameters_set = True


    def check_readyness(self):
        '''Checks if everything is set up else rasises error'''    

        # Checks if all instruments are connected
        if not self.all_connected:
            raise Exception("Connect devices before running experiment")

        # Checks if all parameters are set
        if not self.parameters_set:
            raise Exception("Set parameters before measuring")

            
    def create_directory(self, directory, id):
        '''Creates and returns next available directory name'''
        # Set indent number
        num = 0

        # Checks if directory exists
        data_directory = directory + "\\" + id
        if os.path.exists(data_directory):

            # Loops and adds one to indent number as long as directory exists
            while os.path.exists(data_directory + f"_{num}"):
                num += 1

            # Sets data directory
            data_directory += f"_{num}"

        # Creates data directory
        os.mkdir(data_directory)
        
        return data_directory
    
    def initialize_stage_and_powersupply(self, stage=True, powersupply=True):
        '''Makes sure stage and power supply as the currect
        initial values.'''

        # Notifies user of progress
        print("Ramping up voltage")

        # Loops though all channels and turns them off
        # This prevents errors when checking if ramping
        for i,channel in enumerate(self.used_channels):
            self.volt.disable_channel(channel)

        # Loops though channels, sets new current and turns channel on
        for i,channel in enumerate(self.used_channels):
            self.volt.set_channel_voltage(channel,self.volt_per_chan[i])
            self.volt.enable_channel(channel)

        # Notifies user of progress
        print("Waiting for ramp to be complete")

        # Waits for ramping to be complete
        while self.volt.check_if_ramping(self.used_channels):
            time.sleep(0.5)

        # Notifies user of progress
        print("Making sure stages are homed")

        # Home stages before measurement
        self.stages.home_device()

        # Notifies user of progress
        print("Done ramping")

        
    def get_filtered_shot(self, shot, n):
        for i in range(n):
            maxpos = np.where(shot == np.max(shot))
            shot[maxpos[0][0],maxpos[1][0]] = 0
        return shot
            
    def get_max_exposure(self, exposure_time, max_attempts=30, filternum=2):
        '''Gets the maximum exposure without saturation for 
        the currently imposed settings'''

        # creates a flag for when shot is complete 
        # a counter for number of attemps
        # and a bogus shot in case all fails
        shot_not_taken = True
        attempts = 0
        shot = np.ones((4,4))

        # While more attempts available, get the shot
        while shot_not_taken and attempts < max_attempts:

            # Tries to get shot using sert exposure tume and 
            # marks shot as taken if complete
            try: 
                self.cam.create_snapshot(exposure=exposure_time)
                shot = self.cam.TakeSnapshot()
                shot = self.get_filtered_shot(shot, filternum)
                shot_not_taken = False
            
            # On fail, reinitializes camera and increases attempts
            except:
                self.cam = self.get_camera()
                attempts += 1
                time.sleep(0.05)
                continue
        
        # prints info on the first shot taken
        print(np.max(shot), self.cam.snapshot.exposure)

        # Loops until the pixel values are high enough or exposure time is too great
        while np.max(shot) < 240 and exposure_time < 900:
            
            # Finds step size for exposure
            if exposure_time < 50:
                exposure_time += 3
            elif exposure_time < 100:
                exposure_time += 5
            elif exposure_time < 200:
                exposure_time += 10
            elif exposure_time < 300:
                exposure_time += 20
            else:
                exposure_time += 25
                
            # flag for when shot is complete
            # and counter for attempts
            shot_not_taken = True
            attempts = 0
            
            # Loops till shot is taken or max attempts reached
            while shot_not_taken and attempts < max_attempts:
                
                # Tries to get shot using sert exposure tume and 
                # marks shot as taken if complete
                try: 
                    self.cam.create_snapshot(exposure=exposure_time)
                    shot = self.cam.TakeSnapshot()
                    shot = self.get_filtered_shot(shot, filternum)
                    shot_not_taken = False
                    
                # On fail, reinitializes camera and increases attempts
                except:
                    self.cam = self.get_camera()
                    attempts += 1
                    time.sleep(0.05)
                    continue
                # Waits the exposure time
                # time.sleep(exposure_time/1000)

        # prints and returns final values
        print(np.max(shot), exposure_time)
        return exposure_time
        
        

    def plot_data(self, xaxis="time", yaxis="current", position=0, save=False, show=True, voltage_setting=None):
        '''Creates an illustration of the measured values in the console'''

        # Checks which parameters should be used
        
        # Uses all wire grid positions
        if position == 'all':
            plot_data = self.data

        # Uses all wire grid positions for certain voltage setting
        elif voltage_setting and position == 'all':
            plot_data = self.data.loc[self.data["voltage_setting"] == voltage_setting]

        # Uses only one wire grid position and one voltage setting
        elif voltage_setting:
            plot_data = self.data.loc[self.data["voltage_setting"] == voltage_setting]
            plot_data = plot_data.loc[plot_data["position"] == position]
        else:
            
            # Uses only one wire grid position
            plot_data = self.data.loc[self.data["position"] == position]

        # Plots data based on input x and y
        fig = px.scatter(plot_data, x=xaxis, y=yaxis)

        # Checks if fig should be shown
        if show:
            fig.show()

        # Checks if fig should be saved
        if save:
            if voltage_setting:
                fig.write_html(self.data_directory + "\\" + f"{yaxis}_{voltage_setting}_{position}.html")
            else:
                fig.write_html(self.data_directory + "\\" + f"{yaxis}_{position}.html")


    def clear_data(self):
        '''Deletes all stored data from dataframe'''
        self.data = pd.DataFrame()
    
    def save_data(self, data):
        '''Saves the data to the class pandas dataframe'''
        self.data = self.data.append(data)

    def write_data(self, filename):
        '''Function for writing data of two parameters'''
        self.data.to_csv(filename, index=False)

    def read_data(self, filename):
        '''Function for reading data'''
        return pd.read_csv(filename)

    def get_phospher_current(self, position=None, voltage_setting=None, back_time=None):
        '''Runs measurements on the volt meter based on measurement time
        
        Takes position as input for data frame
        Takes voltage setting as input for data frame
        Takes back_time if background is being collected
        
        Currently outcommented lines are to test whether extra time
        at the end of each measurement is due to unnecessary sleep
        
        '''

        # Checks whether position is stated or taken directly from stage
        if position is None:
            # Gets stage position
            position = self.stages.get_position()

        # Creates dictionaries with entries for currents and voltages
        self.currents = {ch:[] for ch in self.used_channels}
        self.voltages = {ch:[] for ch in self.used_channels}

        # Holder for time stamps
        self.times = []
        
        # Notes the start time of measurement
        start = time.time()
        
        # Loops while measurement still has time left
        while time.time() - start < self.time_in_position:

            # Notes time of measurement
            col_time = time.time()
    
            # During testing sometimes returned errors
            try:
                # Loops though each channel
                for ch in self.used_channels:
                        
                    # Gets current and voltage on given channel
                    self.currents[ch].append(self.volt.get_channel_current(ch))
                    self.voltages[ch].append(self.volt.get_channel_voltage(ch))
            except:
                continue
    
            # Noptes the time of collection if no error occured during data collection
            self.times.append(col_time - start)


            # Notes the time after measurement
            cur_time = time.time()
                
            # Makes system sleep until next measurment
            try:
                time.sleep(1 / self.cur_meas_freq - (cur_time - col_time))
            except:
                continue

        try:
            # Check if voltage setting in use
            if voltage_setting is not None:
                
                # Makes list of voltages (all are the same)
                # This is for the data frame later on
                voltage = [voltage_setting for i in range(len(self.times))]
                
            # Makes list of positions (all are the same)
            # This is for data frame later on
            positions = [position for i in range(len(self.times))]
            
            # Checks if dataframe should be saved with voltages
            if voltage_setting is not None:
                
                # Creates data lists 
                # (time, position, voltage_setting, "current channels", "voltage channels")
                frame_data = [self.times, positions, voltage] + [self.currents[ch] for ch in self.used_channels] + [self.voltages[ch] for ch in self.used_channels]
                
                # Creates list of data names
                frame_names = ["time", "position","voltage_setting"] + [f"current{ch}" for ch in self.used_channels] + [f"voltage{ch}" for ch in self.used_channels]
            else:
                
                # Creates data lists
                # (time, position, "current channels", "voltage channels")
                frame_data = [self.times, positions] + [self.currents[ch] for ch in self.used_channels] + [self.voltages[ch] for ch in self.used_channels]
                
                # Creates list of data names
                frame_names = ["time", "position"] + [f"current{ch}" for ch in self.used_channels] + [f"voltage{ch}" for ch in self.used_channels]

            # Creates data frame style data lists
            frame_data = [[dat[num] for dat in frame_data] for num in range(len(frame_data[0]))]

            # Creates data frame
            frame = pd.DataFrame(frame_data, columns=frame_names)

            # Saves data to class data frame
            self.save_data(frame)

        except Exception as e:
            # In case of the unexpected prints the error
            print(e)

        # Sets phosphor current as done to single other threads to move on
        self.phos_cur_done = True
        
        # Loops though channels and plots the data collected
        # for ch in self.used_channels:
        #     self.plot_data(position=position, xaxis="time", yaxis=f"current{ch}", save=False, voltage_setting=voltage_setting)
        #     self.plot_data(position=position, xaxis="time", yaxis=f"voltage{ch}", save=False, voltage_setting=voltage_setting)

        
    def get_camera_snapshot(self, maxattempts, exposure):
        shot_not_taken = True
        attempts = 0
        shot = np.ones((2,2))
        while shot_not_taken and attempts < maxattempts:
            try: 
                self.cam.create_snapshot(exposure=exposure_time)
                shot = self.cam.TakeSnapshot()
                shot_not_taken = False
            except:
                self.cam = self.get_camera()
                attempts += 1
                time.sleep(0.05)
                continue
        if np.max(shot) == 1:
            raise Exception(f"Could not connect after {maxattempts}\n")
        else:
            return shot


    def take_images(self, position=None, directory=None, volt=None, back_time=None):
        '''Runs the thread which takes images
        
        Takes position as input for image name
        Takes voltage setting as input for image name
        Takes back_time if background is being collected
        
        '''

        # Checks if background
        if back_time is None:
            
            # Sets measure time to parameter time
            back_time = self.time_in_position
        else:
            
            # Else sets position to 90 degrees
            position = 90

        # Checks if directory provided or default should be used
        if directory is None:
            directory = self.data_directory

        # Checks if position provided
        if position is None:
            # Gets position from stage
            position = self.stages.get_position()

        # Loops though the number of images to be taken
        for img in range(int(back_time * self.image_freq)):
    
            # Notes the time of image start
            ts = time.time()
    
            # Takes image
            try:
                image = self.cam.TakeSnapshot()
            except Exception as e:
                print("Connection To Camera Lost: Attempting to reconnect\n")
                try:
                    image = self.get_camera_snapshot(self, 30, self.exposure_time)
                except:
                    time.sleep(0.25)
                    continue

            # Checks if voltage_setting is in use
            if volt is None:
    
                # Saves image
                self.cam.SaveImage(image, directory + "\\" + f"img_{img}_pos_{position}.png")
            else:
                # Saves image with voltage in name
                self.cam.SaveImage(image, directory + "\\" + f"img_{img}_pos_{position}_volt{abs(volt)}.png")
    
            # Notes end time
            tf = time.time()
    
            # Checks if too little time has passed
            if 1/self.image_freq - (tf - ts) > 0:
    
                # Sleeps until next image should be acquired
                time.sleep(1/self.image_freq - (tf - ts))

        # Marks camera as done so other threads can move on
        self.camera_done = True


    def syncronize_voltages(self):
        '''Makes sure all channels syncronize before going to zero V'''
        
        # Checks if channels 0,1,2 are used
        if 0 in self.used_channels and 1 in self.used_channels and 2 in self.used_channels:
            
            # Gets voltages
            volt0 = abs(self.volt_per_chan[self.used_channels.index(0)])
            volt1 = abs(self.volt_per_chan[self.used_channels.index(1)])
            volt2 = abs(self.volt_per_chan[self.used_channels.index(2)])

            # Checks if v1 larger than v0 or the other way around and syncronize them
            if volt1 > volt0:
                self.volt.set_channel_voltage(1,self.volt_per_chan[self.used_channels.index(0)])
            elif volt0 > volt1:
                self.volt.set_channel_voltage(0,self.volt_per_chan[self.used_channels.index(1)])

            # Waits a second for them to start ramping
            time.sleep(1)

            # Waits for channels to be done ramping
            while self.volt.check_if_ramping(self.used_channels):
                time.sleep(0.5)

            # Checks if v1 and v0 larger or smaller than v2
            if volt1 > volt2:
                
                # If v0 and v1 further from 0V then sets them to v2
                self.volt.set_channel_voltage(1,self.volt_per_chan[self.used_channels.index(2)])
                self.volt.set_channel_voltage(0,self.volt_per_chan[self.used_channels.index(2)])
                
            elif volt2 > volt1:
    
                # If v0 and v1 closer to 0V than v2 then sets v2 to thir voltage
                self.volt.set_channel_voltage(2,self.volt_per_chan[self.used_channels.index(1)])

        # If channel 0 is not included
        elif 1 in self.used_channels and 2 in self.used_channels:
            
            # Gets voltages
            volt1 = abs(self.volt_per_chan[self.used_channels.index(1)])
            volt2 = abs(self.volt_per_chan[self.used_channels.index(2)])
            
            # Checks which is largest
            # and sets the one further from 0V to the other's value
            if volt1 > volt2:
                self.volt.set_channel_voltage(1,self.volt_per_chan[self.used_channels.index(2)])
            elif volt2 > volt1:
                self.volt.set_channel_voltage(2,self.volt_per_chan[self.used_channels.index(1)])

        # Wait a second for them to start ramping
        time.sleep(1)

        # Waits for them to be done ramping
        while self.volt.check_if_ramping(self.used_channels):
            time.sleep(0.5)


    def get_power_supply_background(self, back_time=120, voltage=None):
        print("Getting Background")
        self.stages.move_device(90)
        time.sleep(15)
        self.get_phospher_current(90, back_time=back_time, voltage_setting=voltage)

    def get_ampere_meter_background(self, back_time=120):
        print("Getting Background")
        self.stages.move_device(90)
        time.sleep(25)
        self.get_current(90, back_time=back_time)

    def get_camera_background(self, back_time=120):
        print("Getting Background")
        self.stages.move_device(90)
        time.sleep(15)
        self.take_images(90, back_time=back_time)


    
class IasE(Experiment):
    '''Experiment measuring the current as a function of electric field.
    Goal is to show a nice linear FN correlation.
    
    Required settings:
        stage_positions     : list of positions in degrees
        image_freq          : number of images per second
        time_in_position    : seconds at each stage position
        amp_int_time        : time in seconds over which the amperemeter collects current
        amp_trig_time       : time in seconds between new current outputs
        directory           : folder for saving images and current measurements
        num_measurements    : the number of measurements required (overwrites the time_in_position)
    
    '''
    
    def __init__(self, cam=None, amp=None, stages=c_char_p(b'27255354')):

        # Creates experimental class and default parameters
        super().__init__(cam=cam, amp=amp, stages=stages)
        self.amp_trig_time = 0
        self.amp_int_time = 1/10
        self.time_in_position = 10
        self.directory = "D:\\OneDrive - Danmarks Tekniske Universitet\\OneDrive\\Dokumenter\\DTU\\Terahertz\\PaperExperiments\\measurements\\IasE"
        self.image_freq = 1
        self.num_measurements = 0


    def get_current(self, position=None, back_time=None):
        '''Runs current collection thread
        
        Takes position and back_time as input for dataframe
        if no position then gets live position
        if no back_time then uses parameter time
        
        '''

        # Checks if background is being collected
        if back_time is None:
            back_time = self.time_in_position
        else:
            position = 90

        # Checks if position is provided
        if position is None:
            # Gets live stage position
            position = self.stages.get_position()

        # Creates holders for current and time
        self.currents = []
        self.times = []

        # Start measurement time
        start = time.time()

        if not self.num_measurements:
            # Loops while there is more time left of experiment
            while time.time() - start < self.time_in_position:

                # Gets time of collection intialization
                col_time = time.time()

                # Gets current and appends it to currents
                try:
                    self.currents.append(self.amp.get_current())
                except:
                    continue
    
                # If ampere meter anwsered then append time of collectino froms start
                self.times.append(col_time - start)

        else:
            # Loops while there is more time left of experiment
            while len(self.times) < self.num_measurements:

                # Gets time of collection intialization
                col_time = time.time()

                # Gets current and appends it to currents
                try:
                    self.currents.append(self.amp.get_current())
                except:
                    continue

                # If ampere meter anwsered then append time of collectino froms start
                self.times.append(col_time - start)
                time.sleep(0.1)

        try:
            # Creates list of positions for data frame
            positions = [position for i in range(len(self.times))]
            
            # Creates data frame data
            frame_data = [self.times, self.currents, positions]
            
            # Formats data for dataframe
            frame_data = [[dat[num] for dat in frame_data] for num in range(len(frame_data[0]))]
            
            # Creates data frame
            frame = pd.DataFrame(frame_data, columns=["time","current","position"])
            
            # Saves data to class data frame
            self.save_data(frame)

        except Exception as e:
            # In case of unexpected error print error message
            print(e)

        # Plots the data
        # self.plot_data(position=position, save=False)

        # Marks current as done and notifies the other threads to move on
        self.current_done = True



class IasE_current(IasE):
    '''Class for measuring current from E field
    
    
    This uses the current collection super class
    
    Originally multiple different kinds of measurements using the 
    ampere meter was intended. This is why it has a super class.
    Now it is somewhat redundant. 
    
    When running current measurements therefor use this class instaed
    
    '''
    def __init__(self, stages=c_char_p(b'27255354')):
        # Creates super class
        super().__init__(amp=True, stages=stages)
    
    def run_experiment(self, id, back_time=120):
        '''Runs the experiment with the added settings'''

        # Checks if parameters and connections are made
        self.check_readyness()

        # Clears data for new run
        self.clear_data()

        self.data_directory = self.create_directory(self.directory, id)
        
        # Checks if background should be collected
        if back_time > 0:
            # Gets background
            self.get_ampere_meter_background(back_time=back_time)

        # Homes stages before measure start
        self.stages.home_device()

        # Loops through each stage position
        # In each position it starts the measurements and wait till they are done
        # Positions should start close to 0 degrees to make sure stages are always moved correctly
        for i,pos in enumerate(self.stage_positions):

            # Sets the current as not being done
            self.current_done = False

            # Notifies user of progress
            print("Moving to position: ", pos)
            print(f"Position {i+1} out of {len(self.stage_positions)}")

            # Moves to given position
            self.stages.move_device(pos)

            # Makes sure stage waits appropriately long
            if abs(pos - self.stages.get_position() > 15):
                time.sleep(10)
            else:
                time.sleep(5)

            # Creates a thread for running current data collection
            ampere_thread = threading.Thread(target = self.get_current,
                                             args = (pos,))

            # Notifies user of progress
            print("Starting ampere thread")

            # Starts the data collection
            ampere_thread.start()

            # Waits till measure time is exceeded and current thread is done
            while not self.current_done:
                time.sleep(1)

            # Notifies user of progress
            print("Position: ",pos, " is done\n")

            # Writes the data to file
            self.write_data(self.data_directory + "\\" + "current.csv")
            
        # Waits a second to make sure all is complete
        time.sleep(1.5)

        # Notifies user of progress
        print("All positions completed, measurement done!")



class PhosphorCurrent(Experiment):
    '''Experiment measuring the current on the phosphor screen.
    Goal is to create a histogram of current at a given electric field, 
    maybe dependent on incomming electric field and external bias potential.
    
    Settings required:
        
        time_in_position       : Seconds in which it should stay in each position
        stage_positions        : Positions in degrees
        directory              : Directory for saving the output csv file
        cur_meas_freq          : The number of current measurements per second
        used_channels          : The channels which should be enabled during experiment
        volt_per_chan          : The voltages of each channel
        
        if camera is enabled use:
            image_freq         : Images per second
        
        (integration time of the device is currently unknown)
    
    '''
    
    def __init__(self, cam=False, switch_off_after_meas=True, cam_dir=None, stage=c_char_p(b'27255354'), volt=True):

        # Initialize Experiment class with necessary instruments
        # Stages is per default the stage currently set to the wire grid
        super().__init__(volt=volt, stages=stage, cam=cam)

        # Sets directory of camera
        self.cam_dir = cam_dir

        # Sets whether power supply should be switched off after use
        self.switch_off_after_meas = switch_off_after_meas

        # Checks whether camera should be used
        self.use_camera = cam

        # Sets general default parameters
        self.time_in_position = 10
        self.directory = "D:\\OneDrive - Danmarks Tekniske Universitet\\OneDrive\\Dokumenter\\DTU\\Terahertz\\PaperExperiments\\measurements\\PhosphorCurrent"
        self.stage_positions = [0]
        self.cur_meas_freq = 5
        self.used_channels = [1,2,3]
        self.image_freq = 1
        self.num_measurements = 0


    def run_experiment(self, id, back_time=0, start_exposure=20):
        '''Runs the expreiment'''

        # Checks if connections and parameters are set        
        self.check_readyness()
        self.clear_data()

        # Gets and creates directory for current data
        self.data_directory = self.create_directory(self.directory, id)
        
        # Checks if camera should be used
        if self.use_camera: 

            # Checks whether camera directory has been set
            if self.cam_dir is None:
                # Uses data directory with _images if no directory is set
                self.cam_dir = self.data_directory + "_images"

            self.camera_directory = self.create_directory(self.cam_dir, id)
            exposure_frame = pd.DataFrame(columns=["voltage","angle","exposure"])

        # Sets initial settings for power supply and stage
        self.initialize_stage_and_powersupply()

        time.sleep(4)
        
        if self.use_camera:
                
            print("Getting right exposure")
                
            # Gets the maximum exposure without saturation
            self.exposure_time = self.get_max_exposure(start_exposure)
            self.cam.create_snapshot(exposure=self.exposure_time)

            exposure_frame = exposure_frame.append(pd.DataFrame([[0,self.exposure_time]],columns=["angle","exposure"]), ignore_index=True)
            exposure_frame.to_csv(self.camera_directory + "\\exposure.csv", index=False)
            print("Final Exposure : ", self.exposure_time)
        
        
        # Loops though all stage positions set
        for i,pos in enumerate(self.stage_positions):

            # Sets status of current thread to not done
            self.phos_cur_done = False

            # Notifies user of progress
            print("Moving to position: ", pos)
            print(f"Position {i+1} out of {len(self.stage_positions)}")

            # Moves stages to first position
            self.stages.move_device(pos)

            # Makes sure stage waits appropriately long
            if abs(pos - self.stages.get_position() > 15):
                time.sleep(10)
            else:
                time.sleep(5)

            # Creates current thread
            phos_cur_thread = threading.Thread(target = self.get_phospher_current,
                                             args = (pos,))
            # Checks if camera should be used
            if self.use_camera:
                
                # Sets status of camera to not done and creates camera thread
                self.camera_done = False
                camera_thread = threading.Thread(target = self.take_images,
                                                 args = (pos,self.camera_directory,))
                
                # Starts camera thread                
                print("Starting Camera thread")
                camera_thread.start()

            # Starts current thread
            print("Starting Volt/Current thread")
            phos_cur_thread.start()

            # Check if camera should be in use and sets it to done if not in use
            if not self.use_camera:
                self.camera_done = True

            # Notes measure start time
            t = time.time()

            # Waits for measure time to be exceeded and thraeds to be done
            while (time.time() - t < self.time_in_position) or (not self.phos_cur_done) or (not self.camera_done):
                time.sleep(1)

            # Notifies user of progress
            print("Position: ",pos, " is done\n")
            
            # Writes data to csv file
            self.write_data(self.data_directory + "\\" + "voltage_and_currents.csv")
            

        # Checks if power supply should be switched off after measurement
        if self.switch_off_after_meas:
            # Synconzes voltages
            self.syncronize_voltages()

            # Loops though channels used
            for channel in self.used_channels:
                # Shut off channel
                self.volt.disable_channel(channel)

        # Wait a second
        time.sleep(1.5)

        # Notifies user of progress
        print("All positions completed, measurement done!")


class EmissionEnergy(Experiment):
    '''Experiment measuring the current on the phosphor screen.
    Goal is to using a combination of images and currents to estimate 
    electron energies when leaving the metasurface.
    
    Settings required:
        
        time_in_position       : Seconds in which it should stay in each position
        stage_positions        : Positions in degrees
        directory              : Directory for saving the output csv file
        cur_meas_freq          : The number of current measurements per second
        used_channels          : The channels which should be enabled during experiment
        volt_per_chan          : The voltages of each channel

        if camera is enabled use:
            image_freq         : Images per second
        
        (integration time of the device is currently unknown)
    
    '''


    def __init__(self, cam=False, volt=True, stages=c_char_p(b'27255354'), switch_off_after_meas=True, cam_dir=None):

        # Creates Experiment object and connects to instruments
        super().__init__(cam=cam, volt=volt, stages=stages)

        # Sets camera directory
        self.cam_dir = cam_dir

        # Sets whether power supply should be switched off after use
        self.switch_off_after_meas = switch_off_after_meas

        # Checks if camera should be used
        self.use_camera = cam

        # Sets default parameters
        self.time_in_position = 10
        self.directory = "D:\\OneDrive - Danmarks Tekniske Universitet\\OneDrive\\Dokumenter\\DTU\\Terahertz\\PaperExperiments\\measurements\\EmissionEnergy"
        self.stage_positions = [0]
        self.channel_voltages = [-1500]
        self.cur_meas_freq = 5
        self.used_channels = [1,2,3]
        self.image_freq = 1
        
    def run_experiment(self, id, back_time=60, variable_channel=None, start_exposure=20, filternum=2):
        '''Runs EmissionEnergy Experiment'''

        # Checks whether parameters and connections have been made
        self.check_readyness()

        # Clears class dataframe
        self.clear_data()

        # Gets and creates directory for current data
        self.data_directory = self.create_directory(self.directory, id)
        
        # Checks if camera should be used
        if self.use_camera: 

            # Checks whether camera directory has been set
            if self.cam_dir is None:
                # Uses data directory with _images if no directory is set
                self.cam_dir = self.data_directory + "_images"

            self.camera_directory = self.create_directory(self.cam_dir, id)
            exposure_frame = pd.DataFrame(columns=["angle","exposure"])

        # Sets initial settings for power supply and stage
        self.initialize_stage_and_powersupply()

        # Checks if specific channel has been set for varying
        if variable_channel is None:
            # Uses the first of the used channels if none is set
            variable_channel = self.used_channels[0]
        if type(variable_channel) is not list:
            variable_channel = list(variable_channel)

        # Loops though stage positions
        for i,pos in enumerate(self.stage_positions):

            # marks that current thread is not done
            self.phos_cur_done = False

            print(f"Position {i+1} out of {len(self.stage_positions)}")

            # Moves stage to next position
            self.stages.move_device(pos)

            # Makes sure to wait for stage appropriately
            if abs(pos - self.stages.get_position() > 15):
                time.sleep(10)
            else:
                time.sleep(5)
            
            if self.use_camera:
                
                self.next_cam_dir = self.camera_directory + f"\\{pos}"
                os.mkdir(self.next_cam_dir)
                
                print("Getting right exposure")
                
                for chan in variable_channel:
                    self.volt.set_channel_voltage(chan, min(self.channel_voltages))

                while self.volt.check_if_ramping(variable_channel):
                    time.sleep(0.5)

                # Gets the maximum exposure without saturation
                self.exposure_time = self.get_max_exposure(start_exposure, filternum=filternum)
                self.cam.create_snapshot(exposure=self.exposure_time)

                exposure_frame = exposure_frame.append(pd.DataFrame([[pos,self.exposure_time]],columns=["angle","exposure"]), ignore_index=True)
                exposure_frame.to_csv(self.camera_directory + "\\exposure.csv", index=False)
                print("Final Exposure : ", self.exposure_time)

            # Loops through voltages
            for j,v in enumerate(self.channel_voltages):

                # Sets voltage of variable channel to next voltage
                for chan in variable_channel:
                    self.volt.set_channel_voltage(chan, v)

                # Notifies user of progress
                print(f"Voltage : {v}, {j+1} out of {len(self.channel_voltages)}")
                print(f"Position : {np.round(pos,3)}, {i+1} out of {len(self.stage_positions)}")

                time.sleep(2)

                # Wait for ramping to be complete
                while self.volt.check_if_ramping(variable_channel):
                    time.sleep(0.5)

                # Checks if background should be taken
                if back_time > 0 and i == 0:
                    self.get_power_supply_background(back_time=back_time, voltage=v)

                # Creates current thread
                phos_cur_thread = threading.Thread(target = self.get_phospher_current,
                                                 args = (pos,v,))

                # Checks if camera should be used
                if self.use_camera:
                    self.camera_done = False
                    camera_thread = threading.Thread(target = self.take_images,
                                                     args = (pos,self.next_cam_dir,v))

                    # Starts the image thread
                    print("Starting Camera thread")
                    camera_thread.start()

                # Start current thread
                print("Starting Volt/Current thread")
                phos_cur_thread.start()

                # Checks if camera in use else sets it to done
                if not self.use_camera:
                    self.camera_done = True

                # Notes start time of measurement
                t = time.time()

                # Waits for measure time to be exceeded and thread to be done
                while (time.time() - t < self.time_in_position) or (not self.phos_cur_done) or (not self.camera_done):
                    time.sleep(1)

                # Wirtes data to csv file
                self.write_data(self.data_directory + "\\" + f"{pos}.csv")
                self.clear_data()

                # Notifies user of progress
                print("Voltage: ",v, " is done\n")

        # Checks if power supply should be switched off
        if self.switch_off_after_meas:
            
            # Synconizes voltages
            self.syncronize_voltages()

            # Loops though channels
            for channel in self.used_channels:
                # Turns channels off
                self.volt.disable_channel(channel)

        # Waits a second
        time.sleep(1.5)

        # Notify user of progress
        print("All positions completed, measurement done!")


class ExtractionThreshold(Experiment):
    '''Experiment measuring the current on the phosphor screen,
    while doing images of the screen.
    Goal is to using a combination of images and currents to estimate 
    the highest negative extraction with which electrons reach the mcp.
    
    Settings required:
        
        time_in_position       : Seconds in which it should stay in each position
        stage_positions        : Positions in degrees
        directory              : Directory for saving the output csv file
        cur_meas_freq          : The number of current measurements per second
        used_channels          : The channels which should be enabled during experiment
        volt_per_chan          : The voltages of each channel
        channel_voltages       : The voltage settings for the variable channel

        if camera is enabled use:
            image_freq         : Images per second
        
    '''


    def __init__(self, cam=True, volt=True, stages=c_char_p(b'27255354'), switch_off_after_meas=True, cam_dir=None):

        # Creates Experiment object and connects to instruments
        super().__init__(cam=cam, volt=volt, stages=stages)

        # Sets camera directory
        self.cam_dir = cam_dir

        # Sets whether power supply should be switched off after use
        self.switch_off_after_meas = switch_off_after_meas

        # Checks if camera should be used
        self.use_camera = cam

        # Sets default parameters
        self.time_in_position = 10
        self.directory = "D:\\OneDrive - Danmarks Tekniske Universitet\\OneDrive\\Dokumenter\\DTU\\Terahertz\\PaperExperiments\\measurements\\EmissionEnergy"
        self.stage_positions = [0]
        self.channel_voltages = [-1500]
        self.cur_meas_freq = 5
        self.used_channels = [1,2,3]
        self.image_freq = 1
        
    def run_experiment(self, id, back_time=0, variable_channel=None, start_exposure=20, filternum=2):
        '''Runs EmissionEnergy Experiment'''

        # Checks whether parameters and connections have been made
        self.check_readyness()

        # Clears class dataframe
        self.clear_data()

        # Gets and creates directory for current data
        self.data_directory = self.create_directory(self.directory, id)
        
        # Checks if camera should be used
        if self.use_camera: 

            # Checks whether camera directory has been set
            if self.cam_dir is None:
                # Uses data directory with _images if no directory is set
                self.cam_dir = self.data_directory + "_images"

            self.camera_directory = self.create_directory(self.cam_dir, id)
            exposure_frame = pd.DataFrame(columns=["voltage","angle","exposure"])

        # Sets initial settings for power supply and stage
        self.initialize_stage_and_powersupply()

        # Checks if specific channel has been set for varying
        if variable_channel is None:
            # Uses the first of the used channels if none is set
            variable_channel = self.used_channels[0]
        if type(variable_channel) is not list:
            variable_channel = list(variable_channel)

        # Loops though stage positions
        for i,pos in enumerate(self.stage_positions):

            # marks that current thread is not done
            self.phos_cur_done = False

            print(f"Position {i+1} out of {len(self.stage_positions)}")

            # Moves stage to next position
            self.stages.move_device(pos)

            # Makes sure to wait for stage appropriately
            if abs(pos - self.stages.get_position() > 15):
                time.sleep(10)
            else:
                time.sleep(5)
            
            if self.use_camera:                
                self.next_cam_dir = self.camera_directory + f"\\{pos}"
                os.mkdir(self.next_cam_dir)

            # Loops through voltages
            for j,v in enumerate(self.channel_voltages):

                # Sets voltage of variable channel to next voltage
                for chan in variable_channel:
                    self.volt.set_channel_voltage(chan, v)

                # Notifies user of progress
                print(f"Voltage : {v}, {j+1} out of {len(self.channel_voltages)}")
                print(f"Position : {np.round(pos,3)}, {i+1} out of {len(self.stage_positions)}")

                time.sleep(2)

                # Moves stage to next position
                self.stages.move_device(pos)
                # Makes sure to wait for stage appropriately
                time.sleep(10)
                
                # Wait for ramping to be complete
                while self.volt.check_if_ramping(variable_channel):
                    time.sleep(0.5)

                print("Getting right exposure")
                
                # Gets the maximum exposure without saturation
                self.exposure_time = self.get_max_exposure(start_exposure, filternum=filternum)
                self.cam.create_snapshot(exposure=self.exposure_time)
                
                exposure_frame = exposure_frame.append(pd.DataFrame([[v,pos,self.exposure_time]],columns=["voltage","angle","exposure"]), ignore_index=True)
                exposure_frame.to_csv(self.camera_directory + "\\exposure.csv", index=False)
                print("Final Exposure : ", self.exposure_time)
                    
                for bb,p in enumerate([pos,90]):
                    
                    # Moves stage to next position
                    self.stages.move_device(p)
                    # Makes sure to wait for stage appropriately
                    time.sleep(10)

                    # Creates current thread
                    phos_cur_thread = threading.Thread(target = self.get_phospher_current,
                                                     args = (p,v,))

                    # Checks if camera should be used
                    if self.use_camera:
                        self.camera_done = False
                        camera_thread = threading.Thread(target = self.take_images,
                                                         args = (p,self.next_cam_dir,v))

                        # Starts the image thread
                        print("Starting Camera thread")
                        camera_thread.start()

                    # Start current thread
                    print("Starting Volt/Current thread")
                    phos_cur_thread.start()

                    # Checks if camera in use else sets it to done
                    if not self.use_camera:
                        self.camera_done = True

                    # Notes start time of measurement
                    t = time.time()

                    # Waits for measure time to be exceeded and thread to be done
                    while (time.time() - t < self.time_in_position) or (not self.phos_cur_done) or (not self.camera_done):
                        time.sleep(1)

                    # Wirtes data to csv file
                    self.write_data(self.data_directory + "\\" + f"{p}_{v}.csv")
                    self.clear_data()

                    # Notifies user of progress
                    print("Voltage: ",v, " is done\n")

        # Checks if power supply should be switched off
        if self.switch_off_after_meas:
            
            # Synconizes voltages
            self.syncronize_voltages()

            # Loops though channels
            for channel in self.used_channels:
                # Turns channels off
                self.volt.disable_channel(channel)

        # Waits a second
        time.sleep(1.5)

        # Notify user of progress
        print("All positions completed, measurement done!")

        

class CurrentMonitor(Experiment):
    '''Class enabling the volt meter to be used as a current monitor
    for optimizing terahertz setup
    
    '''

    def __init__(self, volt, show_seconds=10):

        # Creates Experiment class and connects to power supply
        super().__init__(volt=volt)

        # Sets neccessary parameters
        self.set_parameters(used_channels=[1,2,3],
                            volt_per_chan=[-2000,-1500,2000],
                            directory="DontNeedOne")
        self.run = True
        
        self.show_seconds = int(show_seconds)
        
    def start(self, show_seconds=None):
        '''Starts the monitor'''
        
        # Checks if ready
        self.check_readyness()
        
        if show_seconds is None:
            show_seconds = self.show_seconds
        else:
            show_seconds = int(show_seconds)

        print("Ramping up voltage")

        # Loops though channels and makes sure they are turned on
        for i,channel in enumerate(self.used_channels):
            self.volt.disable_channel(channel)
        for i,channel in enumerate(self.used_channels):
            self.volt.set_channel_voltage(channel,self.volt_per_chan[i])
            self.volt.enable_channel(channel)

        # Waits for ramping to be done
        print("Waiting for ramp to be complete")
        while self.volt.check_if_ramping(self.used_channels):
            time.sleep(0.5)

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        plt.ion()

        self.fig.show()
        self.fig.canvas.draw()
        self.time = [0]
        self.current = [0]
        self.point_limit = show_seconds*5
        
        # Loops while running
        while self.run:
            
            curr = self.volt.get_channel_current(3)
            try:
                curr = float(curr)
            except:
                curr = 0
            self.current.append(curr)
            self.time.append(self.time[-1]+1)

            self.ax.clear()
            if len(self.current) >= self.point_limit:
                self.ax.plot(self.time[-self.point_limit:], self.current[-self.point_limit:])
            else:
                self.ax.plot(self.time, self.current)
            self.fig.canvas.draw()
                
            # Waits a second
            plt.pause(0.2)

if __name__ == '__main__':

#    positions = np.arccos(np.sqrt(np.array([1 - i*0.02 for i in range(3)]))) * 180 / np.pi
    positions = np.arccos(np.sqrt(np.array([1 - i*0.1 for i in range(1)]))) * 180 / np.pi
#    positions = np.arccos(np.sqrt(np.array([1 - i*0.1 for i in range(2)]))) * 180 / np.pi
    # TODO something is wrong in regards to the time in position. Only stays of about 75% of duration...

#    cur = IasE_current()
#    cur.set_parameters(amp_current_level=1e-18,
#                       stage_positions=positions,
#                       amp_int_time=0.5,
#                       time_in_position=60,
#                       directory="D:\\OneDrive - Danmarks Tekniske Universitet\\OneDrive\\Dokumenter\\DTU\\Terahertz\\PaperExperiments\\measurements\\IasE")
#    cur.run_experiment("dark_current_t1", back_time=0)


    ### TODO
    ##
    ## Camera issues
    ##  - Attribute failure
    ##
    ## Custom channel sweep
    ## Save data after each sweep (all experiments)
    ## 
    ## Voltage issues
    ##  - Phosphor current time is too long
    ##
    ##
    ##
    ##


    # TODO camera should save values to external disc when inserted
#    
    positions = [0]
    phos = PhosphorCurrent(cam=False, volt='ws://192.168.1.2:8080')
    phos.set_parameters(time_in_position=60,
                        directory="D:\\OneDrive - Danmarks Tekniske Universitet\\OneDrive\\Dokumenter\\DTU\\Terahertz\\PaperExperiments\\measurements\\PhosphorCurrent",
                        stage_positions=positions,
                        cur_meas_freq=50,
                        used_channels=[1,2,3],
                        volt_per_chan=[-2000,-1500,2000],
                        image_freq=1,
                        switch_off_after_meas=True,
                        cam_dir="E:\\William_data")
    phos.run_experiment("1000GHz_020721",back_time=30)

#    positions = np.arccos(np.sqrt(np.array([0 + 0.68 + i*0.08 for i in range(5)]))) * 180 / np.pi    
    # positions = [0]
    # channel_voltages=[-2500 + 100 * i for i in range(11)]
    # emission = EmissionEnergy(cam=False, volt='ws://192.168.1.2:8080') # Set cam=True to enable camera
    # emission.set_parameters(time_in_position=45,
    #                     directory="D:\\OneDrive - Danmarks Tekniske Universitet\\OneDrive\\Dokumenter\\DTU\\Terahertz\\PaperExperiments\\measurements\\EmissionEnergy",
    #                     stage_positions=positions,
    #                     cur_meas_freq=50,
    #                     used_channels=[0,1,2,3],
    #                     volt_per_chan=[-2000,-2000,-1500,2000],
    #                     channel_voltages=channel_voltages,
    #                     image_freq=0.05,
    #                     switch_off_after_meas=False,
    #                     cam_dir="D:\\OneDrive - Danmarks Tekniske Universitet\\OneDrive\\Dokumenter\\DTU\\Terahertz\\PaperExperiments\\measurements\\EmissionEnergy\\Cam_dir")
    # emission.run_experiment("code_test", back_time=0)
#    
#    monitor = CurrentMonitor()
#    monitor.start()