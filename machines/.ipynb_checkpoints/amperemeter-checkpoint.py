# -*- coding: utf-8 -*-
"""
Created on Fri May  7 09:12:44 2021

@author: willi
"""

# Import package for camera communication
import pyvisa as visa

# Set address of ampere meter
VISA_ADDRESS = 'USB0::0x0957::0xD318::MY54320345::0::INSTR'


class Amperemeter:
    
    '''
    Class for controlling Keysight B2981A femtoamperemeter
    
    Procedure:
        Initialize class with the adress of the amperemeter
        Set trigger settings
        Start measurement (Arm Trigger and trigger)
    
    '''
    
    def __init__(self, address='USB0::0x0957::0xD318::MY54320345::0::INSTR'):

        # Gets available resources
        resources = visa.ResourceManager()
        try:
            # Gets specific resource from resources
            self.inst = resources.open_resource(address)
            self.connected = True
        except:
            self.connected = False
            print("Amperemeter connection error")
            return

        try:
            # Checks if connected by sending and receiving model number
            self.write("*IDN?")
            model = self.read()
            print("Device Connected:", model)
            self.connected = True
        except:
            print("Connection Failed")
            self.connected = False

    def set_parameter(self, param, val):
        '''Set input parameter and value'''
        if param == 'amp_int_time':
            self.set_integration_time(val)
        elif param == 'amp_current_level':
            self.set_current_level(val)

    def set_upper_current_range(self, r):
        '''Sets the upper current range'''
        self.write(f"SENSe:CURRent:DC:RANGe:AUTO:ULIMit {r}")
        print("Upper limit set to: ", self.query(":SENSe:CURRent:DC:RANGe:AUTO:ULIMit?"))

    def set_lower_current_range(self, r):
        '''Sets the lower current range'''
        self.write(f"SENSe:CURRent:DC:RANGe:AUTO:LLIMit {r}")
        print("Lower limit set to: ", self.query(":SENSe:CURRent:DC:RANGe:AUTO:LLIMit?"))

    def switch_to_auto_current_level(self):
        '''Use the automated current level setting'''
        return self.write("SENS:CURR:DC:RANG:AUTO OFF")

    def set_current_level(self, r):
        '''Set a current level'''
        self.write("SENS:CURR:DC:RANG:AUTO OFF")
        return self.write("SENS:CURR:DC:RANG {r}")
        
    def set_integration_time(self, t):
        '''Sets integration time'''
        self.write(f"SENSe:CURRent:DC:APERture {t}")
        print("Integration time set to: ", self.query(":SENSe:CURRent:DC:APERture?"))
    
    def enable_auto_integration(self):
        '''Uses automatic integration time'''
        self.write("SENSe:CURRent:DC:APERture:AUTO 1")
    
    def disable_auto_integration(self):
        '''Goes back to manual integration time'''
        self.write("SENSe:CURRent:DC:APERture:AUTO 0")

    def set_sample_count(self, n):
        '''Sets number of samples in a collection'''
        return self.write("sample:count {n}")

    def start_measure(self):
        '''Starts a measurement'''
        self.write("initiate")
        self.write("TRIG:ARM")
        self.write("*TRG")
        return "Triggered"

    def end_measure(self):
        '''Ends measurement'''
        return self.write(":ABOR")

    def get_data(self):
        '''Gets all data'''
        data = self.query("trace:data?").split(',')
        return data[0]
    
    def get_current(self):
        '''Gets only current'''
        return float(self.query(":meas:curr?").strip("\n"))

    def reset_settings(self):
        '''Reset settings'''
        return self.write("*RST")

    def read(self):
        '''Read from device
        Does not return anything if nothing to read
        '''
        if self.connected:
            try:
                mess = self.inst.read_raw().decode().rstrip()
            except:
                mess = "Nothing to read"
        else:
            print("Device Not Connected: Could not read")
            return ''
        print(mess)
        return mess
    
    def retire_trigger(self):
        '''Remove current trigger'''
        return self.write("TRIG:ABOR")

    def write(self, cmd):
        '''Sends message to device'''
        if self.connected:
            return self.inst.write(cmd)
        else:
            print(f"Device Not Connected: Could not write {cmd}")

    def query(self, cmd):
        '''Sends and receives meassage to/from device'''
        if self.connected:
            return self.inst.query(cmd)
        else:
            print(f"Device Not Connected: Could not query {cmd}")

    def assert_trigger(self):
        '''Starts trigger'''
        if self.connected:
            return self.inst.assert_trigger()
        else:
            print("Device Not Connected: Could not assert trigger")

    def set_mode_to_single_shot(self):
        '''Use single shots'''
        return self.write("ONES")


if __name__ == '__main__':
    
    amp = Amperemeter()
    interval_in_ms = 500
    number_of_readings = 10
    
    amp.set_mode_to_single_shot()
    amp.set_sample_count(number_of_readings)
    amp.start_measure()
    
    import time
    for i in range(4):
        time.sleep(3)
        print(amp.get_current())
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    