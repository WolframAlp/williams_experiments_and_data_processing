# -*- coding: utf-8 -*-
"""
Created on Fri May  7 09:12:14 2021

@author: willi
"""

# Imports dependecies
import clientCommon
import os
import sys

# c_path = os.getcwd()
# path = os.path.abspath(__file__)[:-25]
# if not path in sys.path:
    # sys.path.append(path)
# print(os.path.abspath(__file__)[:-25])
import icsClientPython_3 as icsClient
# os.chdir(c_path)

class PowerSupply:
    '''Class for interacting with power supply'''

    def __init__(self, ip=None):

        # Sets necessary parameters for connection        
        connectionName = "script_localhost"
        connectionLogin = "will"
        connectionPassword = "password"

        # Checks if ip is inserted
        if ip is not None:
            connectionParameters = ip
        else:
            # Default ip
            # Might change when moving the device due to router asignment
            connectionParameters = "ws://192.168.1.2:8080"

        # Make connection        
        self.connection = clientCommon.establishConnection(icsClient, connectionName, connectionParameters, connectionLogin, connectionPassword)

        # Checks if connected
        if self.connection == 0:
            self.connected = False
        else:
            self.connected = True

        # If not connected returns
        if not self.connected:
            print("Failed to connect => Exit.")
            return

        # Gets modules from connection
        self.modules = self.connection.getModules()

        # Extracts channels from modules
        self.channels = self.modules[0].channels

    def set_parameter(self, param, val):
        '''Functionf for setting specific parameters'''
        pass

    def get_channel_voltage(self, channel):
        '''Gets voltage of channel'''
        return self.channels[channel].getStatusVoltageMeasure()

    def get_channel_current(self, channel):
        '''Gets current of channel'''
        return self.channels[channel].getStatusCurrentMeasure()

    def set_channel_voltage(self, channel, voltage):
        '''Set voltage of channel'''
        return self.channels[channel].setControlVoltageSet(voltage)

    def enable_channel(self, channel):
        '''Turn channel on'''
        return self.channels[channel].setControlOn(1)

    def disable_channel(self, channel):
        '''Turn channel off'''
        return self.channels[channel].setControlOn(0)

    def isRamping(self, channel):
        '''Check if channel is ramping'''
        return self.channels[channel].getStatusVoltageRamping()

    def check_if_ramping(self, channels):
        '''Checks if any channel is ramping'''
        try:
            return (sum([self.isRamping(chan) for chan in channels]) > 0)
        except:
            return False

if __name__ == '__main__':
    power = PowerSupply()