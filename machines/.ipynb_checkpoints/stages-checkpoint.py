
# Import dependencies
import os
import time
from ctypes import c_char_p, cdll, c_int, c_ulong, c_ushort, c_double, byref

# Sets path to thorlab dll
os.chdir(r"C:\Program Files\Thorlabs\Kinesis")
lib = cdll.LoadLibrary("Thorlabs.MotionControl.KCube.DCServo.dll")

#Build device list
lib.TLI_BuildDeviceList()

#set up serial number variable
serial_rot_1 = c_char_p(b'27005004')
serial_rot_2 = c_char_p(b'27255354') #83827937
serial_z = c_char_p(b'27004364')
serial_x = c_char_p(b'27255894')
serial_y = c_char_p(b'27256338')

class PMT_Scanner:
    '''Class for interacting with a stage'''
    
    def __init__(self,
                 filename=None,
                 path=None,
                 save_file=False,
                 HomeStages=False,
                 serialNum=c_char_p(b'27255354')):

        # Sets variouse parameters
        self.serialNum = serialNum
        self.moveTimeout = 25
        self.__path = path
        self.save_file = save_file
        self.filename = filename

        # Setting up stages
        print('Initializing stages')
        self.initialize_device(serialNum)

        # Adjusting stage parameters
        print('Adjusting stage settings')
        self.set_up_device(serialNum, self.settings_devices(serialNum)[0], self.settings_devices(serialNum)[1], self.settings_devices(serialNum)[2])

        # Homing stages
        if HomeStages:
            print('Homing stages')
            self.home_device(serialNum)

    def settings_devices(self, serialNumber):
        '''Sets stage parameters based on rotation of xyz'''

        # Sets xyz parameters
        if serialNumber == serial_x or serial_y or serial_z:
            # constants for the Z812
            stepsPerRev = 512
            gearBoxRatio = 67
            pitch = 1

        # Sets roation parameters
        elif serialNumber == self.serialNum or serial_rot_2:
            # constants for the PRM1-Z8
            stepsPerRev = 512
            gearBoxRatio = 67
            pitch = 17.87

        # Returns parameters
        return stepsPerRev, gearBoxRatio, pitch


    def initialize_device(self, serialNumber):
        '''Connects to device'''

        # Connects to device
        lib.CC_Open(serialNumber)
        lib.CC_StartPolling(serialNumber, c_int(200))

        # Waits a seconds and then clears meassage queue
        time.sleep(3)
        lib.CC_ClearMessageQueue(serialNumber)


    def clean_up_device(self, serialNumber):
        '''Removes previouse commands and settings'''
        lib.CC_ClearMessageQueue(serialNumber)
        lib.CC_StopPolling(serialNumber)
        lib.CC_Close(serialNumber)

    def clean_up_all(self):
        '''Cleans xyz stages'''
        self.clean_up_device(serial_x)
        self.clean_up_device(serial_y)
        self.clean_up_device(serial_z)

    def home_device(self, serialNumber=None):
        '''Homes device'''

        # Checks if serial number is provided
        if serialNumber is None:
            serialNumber = self.serialNum

        # Notes start time
        homeStartTime = time.time()

        # Sends home command
        lib.CC_Home(serialNumber)

        # Define meassage types
        self.messageType = c_ushort()
        self.messageID = c_ushort()
        self.messageData = c_ulong()

        # Waits for homed to be complete
        homed = False
        while (homed == False):
            lib.CC_GetNextMessage(serialNumber, byref(self.messageType), byref(self.messageID), byref(self.messageData))
            if ((self.messageID.value == 0 and self.messageType.value == 2) or (time.time() - homeStartTime) > 2*self.moveTimeout):
                homed = True

        # Notify user of progress
        print("Homed")

        # Clears stage messages
        lib.CC_ClearMessageQueue(serialNumber)
        return True


    def set_up_device(self, serialNumber, stepsPerRev, gearBoxRatio, pitch):
        '''Set up device based on gear parameters'''
        # Set up to convert physical units to units on the device
        lib.CC_SetMotorParamsExt(serialNumber, c_double(stepsPerRev), c_double(gearBoxRatio), c_double(pitch))
        deviceUnit = c_int()
        self.deviceUnit = deviceUnit

    def move_device(self, position, serialNumber=None, unit='deg'):
        '''Moves device to given position'''

        # Checks if serial number is provided
        if serialNumber is None:
            serialNumber = self.serialNum

        # Checks unit
        if unit=='deg':
            position /= 17.87

        # Gets the unit from the device
        deviceUnit = c_int()
        realUnit = c_double(position)
        lib.CC_GetDeviceUnitFromRealValue(serialNumber, realUnit, byref(deviceUnit), 0)

        # Note move start and start movement
        moveStartTime = time.time()
        lib.CC_MoveToPosition(serialNumber, deviceUnit)

        # Wait for move to be complete
        moved = False
        messageType = c_ushort()
        messageID = c_ushort()
        messageData = c_ulong()
        while (moved == False):
            lib.CC_GetNextMessage(serialNumber, byref(messageType), byref(messageID), byref(messageData))

            if ((messageID.value == 1 and messageType.value == 2) or (time.time() - moveStartTime) > self.moveTimeout):
                moved = True
        return True

    def get_position(self):
        '''Gets position of stage'''
        return lib.CC_GetPosition(self.serialNum)
