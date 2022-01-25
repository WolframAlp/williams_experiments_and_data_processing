import sys
import importlib
import time
import traceback
import os

def checkIsWindowsPlatform():
    if sys.platform.startswith('win'):
        return True
    elif sys.platform.startswith('linux'):
        return False
    elif sys.platform.startswith('darwin'):
        return False
    return None

def checkIsMacPlatform():
    if sys.platform.startswith('darwin'):
        return True
    elif sys.platform.startswith('linux'):
        return False
    elif sys.platform.startswith('win'):
        return False
    return None

def checkIsLinuxPlatform():
    if checkIsMacPlatform() :
         return False
    if checkIsWindowsPlatform() :
         return False
    return True

def checkIsArmPlatform():
    if os.uname()[4] == "armv7l"  :
        return True
    return False

def checkIs64BitPlatform():
    if sys.maxsize > 2**32 :
        return True
    return False

def getScriptName():
    script_path = sys.argv[0]
    script_path = os.path.abspath(script_path)
    script_name = os.path.basename(script_path)
    return script_name

def getSessionIdSupervisor():
    session_id_supervisor = ""
    for argvIndex in range(0, len(sys.argv)):
        if argvIndex + 1 < len(sys.argv):
            if sys.argv[argvIndex] == "-i":
                session_id_supervisor = sys.argv[argvIndex + 1]
    return session_id_supervisor

def getConnectionParameters():
    connection_parameters = ""
    for argvIndex in range(0, len(sys.argv)):
        if argvIndex + 1 < len(sys.argv):
            if sys.argv[argvIndex] == "-c":
                connection_parameters = sys.argv[argvIndex + 1]
    return connection_parameters

## this function imports the icsClient Python module
##      the path and module name are defined here if they aren't given above
def importFromPath(module_path, module_name):

    script_path = os.path.dirname(sys.argv[0])
    script_path = os.path.abspath(script_path)

    ## module_path defines the location of the python module
    if module_path == "" :
        if checkIsWindowsPlatform():
            module_path = 'D:\\work\\iseg-build-pro\\build32\\isegcontrol_DEV_171207_32\\bin\\'
        elif checkIsLinuxPlatform():
            module_path = os.path.dirname(script_path)
            module_path = os.path.join(module_path, "platform")
            module_path = os.path.join(module_path, "linux")
            if checkIs64BitPlatform():
                module_path = os.path.join(module_path, "64")
            else:
                module_path = os.path.join(module_path, "32")
        elif checkIsMacPlatform():
            module_path = '/Applications/isegcontrol.app/Contents/Frameworks/'

    if module_path != "" :
        sys.path.append(module_path)

    ## the module name (depends on the python version)
    if module_name == "" :
#        module_name = "icsClientPython_{0}".format(sys.version_info[0])
        module_name = "icsClientPython_{0}_for_iCS".format(sys.version_info[0])

    ## try to import the module
    try:
        icsClient =  importlib.import_module(module_name)

    except ImportError as exc:

        ## print traceback
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        print("".join('!! ' + line for line in lines))

        print("Failed to load the module \"{0}\"".format(module_name))
        print("   from path \"{0}\"".format(module_path))
        full_path = os.path.join(module_path, module_name)
        if checkIsLinuxPlatform():
            full_path = full_path + ".so"
        if not os.path.exists(full_path) :
            print("   {0} doesn't exist".format(full_path))
            print("   => make sure that the module is found at this location")
        print("= The Python modules links to shared Qt libraries")
        print("   => make sure the Qt libraries are found")
        if checkIsLinuxPlatform() :
            print("   => make sure the LD_LIBRARY_PATH environment variable is set properly.")
            try:
                path = os.environ['LD_LIBRARY_PATH']
            except KeyError as exc:
                path = "--undefined--"
            print("   => LD_LIBRARY_PATH == {0}".format(path))

        return 0

    print("Sucessfully imported module \"{0}\" (version {1}) from path \"{2}\"".format(module_name, icsClient.getVersion(), module_path))

    return icsClient


def establishConnection(icsClient, connection_name, connection_ip, connection_login, connection_password):

    # icsClient.connect(<connection name>, <ip:port>, <login>, < password>)
    rc = icsClient.connect(connection_name, connection_ip, connection_login, connection_password)
    if rc == 'error':
        print("The connection \"{0}\" already exists".format(connection_name))

    return waitForConnection(icsClient, connection_name, connection_ip)

def establishScriptConnection(icsClient, connection_name, connection_ip, session_id_client, login, password):

    # third connection is created when the script is launched from an iCSservice.
    # It allows to exchange any kind of data with the iCS.

    if connection_ip == "":
        connection_ip = "ws://localhost:8080"

    if connection_name == "":
        connection_name = "script_localhost"

    # icsClient.scriptConnect(<connection name>, <ip:port>, <script_name>)
    script_name = getScriptName();
    rc = icsClient.scriptConnect(connection_name, connection_ip, script_name, session_id_client, login, password)
    if rc == 'error':
        print("The connection \"{0}\" already exists".format(connection_name))

    connection = waitForConnection(icsClient, connection_name, connection_ip)
    if type(connection) == icsClient.connection:
        # success => activate data passthrough
        print("Activating Pass Through for \"{0}\".".format(script_name))
        connection.activateDataPassThrough()

    return connection

def waitForConnection(icsClient, connection_name, connection_ip):

    # this might take a while.
    print("Waiting for connection to be created ...")
    counter = 0
    connection = icsClient.findConnection(connection_name)
    while counter < 10 and type(connection) != icsClient.connection:
        time.sleep(1)
        connection = icsClient.findConnection(connection_name)
        counter = counter + 1

    if type(connection) != icsClient.connection:
        ## list is empty
        print("The connection \"{0}\" couldn't be created.".format(connection_name))
        return 0

    # this might take a while.
    print("Waiting for connection to {0} to be ready ...".format(connection_ip))
    counter = 0
    status = connection.getStatusConnectionStatus()
    while counter < 10 and status != "Configured" :
        time.sleep(1)
        new_status = connection.getStatusConnectionStatus()
        counter = counter + 1
        if new_status != status:
            status = new_status
            print("{0} sec : connection status == {1}".format(counter, status))

    if status != "Configured" :
        ## list is empty
        print("Connection to {0} couldn't be established.".format(connection_ip))
        return 0

    print("Connected to \"{0}\"".format(connection))
    return connection

def disconnectAll(icsClient):
    all_connections = icsClient.getConnections()
    for connection in all_connections:
        connection.disconnect()

def troubleshootConnection(connection):
    status = connection.getStatusConnectionStatus()
    if status != "Configured" :
        print("troubleshoot : The connection {0} isn't ready.".format(connection))
        return False
    print("troubleshoot : The connection {0} is ready.".format(connection))
    return True

def troubleshoot(icsClient):

    all_connections = icsClient.getConnections()
    if len(all_connections) == 0:
        print("troubleshoot : No connection defined.")
        return False

    for connection in all_connections:
        # check the connections first
        rc = troubleshootConnection(connection)
        # if rc == True:

    return True

