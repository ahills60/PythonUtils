"""
A simple implementation of a Client that receives variables from a server and
informs the server of a finished job, in addition to optionally uploading data.
$Author: andrew $
$Rev: 167 $
$Date: 2009-11-03 14:32:49 +0000 (Tue, 03 Nov 2009) $
"""

import cPickle
import socket
import uuid
import time

class QueueClient:
    
    def __init__(self, appID, serverAddress = None, serverPort = 29382, bufferSize = 4096):
        """
            QueueClient(appID, serverAddress = None, serverPort = 29328, bufferSize = 4096)
            
            An AppID must be specified to ensure the correct data is transferred between the
            server and clients. If the server address isn't specified, it is assumed the
            local host will also act as a server.
        """
        self.appID = appID
        if serverAddress is not None:
            self.serverAddress = serverAddress
        else:
            self.serverAddress = socket.gethostbyname(socket.gethostname()) # Get the local host's address
        self.serverPort = serverPort
        self.bufferSize = bufferSize
        self.QueueUUID = str(uuid.getnode())
        self.CurrentJobUUID = ""
        self.transmissionType = 2
    
    def GetNewParameters(self):
        """
            GetNewParameters()
            
            Contacts the server for a new batch of parameters. Returns a dictionary
            object of parameters. Should an integer of value -1 be returned, an invalid
            AppID was received from the server. A returned value of -2 indicates an End
            of Queue state such that no future jobs exist on the server. A returned
            value of -3 represents a problem unpickling the data. This may occur should
            the transmission of data become corrupted or is incomplete.
        """
        # Contact server to request new parameters
        serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        print "Attempting to connect to the server..."
        ADDR = (self.serverAddress, self.serverPort)
        # addr = (socket.gethostbyname(socket.gethostname()), 12345)
        # serverSocket.bind(addr)
        print "Attempting to connect to address..."
        serverSocket.connect(ADDR)
        
        print "Sending AppID to server..."
        serverSocket.send(str(self.appID))
        
        print "Receiving Data..."
        data = serverSocket.recv(self.bufferSize)
        if data == "Invalid AppID":
            print "QueueClient\\GetNewParameters: Received Invalid AppID response from Server"
            serverSocket.close()
            serverSocket = None
            return -1
        
        print "Sending Client UUID to server..."
        serverSocket.send(self.QueueUUID)
        
        print "Receiving Item UUID..."
        self.CurrentJobUUID = serverSocket.recv(self.bufferSize)
        
        print "Item UUID is " + self.CurrentJobUUID
        
        # Tell server which service we wish to use
        serverSocket.send("PARAM")
        
        print "Receiving Data..."
        data = serverSocket.recv(self.bufferSize)
        
        if data == "EOQ":
            # End of Queue
            print "QueueClient\\GetNewParameters: Received End of Queue response from Server"
            serverSocket.close
            return -2
        
        print "Data received: \"" + data + "\""
        if data.isdigit():
            # This indicates the number of pages. Return the number to confirm and begin
            print "Digit received. Confirming with server..."
            noPages = int(data)
            serverSocket.send(str(noPages))
            
            print "Confirmed. Starting data receive..."
            outputString = ""
            
            for page in range(noPages):
                print "     Receiving page " + str(page) + " of " + str(noPages)
                data = serverSocket.recv(self.bufferSize)
                outputString = outputString + data
        else:
            # This doesn't have an indication as to the number of pages. It's dynamic. See what we can do...        
            print "Data received. Keep coming..."
            outputString = ""
        
            while not data.endswith('.'):
                # End of transmission
                outputString = outputString + data
                data = serverSocket.recv(self.bufferSize)
            outputString = outputString + data
        
        print "Receive complete."
        
        # Data received from node. Unpickle and return
        try:
            print "Attempting to pickle data..."
            outputString = cPickle.loads(outputString)
        except Exception, e:
            outputString = -3
        return outputString
    
    def JobComplete(self, results = None, filename = None):
        """
            JobComplete(results = None, filename = None)
            
            Let's the server know that a job has been completed.
            If results is specified, then the results are also uploaded to the server.
            If a filename is provided, then the results are saved on the server with the
            given filename, otherwise the filename is generated from the current job
            UUID, date and time.
            Returns a boolean determining the status of the transfer. If an error occurs
            in either the handshaking of the AppID or transfer, then a False is returned.
            Otherwise a True is returned.
        """
        
        if self.CurrentJobUUID == "":
            print "Unable to send a Job Complete command when a job hasn't started."
            return False
        
        # Contact server to Send results if there are any
        serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        print "Attempting to connect to the server..."
        ADDR = (self.serverAddress, self.serverPort)
        
        print "Attempting to connect to address..."
        serverSocket.connect(ADDR)
        
        print "Sending AppID to server..."
        serverSocket.send(str(self.appID))
        
        print "Receiving Data..."
        data = serverSocket.recv(self.bufferSize)
        if data == "Invalid AppID":
            print "QueueClient\\JobComplete: Received Invalid AppID response from Server"
            serverSocket.close()
            serverSocket = None
            return False
        
        print "Sending Client UUID to server..."
        serverSocket.send(self.QueueUUID)
        
        print "Receiving Transaction UUID..."
        transUUID = serverSocket.recv(self.bufferSize)
        
        print "Transaction UUID is " + transUUID
        
        serverSocket.send("COMPL")
        
        print "Sending Current Job UUID to server..."
        serverSocket.send(self.CurrentJobUUID)
        
        data = serverSocket.recv(self.bufferSize)
        
        if data != "OK":
            print "Response from server: " + data
            serverSocket.close()
            return False
        
        if results is None:
            # No results to send
            serverSocket.send("Done")
            serverSocket.close()
            self.CurrentJobUUID = ""
            return True
        else:
            # Has a result but it's not in the dictionary object. Quickly wrap it...
            if not isinstance(results, dict): results = {'result': results}
            serverSocket.send("Sending results")
        
        if not serverSocket.recv(self.bufferSize) == "GO":
            print "Can't send data. Unrecognised command."
            return False
        
        if filename is None:
            # Quickly generate something
            today = time.gmtime()
            filename = self.CurrentJobUUID + "-" + str(today.tm_year) + str(today.tm_mon) + str(today.tm_mday) + "-" + str(today.tm_hour) + str(today.tm_min) + '.sav'
        
        # Send the filename to the server
        serverSocket.send(filename)
        
        if not serverSocket.recv(self.bufferSize) == "OK":
            print "Filename rejected."
            return False
        
        results = cPickle.dumps(results, cPickle.HIGHEST_PROTOCOL) # Pickle the object so it can be sent via the network
        
        if self.transmissionType == 1:
            # Based on dynamically stopping based on a '.' at the end of the message
            print "    Transmission type 1:"
            breakPoints = range(0, len(results), self.bufferSize - 24)
            if breakPoints[-1] != len(results): breakPoints.append(len(results))
            for ind in range(len(breakPoints) - 1):
                while results[breakPoints[ind + 1]] is '.': breakPoints[ind + 1] -= 1 # Adjust the partition
                if breakPoints[ind] <= breakPoints[ind + 1]:
                    while results[breakPoints[ind + 1]] is '.': breakPoints[ind + 1] += 1
                if breakPoints[ind + 1] - breakPoints[ind - 1] > self.bufferSize:
                    print "QueueClient\\JobComplete: Error sending. Unable to find a good break point in the data."
                    return False
                txtFragment = results[breakPoints[ind]:breakPoints[ind+1]]
                print "    Sending text Fragment..."
                serverSocket.send(txtFragment)
        else:
            # Agrees with client as to the number of pages to send
            print "    Transmission type 2"
            breakPoints = range(0, len(results), self.bufferSize)
            if breakPoints[-1] != len(results): breakPoints.append(len(results))
            serverSocket.send(str(len(breakPoints) - 1))
            if int(serverSocket.recv(self.bufferSize)) == len(breakPoints) - 1:
                # Agreed with the size. Begin sending
                for ind in range(len(breakPoints) - 1):
                    txtFragment = results[breakPoints[ind]:breakPoints[ind+1]]
                    print "    Sending text Fragment..."
                    serverSocket.send(txtFragment)
            else:
                print "QueueClient\\JobComplete: Unable to agree with size to send"
        data = serverSocket.recv(self.bufferSize)
        
        if data == "OK":
            print "Data saved successfully."
            self.CurrentJobUUID = ""
            return True
        else:
            print "An error occurred. Message received from server:" + data
            return False
    
    def GetFile(self, filename = None):
        """
            GetFile(filename)
        """
        
        if filename is None:
            print "File not defined"
            return False
        
        # Contact server to Send results if there are any
        serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        print "Attempting to connect to the server..."
        ADDR = (self.serverAddress, self.serverPort)
        
        print "Attempting to connect to address..."
        serverSocket.connect(ADDR)
        
        print "Sending AppID to server..."
        serverSocket.send(str(self.appID))
        
        print "Receiving Data..."
        data = serverSocket.recv(self.bufferSize)
        if data == "Invalid AppID":
            print "QueueClient\\GetFile: Received Invalid AppID response from Server"
            serverSocket.close()
            serverSocket = None
            return False
        
        print "Sending Client UUID to server..."
        serverSocket.send(self.QueueUUID)
        
        print "Receiving Transaction UUID..."
        transUUID = serverSocket.recv(self.bufferSize)
        
        print "Transaction UUID is " + transUUID
        
        serverSocket.send("RETRV")
        
        serverSocket.send(filename)
        
        serverReply = serverSocket.recv(self.bufferSize)
        
        if not serverReply == "OK":
            # was not ok
            print "QueueClient\\GetFile: Error getting file. Status from server: " + serverReply
            serverSocket.close()
            serverSocket = None
            return False
        
        # File has been found. Should receive now.
        
        serverSocket.send("GO")
        
        
    
if __name__ == "__main__":
    print "Creating new Client object..."
    QC = QueueClient(1234)#, 'dyn044170.shef.ac.uk')
    print "Obtaining new parameters from server"
    parameters = QC.GetNewParameters()
    print parameters
    time.sleep(5)
    print QC.JobComplete(parameters)
