"""
A simple implementation of a server that distributes variables to clients
that provide the correct ID
$Author: andrew $
$Rev: 167 $
$Date: 2009-11-03 14:32:49 +0000 (Tue, 03 Nov 2009) $
"""

import os
import socket
import multiprocessing
import cPickle
import time
import uuid
import quickio
import copy

class QueueServer:
    
    def __init__(self, appID, listenPortNumber = 29382, bufferSize = 4096, maintenanceTime = 30, AutoResume = True, transmissionType = 2):
        """
            QueueServer(appID, listenPortNumber = 29382, bufferSize = 4096, maintenanceTime = 30, AutoResume = True)
            
            An AppID should be specified to ensure the correct variables are exchanged should many
            instances of the Server and Clients be operational within a networked enviroment.
            The maintenanceTime dictates how many seconds can pass before the maintenanceThread
            executes to clean up 'forgotten' threads that exceed the timeThreshold variable.
            Auto resume (AutoResume) is set to True as default so the server will open any auto
            resume file it finds at the last save point. This can be disabled by setting the
            variable AutoResume to False.
            
        """
        self.listenPortNumber = listenPortNumber
        self.bufferSize = bufferSize
        self.appID = appID
        self.QueueManager = multiprocessing.Manager()
        self.QueueItemsLock = multiprocessing.Lock()
        self.QueueItems = self.QueueManager.list()
        self.QueueHistory = self.QueueManager.list()
        self.QueueTime = self.QueueManager.list()
        self.QueueDone = self.QueueManager.list()
        self.QueueItID = self.QueueManager.list()
        self.QueueClID = self.QueueManager.list()
        self.EndProgram = multiprocessing.Event()
        self.ServerThreadsOpen = self.QueueManager.Value('ThreadCount', 0)
        self.ServerThreadsOpenLock = multiprocessing.Lock()
        self.ServerThreadsOpenCheck = self.QueueManager.Value('FinalCheck', 0)
        self.UpdatesMade = self.QueueManager.Value('Updated', 0) # No need to lock this as anything != 0 requires an update
        self.maintenanceTime = maintenanceTime
        self.timeThreshold = 12*60*60
        self.transmissionType = transmissionType
        self.SAVE_PATH = "results/"
        self.AutoResume = AutoResume
    
    def ServerThreadOpen(self):
        """
            Updates the statistics of how many threads are open
        """
        if not self.ServerThreadsOpenLock.acquire():
            # Could not obtain lock on server thread counter
            print "QueueServer: An error occurred when trying to get a lock of the Server Thread count."
            return None
        self.ServerThreadsOpen.set(self.ServerThreadsOpen.get() + 1)
        print "Number of threads open: " + str(self.ServerThreadsOpen.get())
        self.ServerThreadsOpenLock.release()
    
    def ServerThreadClose(self):
        """
            Updates the statistics of how many threads are open
        """
        if not self.ServerThreadsOpenLock.acquire():
            # Could not obtain lock on server thread counter
            print "QueueServer: An error occurred when trying to get a lock of the Server Thread count."
            return None
        self.ServerThreadsOpen.set(self.ServerThreadsOpen.get() - 1)
        print "Number of threads open: " + str(self.ServerThreadsOpen.get())
        self.ServerThreadsOpenLock.release()
    
    def ServerThread(self, clientSocket, Address):
        """
            Deals with Server Connections and is given an individual Socket and Address
            to perform this task. This thread performs both the distribution of variables
            in addition to accepting the uploading of data.
        """
        self.ServerThreadOpen()
        print "    Server child thread active..."
        data = clientSocket.recv(self.bufferSize)
        if str(self.appID) != data:
            print "QueueServer\\ServerThread: Invalid AppID from remote computer."
            clientSocket.send("Invalid AppID")
            clientSocket.close()
            self.ServerThreadClose()
            return None
        else:
            print "Confirm AppID is valid..."
            clientSocket.send("AppID Valid")
        
        print "    Getting UUID from client..."
        clientUUID = clientSocket.recv(self.bufferSize)
        
        print "    Send a transaction UUID to client..."
        transUUID = str(uuid.uuid1()) # Randomly generated UUID
        clientSocket.send(transUUID)
        
        # Service client requires:
        serviceType = clientSocket.recv(5)
        
        print "Service type received: " + serviceType
        
        print "    Acquiring a lock on the QueueItems..."
        # Attempt to get a lock of the QueueItems object
        if not self.QueueItemsLock.acquire(): 
            # Could not acquire a lock
            print "QueueServer\\ServerThread: An error occurred when trying to get a lock of the QueueItems resource."
            self.ServerThreadClose()
            return None
        print "    Lock acquired"
        
        if serviceType == "PARAM":
            # Hand out new Parameters if there are some
            if len(self.QueueItems) > 0:
                # Items are still in the queue to process
                print "    Items still to be processed. Processing..."
                variables = self.QueueItems.pop(0) # Pop the first element within the queue
            
                print self.QueueItems
                # Move item to a history list
                self.QueueHistory.append(variables)
                variables = cPickle.dumps(variables, cPickle.HIGHEST_PROTOCOL) # Pickle the object so it can be sent via the network
            
                if self.transmissionType == 1:
                    # Based on dynamically stopping based on a '.' at the end of the message
                    print "    Transmission type 1:"
                    breakPoints = range(0, len(variables), self.bufferSize - 24)
                    if breakPoints[-1] != len(variables): breakPoints.append(len(variables))
                    for ind in range(len(breakPoints) - 1):
                        while variables[breakPoints[ind + 1]] is '.': breakPoints[ind + 1] -= 1 # Adjust the partition
                        if breakPoints[ind] <= breakPoints[ind + 1]:
                            while variables[breakPoints[ind + 1]] is '.': breakPoints[ind + 1] += 1
                        if breakPoints[ind + 1] - breakPoints[ind - 1] > self.bufferSize:
                            print "QueueServer\\ServerThread: Error sending. Unable to find a good break point in the data."
                            self.QueueItemsLock.release() # Release the lock
                            self.ServerThreadClose()
                            return None
                        txtFragment = variables[breakPoints[ind]:breakPoints[ind+1]]
                        print "    Sending text Fragment..."
                        clientSocket.send(txtFragment)
                else:
                    # Agrees with client as to the number of pages to send
                    print "    Transmission type 2"
                    breakPoints = range(0, len(variables), self.bufferSize)
                    if breakPoints[-1] != len(variables): breakPoints.append(len(variables))
                    clientSocket.send(str(len(breakPoints) - 1))
                    if int(clientSocket.recv(self.bufferSize)) == len(breakPoints) - 1:
                        # Agreed with the size. Begin sending
                        for ind in range(len(breakPoints) - 1):
                            txtFragment = variables[breakPoints[ind]:breakPoints[ind+1]]
                            print "    Sending text Fragment..."
                            clientSocket.send(txtFragment)
                    else:
                        print "QueueServer\\ServerThread: Unable to agree with size to send"
            
                print "    Releasing lock..."
                self.QueueItemsLock.release() # Release the lock
                self.QueueDone.append(False)
                self.QueueTime.append(time.time())
                self.QueueItID.append(transUUID)
                self.QueueClID.append(clientUUID)
            else:
                # No items remaining. Inform the node and deal with connection
                self.QueueItemsLock.release() # Might as well release this object early
                clientSocket.send("EOQ")
                clientSocket.close()
                print "    Connection Closed."
                clientSocket = None
                self.ServerThreadClose()
                self.ServerThreadsOpenCheck.set(1)
                return None
        elif serviceType == "RETRV":
            # For Receiving files from the server
            print "Receiving Filename"
            filename = clientSocket.recv(self.bufferSize)
            
            
        elif serviceType == "COMPL":
            # For informing the server that a task has completed
            print "Receiving UUID from client..."
            jobUUID = clientSocket.recv(self.bufferSize)
            
            print "Checking job status..."
            # Check to see if this exists
            if self.QueueItID.count(jobUUID) == 0:
                # This UUID doesn't exist.
                clientSocket.send("Job does not exist")
                clientSocket.close()
                self.QueueItemsLock.release()
                self.ServerThreadClose()
                return None
            
            if self.QueueClID[self.QueueItID.index(jobUUID)] != clientUUID:
                # Client ID doesn't match. Needs to match in order to save data
                clientSocket.send("Client UUID mismatch")
                clientSocket.close()
                self.QueueItemsLock.release()
                self.ServerThreadClose()
                return None
            
            if self.QueueDone[self.QueueItID.index(jobUUID)]:
                # Job has already been done. This shouldn't be the case.
                clientSocket.send("Data already sent")
                clientSocket.close()
                self.QueueItemsLock.release()
                self.ServerThreadClose()
                return None
            
            print "Job exists. Marking job as complete..."
            # This exists. Continue.
            clientSocket.send("OK")
            self.QueueDone[self.QueueItID.index(jobUUID)] = True
            
            # Release Lock
            print "Releasing lock..."
            self.QueueItemsLock.release()
            
            # See what's going on
            data = clientSocket.recv(self.bufferSize)
            
            if data == "Done":
                # No data to send. Close up and go.
                print "No data to receive. Closing Socket..."
                clientSocket.close()
                print "Connection Closed."
                self.ServerThreadClose()
                self.UpdatesMade.set(self.UpdatesMade.get() + 1)
                return None
            elif data == "Sending results":
                # About to receive data. 
                print "Preparing to receive data..."
                clientSocket.send("GO")
            else:
                # Unknown command
                print "Unrecognised command. Terminating connection..."
                clientSocket.send("Unrecognised command. Terminating connection.")
                clientSocket.close()
                self.ServerThreadClose()
                return None
            
            # Should only be here if sending the results
            
            # Get Filename
            print "Getting filename..."
            resultsFilename = clientSocket.recv(self.bufferSize)
            
            clientSocket.send("OK")
            
            data = clientSocket.recv(self.bufferSize)
            print "Receiving Data..."
            if data.isdigit():
                # This indicates the number of pages. Return the number to confirm and begin
                print "Digit received. Confirming with server..."
                noPages = int(data)
                clientSocket.send(str(noPages))
                
                print "Confirmed. Starting data receive..."
                outputString = ""
                
                for page in range(noPages):
                    print "     Receiving page " + str(page) + " of " + str(noPages)
                    data = clientSocket.recv(self.bufferSize)
                    outputString = outputString + data
            else:
                # This doesn't have an indication as to the number of pages. It's dynamic. See what we can do...        
                print "Data received. Keep coming..."
                outputString = ""
                
                while not data.endswith('.'):
                    # End of transmission
                    outputString = outputString + data
                    data = clientSocket.recv(self.bufferSize)
                outputString = outputString + data
            
            print "Receive complete."
            
            # Data received from node. Unpickle and return
            try:
                print "Attempting to pickle data..."
                outputString = cPickle.loads(outputString)
                if not isinstance(outputString, dict):
                    print "Data is in an unexpected type. Terminating connection..."
                    clientSocket.send("Data in unexpected form. Terminating connection.")
                    clientSocket.close()
                    self.ServerThreadClose()
                    return None
                else:
                    if not quickio.write(self.SAVE_PATH + resultsFilename, outputString):
                        # Could not write file
                        clientSocket.send("Could not save results")
                    else:
                        clientSocket.send("OK")
            except Exception, e:
                # Couldn't unpickle
                clientSocket.send("Could not construct data from string.")
        self.UpdatesMade.set(self.UpdatesMade.get() + 1)
        clientSocket.close()
        print "    Connection Closed."
        clientSocket = None
        self.ServerThreadClose()
    
    def ServerMaintenanceThread(self):
        """
            Keeps the system in good working order by placing items back on the queue if
            the job is still outstanding and has been ongoing for a long time.
            
            Executes every maintenanceTime seconds, and moves 'forgotten' jobs that are
            timeThreshold seconds old.
        """
        while not self.EndProgram.is_set():
            if len(self.QueueHistory) > 0:
                # There are items within the Queue History for processing
                if self.QueueDone.count(False) > 0:
                    self.QueueItemsLock.acquire() # Get a steady lock
                    queueIndices = [index for index, item in enumerate(self.QueueDone) if not item] # Items worth investigating
                    itemsToRemove = []
                    for item in queueIndices:
                        if time.time() - self.QueueTime[item] > self.timeThreshold:
                            # Time Threshold Exceeded
                            itemsToRemove.append(item)
                            print "QueueServer\\ServerMaintenanceThread: Queue time threshold exceeded. Adding back to queue..."
                            self.QueueItems.append(self.QueueHistory[item])
                    if len(itemsToRemove) > 0:
                        # Clean up the history
                        itemsToRemove.reverse() # Reverse the removal process so the later indices remain unchanged (considering it's in ascending order)
                        for item in itemsToRemove:
                            print "QueueServer\\ServerMaintenanceThread: Cleaning up outstanding queue history."
                            self.QueueHistory.pop(item);
                            self.QueueTime.pop(item);
                            self.QueueDone.pop(item);
                            self.QueueItID.pop(item);
                            self.QueueClID.pop(item);
                    # put items back in queue
                    self.QueueItemsLock.release() # We can let go.
                    self.UpdatesMade.set(self.UpdatesMade.get() + 1)
                else:
                    # There aren't any outstanding jobs in the queue. Check the main queue
                    if len(self.QueueItems) == 0:
                        print "QueueServer\\ServerMaintenanceThread: Main queue is empty and there are no outstanding jobs."
                        self.EndProgram.set()
                        continue
            if self.UpdatesMade.get() > 0:
                print "QueueServer\\ServerMaintenanceThread: Attempting to write variables following a Queue update..."
                quickio.write(str(appID) + ".sav", {'QueueHistory': self.QueueHistory, 'QueueTime': self.QueueTime, 'QueueDone': self.QueueDone, 'QueueItID': self.QueueItID, 'QueueClID': self.QueueClID, 'QueueItems': self.QueueItems}, True)
                self.UpdatesMade.set(0) # Reset the number of changes made
                print "QueueServer\\ServerMaintenanceThread: Write complete."
                print "\nQueueServer\\ServerMaintenanceThread: Status update: " + str(len(self.QueueItems)) + " items in the queue, " + str(len(itemsToRemove)) + " items moved back to the Queue, " + str(len(queueIndices) - len(itemsToRemove)) + " items being processed, and " + str(len(self.QueueHistory) - len(queueItems) + len(itemsToRemove)) + " completed.\n"
            # Now get this thread to sleep for a custom time (seconds)
            time.sleep(self.maintenanceTime)
        
        # EndProgam == true at this point Clean up if necessary
        return None
    
    def StartServer(self):
        """
            The main part of the code that initates the server connection
        """
        # Start Server Maintenance Thread
        
        if os.path.exists(str(self.appID) + ".sav") and self.AutoResume:
            print "Loading presaved state. "
            variables = quickio.read(str(self.appID) + ".sav")
            self.QueueHistory.extend(variables['QueueHistory'])
            self.QueueTime.extend(variables['QueueTime'])
            self.QueueDone.extend(variables['QueueDone'])
            self.QueueItID.extend(variables['QueueItID'])
            self.QueueClID.extend(variables['QueueClID'])
            self.QueueItems.extend(variables['QueueItems'])
        
        print "Starting Server Maintenance thread..."
        multiprocessing.Process(target=self.ServerMaintenanceThread).start()
        
        addr = (socket.gethostbyname(socket.gethostname()), self.listenPortNumber)
        print "Creating Server socket..."
        serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print "Binding Server socket..."
        serverSocket.bind(addr)
        print "Setting number of listeners..."
        serverSocket.listen(5)
        # serverSocket.settimeout(5.0)
        
        while not self.EndProgram.is_set():
            print "Listening for connections..."
            try:
                clientSocket, Address = serverSocket.accept()
            except socket.timeout, e:
                continue
            
            print "Connection received. Spawning child process..."
            
            # Start a child process to deal with connection
            multiprocessing.Process(target=self.ServerThread, args=(copy.copy(clientSocket), copy.copy(Address))).start()
        print "QueueServer\\StartServer: End Program Acknowledged"
        while self.ServerThreadsOpen.get() > 0 or len(multiprocessing.process.active_children()) > 1 or self.ServerThreadsOpenCheck.get() == 0:
            pass # Wait until any child threads are freed up
        serverSocket.close()
    
    def SetQueue(self, queueItems = None):
        """
            SetQueue(queueItems)
            
            Set items within the queue. This should be a list of dictionary objects.
            If the input is not of type list, then a TypeError exception is raised.
        """
        if not isinstance(queueItems, list) or queueItems is None:
            raise TypeError("Input must be defined and needs to be of type list.")
        self.QueueItemsLock.acquire() # Unlikely it'll be required, but it's good practice
        self.QueueItems.extend(queueItems)
        print self.QueueItems
        self.QueueItemsLock.release()
    
if __name__ == "__main__":
    print "Creating new Server object..."
    QS = QueueServer(1234)
    print "Creating list of items for the queue..."
    queueList = [{'variable1': 1234, 'variable2': 'Test1'}, {'variable1': 4567, 'variable2': 'Test2'}, {'variable1': 7890, 'variable2': 'Test3'}]
    print "Setting the queue..."
    QS.SetQueue(queueList)
    print "Starting the Server process..."
    QS.StartServer()
    print "Server process finished"