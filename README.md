# PythonUtils
PythonUtils provides useful Python functions including file reading and writing tools. 

## Libraries
PythonUtils comes with various Python but are not bundled together. Instead, they are currently partitioned into separate directories. 

### IOTools
IOTools contains `quickio.py` which provides numerous functions for reading and writing files. The `quickio` class can read and write MATLAB `mat` files providing the file extension is set to `.mat`. Commands to ``load`` and ``save`` mimic MATLAB's behaviour.

* ``quickio.load(filename)`` extracts the contents of the file into the parent workspace
* ``quickio.save(filename, *args)`` saves the variables named in ``args`` into a file. Variables can be separated with commas within a single string or as multiple arguments.
* ``quickio.read(filename)`` returns the contents of the file as a dictionary object.
* ``quickio.write(filename, dict)`` saves the contents of the dictionary object ``dict`` as a file.

The `quickio` library uses cPickle to convert variables into saveable objects. The resultant pickled object is compressed using the ``gzip`` library and saved as a file. MATLAB files are written and read using functions provided by ``scipy.io``.

### MatrixTools
MatrixTools was a series of classes built in Cython to handle block and sparse block matrices using ``numpy`` as a foundation. Progress within this project has stagnated but may be resumed at a later time. It is included as it contains code that may be useful for other projects.

### QueueUtils
QueueUtils are some classes that can be used to distribute tasks across multiple computers. The server class `QueueServer` handles and manages jobs that are issued to clients using the `QueueClient` class. 

Rudimentary job handle is performed by the server such that jobs that were assigned but remain incomplete after a defined time are placed back onto the job list and reassigned.

* `QueueServer.py` provides basis server functionality that is used to distribute data to clients. It can also receive results from clients and save them on the server. Basic job handling and management is performed by the `QueueServer` class.
* `QueueClient.py` provides tools to request data from a server and send results back to the server.

### Template Builder
The template builder script `build.py` constructs a file by taking the contents of multiple files as instructed by a master template file. Technically, the same behaviour can be achieved by calling a functions or macros within other files, but this template builder can remove some of the overheads this can incur. Indents are preserved meaning the template builder can be used by languages such as Python.

The master file can link to another file by declaring a comment with the instructions: `INPUTFILE:filename.ext`. For instance, in the following Python example, the master script `testscript.py` links to `innerloop.py`:

```python
# This is a test script
for a in range(10):
    #INPUTFILE:innerloop.py
print "End of script."
```

If the contents of `innerloop.py` contained the following Python code:

```python
print "The result here is..."
print a
```

Then the resulting constructed script using `build.py testscript.py compiledscript.py` will be:

```python
# This is a test script
for a in range(10):
    print "The result here is..."
    print a
print "End of script."
```

Currently the template builder supports the following languages:

* MATLAB
* LaTeX
* Python

*Note: The template builder does not handle nested `INPUTFILE` requests: only the master file is processed. A workaround would be to perform further iterations of `build.py` on the constructed file.*