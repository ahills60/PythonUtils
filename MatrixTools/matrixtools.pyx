#!/usr/bin/env cython

"""
A simple Cython implementation of Matrix commands
Uses an ndarray for a point implementation.
$Author: andrew $
$Rev: 142 $
$Date: 2009-06-11 15:07:01 +0100 (Thu, 11 Jun 2009) $
"""

from c_numpy cimport import_array, ndarray

import_array()

import numpy
import numpy.linalg
import copy

cdef extern from "math.h":
    double sqrt(double x)

cdef double sqdist(ndarray P, ndarray Q):
    """
        Distance between points
    """
    cdef double *p = <double *>P.data, *q = <double *>Q.data
    cdef double tmp, result = 0
    cdef int i
    cdef int dim = P.dimensions[0]
    
    for i from 0<=i<dim:
        tmp = p[i] - q[i]
        result += tmp * tmp
    return result

cpdef ndarray inversepart(ndarray PI, ndarray Q, ndarray R = None, ndarray S = None):
    """
        inversepart(PI, Q, R, S)
        
        Q, R and S are defined as follows:
            
                [[P, Q],
                 [R, S]]
            
        Where P and S are square matrices and PI is the inverse of P.
        
        Inverse a matrix using the partition matrix lemma
        
    """
    
    assert PI.shape[0] == PI.shape[1], "The inverse P matrix needs to be square"
    
    if R is None or S is None:
        # Go on the assumption that we need to do Q(:end-1) -> Q', Q'.T -> R, and Q(end) -> S
        
        if len(Q.shape) == 1:
            # A vector
            S = numpy.array([Q[-1]]);
            Q = numpy.array([Q[:-1]]);
            R = Q.T;
        else:
            # A matrix
            S = Q[PI.shape[0]:];
            Q = Q[:PI.shape[0]];
            R = Q.T;
    
    assert S.shape[0] == S.shape[1], "The S matrix must be square."
    
    # Transpose if necessary        
    if R.shape[0] == PI.shape[1] and R.shape[0] != R.shape[1]: R = R.T;
    if Q.shape[1] == PI.shape[0] and Q.shape[0] != Q.shape[1]: Q = Q.T;
    
    # Calculate solution
    PIQ = numpy.dot(PI, Q);
    SRPIQ = numpy.linalg.inv(S - numpy.dot(R, PIQ));
    RPI = numpy.dot(R, PI);
    
    QP = -1 * numpy.dot(PIQ, SRPIQ);
    PP = PI - numpy.dot(QP, RPI);
    RP = -1 * numpy.dot(SRPIQ, RPI);
    SP = SRPIQ;
    
    return numpy.vstack([numpy.hstack([PP, QP]), numpy.hstack([RP, SP])])

cdef class blockmat:
    """
        A block matrix
    """
    cdef list blocks, start_row, start_col, end_row, end_col, diagonal_hit, gradients, reference
    cdef int rows, cols
    cdef ndarray offset
    cdef int counter
    
    def __init__(self, rows = None, columns = None, offset = None):
        """
            blockmatrix(rows = None, columns = None, offset = None)
            
            Create a Block Matrix object of a given size. If no size specified, will take
            the shape of the data inside the matrix.
            
            The offset (default is zero) sets the value of the sparsity in the otherwise 
            sparse matrix.
            
        """
        self.counter = -1
        self.reference = []
        self.blocks = []
        self.start_row = []
        self.start_col = []
        self.end_row = []
        self.end_col = []
        self.diagonal_hit = []
        self.gradients = []
        if rows is None:
            self.rows = -1;
        else:
            self.rows = rows;
        if columns is None:
            self.cols = -1;
        else:
            self.cols = columns;
        
        if not isinstance(offset, numpy.ndarray):
            if offset is None:
                self.offset = numpy.array([[0]]);
            else:
                self.offset = numpy.array([[offset]]);
        else:
            self.offset = offset;
    
    cpdef __quickSet(self, list blocks, list start_row, list start_col, list end_row, list end_col, list diagonal_hit, list gradients, list reference, int counter):
        """
            Direct set method to internal structure
        """
        self.blocks = blocks;
        self.start_row = start_row
        self.start_col = start_col
        self.end_row = end_row
        self.end_col = end_col
        self.diagonal_hit = diagonal_hit
        self.gradients = gradients
        self.reference = reference
        self.counter = counter
    
    cpdef ndarray getOffset(self):
        """
            a.getOffset
            
            Get the offset of the matrix
        """
        return self.offset
    
    cpdef tuple shape(self):
        """
            Tuple of array dimensions
        """
        cdef int rows
        cdef int cols
        
        rows = self.rows
        if rows == -1: rows = max(self.end_row)
        cols = self.cols
        if cols == -1: cols = max(self.end_col)
        
        return tuple((rows, cols))
    
    cpdef int addBlock(self, ndarray block, int atRow, int atCol):
        """
            a.addBlock(ndarray block, int Row, int Column)
        
            Add a block to the matrix to be positioned at (Row, Column)
            
            Returns the submatrix index for the purpose of editing the matrix later.
            
        """
        assert atRow + block.shape[0] <= self.rows or self.rows == -1, "Not enough rows in fixed matrix"
        assert atCol + block.shape[1] <= self.cols or self.cols == -1, "Not enough columns in fixed matrix"
        
        assert self.checkClash(block, atRow, atCol), "Can not insert this matrix at this location due to a clash with another matrix."
        
        # Increment block reference counter and add data.
        self.counter += 1
        self.blocks.append(block)
        self.start_row.append(atRow)
        self.start_col.append(atCol)
        self.end_row.append(atRow + block.shape[0]);
        self.end_col.append(atCol + block.shape[1]);
        self.diagonal_hit.append(self.checkDiagonal(len(self.blocks) - 1, True))
        self.reference.append(self.counter)
        
        # Return the counter for access later
        return self.counter
    
    cpdef removeBlock(self, int blockID = -1):
        """
            a.removeBlock(blockID = None)
            
            Removes a block at the given submatrix index.
            Default: removes last block entered
            
        """
        
        assert len(self.blocks) > 0, "No blocks on block stack to delete."
        
        if blockID == -1:
            # Delete the last item added to the block stack
            blockID = len(self.blocks) - 1
        else:
            # Search for ID in references to delete the object
            if blockID >= self.counter or blockID < 0: raise IndexError("Invalid block index.")
            if self.reference.count(blockID) == 0: raise KeyError("Unable to find block reference. Possibly removed earlier.")
            
            blockID = self.reference.index(blockID)
        
        self.blocks.__delslice__(blockID, blockID + 1)
        self.start_row.__delslice__(blockID, blockID + 1)
        self.start_col.__delslice__(blockID, blockID + 1)
        self.end_row.__delslice__(blockID, blockID + 1)
        self.end_col.__delslice__(blockID, blockID + 1)
        self.diagonal_hit.__delslice__(blockID, blockID + 1)
        self.reference.__delslice__(blockID, blockID + 1)
    
    cpdef bint changeBlock(self, int blockID, ndarray newBlock = None, int newStartingRow = -1, int newStartingCol = -1):
        """
            a.changeBlock(blockID, ndarray newBlock = None, int newStartingRow = None, int newStartingCol = None)
            
            Replaces the block with the blockID with the block 'newBlock' if specified. If no block is specified,
            the function assumes the current block should be used.
            New coordinates can also be supplied. If new starting coordinates aren't supplied, it's assumed the
            function should use the current block's location
        """
        cdef list clashList
        
        # Perform input checks
        assert not newBlock is None or not newStartingRow == -1 or not newStartingCol == -1, "Unable to comply. Nothing to do."
        
        if blockID > self.counter or blockID < 0: raise IndexError("Invalid input. Input exceeds counter limits.")
        if self.reference.count(blockID) == 0: raise KeyError("Invalid Input. This reference no longer exists.")
        
        # find where the blockID exists
        blockID = self.reference.index(blockID)
        
        # Convert Nones to meaningful data
        if newBlock is None:
            newBlock = self.blocks[blockID]
        if newStartingRow == -1:
            newStartingRow = self.start_row[blockID]            
        if newStartingCol == -1:
            newStartingCol = self.start_col[blockID]
        
        # Determine if any of the changes will lead to collisions with other blocks
        clashList = self.checkClashWith(newBlock, newStartingRow, newStartingCol)
        # Remove the current blockID from the list returned if it exists
        if clashList.count(blockID) > 0: clashList.remove(blockID)
        
        if len(clashList) > 0:
            print "Clashes appear to exist with other blocks in the matrix. Internal IDs:", clashList
            return False
        
        # Providing no collisions, apply changes
        self.blocks[blockID] = newBlock
        self.start_row[blockID] = newStartingRow
        self.start_col[blockID] = newStartingCol
        
        # Whatever happens, it's best to recalculate the end points in the event the matrix has moved position
        self.end_row[blockID] = newStartingRow + newBlock.shape[0]
        self.end_col[blockID] = newStartingCol + newBlock.shape[1]
        
        self.diagonal_hit[blockID] = self.checkDiagonal(blockID, True)
        
        return True
    
    cpdef ndarray getBlock(self, int blockID = -1):
        """
            a.getBlock(blockID)
            
            Retrieves the block with the given index.
            
        """
        if blockID > self.counter or blockID < 0: raise IndexError("Invalid input. Input either exceeds counter limits")
        if self.reference.count(blockID) == 0: raise KeyError("Invalid input. This reference no longer exists")
        
        # find where the blockID exists
        blockID = self.reference.index(blockID)
        
        return self.blocks[blockID]
    
    cpdef tuple getStartingPoint(self, int blockID = -1):
        """
            a.getStartingPoint(blockID)
            
            Retrieves the starting point of the block with the given index
            
        """
        
        if blockID > self.counter or blockID < 0: raise IndexError("Invalid input. Input either exceeds counter limits")
        if self.reference.count(blockID) == 0: raise KeyError("Invalid input. This reference no longer exists")
        
        # find where the blockID exists
        blockID = self.reference.index(blockID)
        
        return (self.start_row[blockID], self.start_col[blockID])
    
    def __copy__(self):
        """
            Copy the variable
        """
        return self.copy()
    
    cpdef copy(self):
        """
            Copy the object
        """
        
        cdef blockmat tempVar
        
        tempVar = blockmat(self.rows, self.cols, self.offset)
        tempVar.__quickSet(copy.deepcopy(self.blocks), self.start_row, self.start_col, self.end_row, self.end_col, self.diagonal_hit, self.gradients, self.reference, self.counter)
        
        return tempVar
    
    def __add__(self, y):
        """
            Override the add function to cater for block matrices
        """
        tempVar = self.copy()
        tempVar._add(y)
        return tempVar
    
    cpdef _add(self, y):
        if isinstance(y, type(self)):
            # Working with another block matrix
            print "Another block matrix"
        elif isinstance(y, ndarray):
            # Working with a numpy array
            print "A numpy array"
        elif isinstance(y, int) or isinstance(y, float):
            # Working with a constant
            for blockIndex, block in enumerate(self.blocks):
                self.blocks[blockIndex] = block + y
            self.offset += numpy.array([[y]]);
    
    def __radd__(self, y):
        """
            Override the radd function to cater for block matrices
        """
        tempVar = self.copy()
        tempVar._radd(y)
        return tempVar
    
    cpdef _radd(self, y):
        if isinstance(y, type(self)):
            # Working with another block matrix
            print "Another block matrix"
        elif isinstance(y, ndarray):
            # Working with a numpy array
            print "A numpy array"
        elif isinstance(y, int) or isinstance(y, float):
            # Working with a constant
            for blockIndex, block in enumerate(self.blocks):
                self.blocks[blockIndex] = block + y
            self.offset += numpy.array([[y]]);
    
    def __sub__(self, y):
        """
            Override the subtraction function for block matrices
        """
        tempVar = self.copy()
        tempVar._sub(y)
        return tempVar
    
    cpdef _sub(self, y):
        if isinstance(y, type(self)):
            # Working with another block matrix
            print "Another block matrix"
        elif isinstance(y, ndarray):
            # Working with a numpy array
            print "A numpy array"
        elif isinstance(y, int) or isinstance(y, float):
            # Working with a constant
            for blockIndex, block in enumerate(self.blocks):
                self.blocks[blockIndex] = block - y
            self.offset -= numpy.array([[y]]);
    
    def __rsub__(self, y):
        """
            Override the reflected subtraction function for block matrices
        """
        tempVar = self.copy()
        tempVar._rsub(y)
        return tempVar
    
    cpdef _rsub(self, y):
        if isinstance(y, type(self)):
            # Working with another block matrix
            print "Another block matrix"
        elif isinstance(y, ndarray):
            # Working with a numpy array
            print "A numpy array"
        elif isinstance(y, int) or isinstance(y, float):
            # Working with a constant
            for blockIndex, block in enumerate(self.blocks):
                self.blocks[blockIndex] = y - block
            self.offset = y - numpy.array([[y]]);
    
    def __mul__(self, y):
        """
            Override the multiplication function for block matrices
        """
        tempVar = self.copy()
        tempVar._mul(y)
        return tempVar
    
    cpdef _mul(self, y):
        if isinstance(y, type(self)):
            # Working with another block matrix
            print "Another block matrix"
        elif isinstance(y, ndarray):
            # Working with a numpy array
            print "A numpy array"
        elif isinstance(y, int) or isinstance(y, float):
            # Working with a constant
            for blockIndex, block in enumerate(self.blocks):
                self.blocks[blockIndex] = block * y
            self.offset *= numpy.array([[y]]);
    
    def __rmul__(self, y):
        """
            Override the reflective multiplication function for block matrices
        """
        tempVar = self.copy()
        tempVar._rmul(y)
        return tempVar
    
    cpdef _rmul(self, y):
        if isinstance(y, type(self)):
            # Working with another block matrix
            print "Another block matrix"
        elif isinstance(y, ndarray):
            # Working with a numpy array
            print "A numpy array"
        elif isinstance(y, int) or isinstance(y, float):
            # Working with a constant
            for blockIndex, block in enumerate(self.blocks):
                self.blocks[blockIndex] = y * block
            self.offset = y * numpy.array([[y]]);
    
    def __div__(self, y):
        """
            Override the division function for block matrices
        """
        tempVar = self.copy()
        tempVar._div(y)
        return tempVar
    
    cpdef _div(self, y):
        if isinstance(y, type(self)):
            # Working with another block matrix
            print "Another block matrix"
        elif isinstance(y, ndarray):
            # Working with another numpy array
            print "A numpy array"
        elif isinstance(y, int) or isinstance(y, float):
            for blockIndex, block in enumerate(self.blocks):
                self.blocks[blockIndex] = block / y
            self.offset /= numpy.array([[y]])
    
    def __rdiv__(self, y):
        """
            Override the reflective division function for block matrices
        """
        tempVar = self.copy()
        tempVar._rdiv(y)
        return tempVar
    
    cpdef _rdiv(self, y):
        if isinstance(y, type(self)):
            # Working with another block matrix
            print "Another block matrix"
        elif isinstance(y, ndarray):
            # Working with another numpy array
            print "A numpy array"
        elif isinstance(y, int) or isinstance(y, float):
            for blockIndex, block in enumerate(self.blocks):
                self.blocks[blockIndex] = y / block
            self.offset = y / numpy.array([[y]])
    
    def __pow__(self, y, z):
        """
            Override the power function for block matrices
        """
        tempVar = self.copy()
        tempVar._pow(y, z)
        return tempVar
    
    cpdef _pow(self, y, z):
        self.offset = pow(self.offset, y, z)
        for blockIndex, block in enumerate(self.blocks):
            self.blocks[blockIndex] = pow(block, y, z)
            
    def __mod__(self, y):
        """
            Override the mod function (a % b) for block matrices
        """
        tempVar = self.copy()
        tempVar._mod(y)
        return tempVar
    
    cpdef _mod(self, y):
        if isinstance(y, type(self)):
            # working with another block matrix
            print "Another block matrix"
        elif isinstance(y, ndarray):
            # working with a numpy array
            print "A numpy array"
        else:
            # Working with something else (e.g. integer, float)
            self.offset = self.offset % y
            for blockIndex, block in enumerate(self.blocks):
                self.blocks[blockIndex] = block % y
    
    def __richcmp__(self, y, op):
        """
            Override the rich comparison function for block matrices
        """
        tempVar = self.copy()
        if op == 0:
            tempVar._lt(y)
        elif op == 1:
            tempVar._le(y)
        elif op == 2:
            tempVar._eq(y)
        elif op == 3:
            tempVar._ne(y)
        elif op == 4:
            tempVar._gt(y)
        elif op == 5:
            tempVar._ge(y)
        else:
            return None
        return tempVar
    
    cpdef _lt(self, y):
        if isinstance(y, type(self)):
            # Working with another block matrix
            print "Another block matrix"
        elif isinstance(y, ndarray):
            # Working with a numpy array
            print "A numpy array"
        else:
            # Anything else:
            print "other"
    
    cpdef _le(self, y):
        if isinstance(y, type(self)):
            # Working with another block matrix
            print "Another block matrix"
        elif isinstance(y, ndarray):
            # Working with a numpy array
            print "A numpy array"
        else:
            # Anything else:
            print "other"
    
    cpdef _eq(self, y):
        if isinstance(y, type(self)):
            # Working with another block matrix
            print "Another block matrix"
        elif isinstance(y, ndarray):
            # Working with a numpy array
            print "A numpy array"
        else:
            # Anything else:
            print "other"    
    
    cpdef _ne(self, y):
        if isinstance(y, type(self)):
            # Working with another block matrix
            print "Another block matrix"
        elif isinstance(y, ndarray):
            # Working with a numpy array
            print "A numpy array"
        else:
            # Anything else:
            print "other"
    
    cpdef _ge(self, y):
        if isinstance(y, type(self)):
            # Working with another block matrix
            print "Another block matrix"
        elif isinstance(y, ndarray):
            # Working with a numpy array
            print "A numpy array"
        else:
            # Anything else:
            print "other"
    
    cpdef _gt(self, y):
        if isinstance(y, type(self)):
            # Working with another block matrix
            print "Another block matrix"
        elif isinstance(y, ndarray):
            # Working with a numpy array
            print "A numpy array"
        else:
            # Anything else:
            print "other"
    
    def __abs__(self):
        """
            Override the absolute function for block matrices
        """
        tempVar = self.copy()
        tempVar._abs()
        return tempVar
    
    cpdef _abs(self):
        self.offset = abs(self.offset)
        for blockIndex, block in enumerate(self.blocks):
            self.blocks[blockIndex] = abs(block)
    
    cpdef T(self):
        """
            Same as self.transpose() except self is returned for self
        """
        cdef blockmat tempMat
        
        # Perhaps not the most efficient method, so may need reviewing
        tempMat = self.copy()
        tempMat.transpose()
        return tempMat
    
    cpdef transpose(self):
        """
            a.transpose()
        
            Returns a view of 'a' with axes tranposed.
        """
        cdef int tempInt
        cdef list tempList
        
        # Simply a case of swapping over variables and finding transpose of block matrices
        tempInt = self.rows;
        self.rows = self.cols;
        self.cols = tempInt;
        tempList = self.start_row;
        self.start_row = self.start_col;
        self.start_col = tempList;
        tempList = self.end_row;
        self.end_row = self.end_col;
        self.end_col = tempList;
        
        # Gradients will also change, so do 1/x.
        self.gradients = [1.0 / item for item in self.gradients]
        
        self.blocks = [a.T for a in self.blocks];
    
    cpdef float trace(self):
        """
            a.trace()
            
            Return the sum along the diagonals of the block matrix
        """
        cdef list refined
        cdef float result = 0.0
        cdef int endpoint_row, endpoint_col, startpoint_row, startpoint_col, fpoint, r_row, r_col, edge_dist, index, counter
        cdef float gradient, offset
        cdef ndarray value
        
        # Get dimensions of parent matrix for the diagonal
        counter = min(self.rows, self.cols)
        if counter == -1: counter = min(max(self.end_row), max(self.end_col))
        
        # Narrow search to just those that hit the diagonal intersection
        refined = [(index, value) for index, value in enumerate(self.blocks) if self.diagonal_hit[index]]
        
        # Now extract elements
        for element in refined:
            endpoint_row = self.end_row[element[0]]
            endpoint_col = self.end_col[element[0]]
            startpoint_row = self.start_row[element[0]]
            startpoint_col = self.start_col[element[0]]
            gradient = self.gradients[element[0]]
            
            offset = startpoint_col - (endpoint_row * gradient) - 1
            
            # We have an issue where we could end up with a remainder. Take the floor as if it's not in this
            # one, we simply add 1 to the intersection point to determine the next coordinate.
            fpoint = numpy.floor(offset / (1.0 - gradient))
            
            # verify it exists within the confines of the matrix
            r_row = fpoint - startpoint_row
            r_col = fpoint - startpoint_col
            
            # align with edge of matrix
            edge_dist = min(r_row, r_col)
            r_row -= edge_dist
            r_col -= edge_dist
                        
            edge_dist = min(endpoint_row - startpoint_row - r_row, endpoint_col - startpoint_col - r_col)
            for fpoint in range(edge_dist):
                result += element[1][r_row + fpoint, r_col + fpoint]
                counter -= 1
            
        return result + float(counter) * self.offset[0, 0]
    
    cdef bint checkDiagonal(self, int i, bint recalculate = True):
        """
            Check to see if a sub-matrix lies on the diagonal
        """
        cdef int rows, cols, startpoint_row, startpoint_col
        cdef bint result = False
        cdef float gradient, offset, intersectionpoint
        
        startpoint_row = self.start_row[i]
        startpoint_col = self.start_col[i]
        
        rows = self.end_row[i]
        cols = self.end_col[i]
        
        if recalculate:                        
            gradient = float(startpoint_col - cols) / float(rows - startpoint_row)
        else:
            gradient = self.gradients[i]
        
        offset = startpoint_col - (rows * gradient) - 1
        
        # Store or update gradient
        if len(self.gradients) == i:
            self.gradients.append(gradient)
        else:
            self.gradients[i] = gradient
        
        intersectionpoint = offset / (1 - gradient)
        
        # Now check to see if the intersection point occurs within the confines of the block matrix
        if intersectionpoint >= float(startpoint_row) and intersectionpoint < float(rows) and  intersectionpoint >= float(startpoint_col) and intersectionpoint < float(cols): result = True
        
        return result
    
    cdef bint checkClash(self, ndarray block, int atRow, int atCol):
        """
            Returns a simple boolean to indicate whether an interception exists
        """
        # For now, simply use the checkClashWith, but this could be made faster
        # by prematurely terminating/breaking out the search loop upon the first
        # instance of detecting a clash
        
        if len(self.checkClashWith(block, atRow, atCol)) == 0:
            return True
        else:
            return False
    
    cdef list checkClashWith(self, ndarray block, int atRow, int atCol):
        """
            Returns those blocks that the current block will intercept.
        """
        cdef list output = []
        cdef int endRow, endCol, data
        
        endRow = atRow + block.shape[0]
        endCol = atCol + block.shape[1]
        
        for index, data in enumerate(self.reference):
            # Got to check all corners
            if (atRow >= self.start_row[index] and atRow < self.end_row[index]) or (endRow > self.start_row[index] and endRow <= self.end_row[index]) or (atRow <= self.start_row[index] and endRow > self.start_row[index]) or (atRow < self.end_row[index] and endRow >= self.end_row[index]):
                if (atCol >= self.start_col[index] and atCol < self.end_col[index]) or (endCol > self.start_col[index] and endCol <= self.end_col[index]) or (atCol <= self.start_col[index] and endCol > self.start_col[index]) or (atCol < self.end_col[index] and endCol >= self.end_col[index]):
                    output.append(index)
        
        # Output the List through a set which is then converted into a list again.
        # This filters output by only displaying the unique values and even sorts them
        return list(set(output))
    
    cpdef ndarray diagonal(self):
        """
            a.diagonal()
            
            Returns the diagonal of the matrix.
        """
        cdef ndarray output, value
        cdef list refined
        cdef int counter, endpoint_row, endpoint_col, startpoint_row, startpoint_col, fpoint, r_row, r_col, edge_dist, index
        cdef float gradient, offset
        
        # This function is almost a replica of the trace function with the exception to the second "for" loop.
        
        # Get dimensions of parent matrix for the diagonal
        counter = min(self.rows, self.cols)
        if counter == -1: counter = min(max(self.end_row), max(self.end_col))
        
        # Prepare empty vector for filling in with diagonal information
        output = numpy.ones((1, counter)) * self.offset[0, 0]
        
        # Fill this vector with data
        refined = [(index, value) for index, value in enumerate(self.blocks) if self.diagonal_hit[index]]
        
        # Extract Elements
        for element in refined:
            endpoint_row = self.end_row[element[0]]
            endpoint_col = self.end_col[element[0]]
            startpoint_row = self.start_row[element[0]]
            startpoint_col = self.start_col[element[0]]
            gradient = self.gradients[element[0]]
            
            offset = startpoint_col - (endpoint_row * gradient) - 1
            fpoint = numpy.floor(offset / (1.0 - gradient))
            
            r_row = fpoint - startpoint_row
            r_col = fpoint - startpoint_col
            
            edge_dist = min(r_row, r_col)
            r_row -= edge_dist
            r_col -= edge_dist
            
            # Relative distance correction
            fpoint -= edge_dist
            
            edge_dist = min(endpoint_row - startpoint_row - r_row, endpoint_col - startpoint_col - r_col)
            
            for index in range(edge_dist):
                output[0, fpoint + index] = element[1][r_row + index, r_col + index]
            
        return output
    
    cpdef ndarray __array__(self):
        """
            a.__array__() -> copy to a numpy array
        """
        cdef ndarray output
        
        if len(self.blocks) == 0 and min(self.rows, self.cols) <= 0:
            # No blocks, no dimensions
            output = numpy.array([[]])
        else:
            # No blocks, has dimensions OR blocks and with/without dimensions
            output = self.offset * numpy.ones(self.shape())
            
            # Each block in its rightful location
            for blocki, block in enumerate(self.blocks):
                output[self.start_row[blocki]:self.end_row[blocki], self.start_col[blocki]:self.end_col[blocki]] = block
        
        return output
    
    def __str__(self):
        """
            Display the block matrix as a string (convert to an array first)
        """
        return self.__array__().__str__()
    
    def __repr__(self):
        return self.__array__().__repr__()
    
    def __getitem__(self, key):
        """
            Override the indexing API in Python.
        """
        if isinstance(key, int):
            return self.getBlock(key)
        elif isinstance(key, tuple):
            raise NotImplementedError
        else:
            raise NotImplementedError
    
    def __setitem__(self, key, value):
        if isinstance(key, int):
            self.changeBlock(key, value)
        elif isinstance(key, tuple):
            raise NotImplementedError
        else:
            raise NotImplementedError
    
    def tostring(self, order = 'C'):
        """
            a.tostring(order='C')
            
            Construct a Python string containing the raw data bytes in the array.
            
            Parameters
            ----------
            order : {'C', 'F', None}
                Order of the data for multidimensional arrays:
                C, Fortran, or the same as for the original array.
        """
        return self.__array__().tostring(order = order)
    
    def view(self, dtype=None, ptype=None):
        """
            a.view(dtype=None, type=None)
            
            New view of array with the same data.
            
            Parameters
            ----------
            dtype : data-type
                Data-type descriptor of the returned view, e.g. float32 or int16.
            type : python type
                Type of the returned view, e.g. ndarray or matrix.
        """
        if dtype is None:
            if ptype is None:
                return self.__array__().view()
            else:
                return self.__array__().view(type=ptype)
        else:
             if ptype is None:
                 return self.__array__().view(dtype=dtype)
             else:
                 return self.__array__().view(dtype=dtype, type=ptype)
    
cdef class nsblock:
    """
        A non-sparse block framework wrapper for the sparse framework
    """
    cdef blockmat spmat
    cdef int rows, cols
    cdef list blockAddresses, blockIndex, blocksize, rowInstance, colInstance
    
    def __init__(self, rows = None, columns = None):
        self.blockAddresses = []
        self.blockIndex = []
        self.blocksize = []
        self.rowInstance = []
        self.colInstance = []
        if rows is None:
            self.rows = -1
        else:
            self.rows = rows
        if columns is None:
            self.cols = -1
        else:
            self.cols = columns
        # Initialise object
        self.spmat = blockmat()
    
    cpdef __quickSet(self, blockmat spmat, list blockAddresses, list blockIndex, list blocksize, list rowInstance, list colInstance):
        """
            Direct access to set internal structure
        """
        self.spmat = spmat
        self.blockAddresses = blockAddresses
        self.blockIndex = blockIndex
        self.blocksize = blocksize
        self.rowInstance = rowInstance
        self.colInstance = colInstance
    
    cpdef addBlock(self, ndarray block, int row, int column, bint Overwrite = False):
        """
            a.addBlock(ndarray block, int row, int column)
            
            Add the given numpy array block at the given location
            
            If Overwrite is True, the function is the same as a.changeBlock()
            
        """
        cdef int indexAddress, r_row, r_col
        
        assert row >= 0 and column >= 0, "Row or column cannot be negative boundaries"
        if self.cols > -1:
            assert column < self.cols, "Column must not exceed boundary"
        if self.rows > -1:
            assert row < self.rows, "Row must not exceed boundary"
        
        if not Overwrite:
            assert self.blockAddresses.count((row, column)) == 0, "A block matrix already exists at this location"
            
            r_row, r_col = self.getCoordinates(row, column, block.shape)
            
            if r_row == -1:
                print "Unable to add matrix. Size mismatch (row)."
                return
            elif r_row == -2:
                print "Unable to add matrix. Add elements to previous rows before adding element here."
                return
            if r_col == -1:
                print "Unable to add matrix. Size mismatch (col)."
                return
            elif r_col == -2:
                print "Unable to add matrix. Add elements to previous columns before adding element here."
                return
            
            if row == len(self.rowInstance):
                self.rowInstance.append(len(self.blockAddresses))
            if column == len(self.colInstance):
                self.colInstance.append(len(self.blockAddresses))
                        
            indexAddress = self.spmat.addBlock(block, r_row, r_col)
            
            # Create a new address location and block matrix
            self.blockAddresses.append((row, column))
            self.blockIndex.append(indexAddress)
            self.blocksize.append(block.shape)
            
        else:
            # Forward to the changeBlock command
            self.changeBlock(block, row, column)
    
    cpdef bint changeBlock(self, ndarray block, int row, int column, bint CreateIfNonExistant = False):
        """
            a.changeBlock(ndarray block, int row, int column, boolean CreateIfNonExistant)
            
            Change the contents of a block at a given location.
            
            If the CreateIfNonExistant is True, the function is the same as a.addBlock().
        """
        if row < 0 or column < 0: raise IndexError("Row or column cannot be negative boundaries")
        if self.cols > -1:
            if column > self.cols: raise IndexError("Column must not exceed boundary")
        if self.rows > -1:
            if row > self.rows: raise IndexError("Row must not exceed boundary")
        
        if CreateIfNonExistant:
            if self.blockAddresses.count((row,column)) == 0:
                # Run the addBlock command
                self.addBlock(block, row, column)
                return True
            else:
                # Verify shape is the same.
                if not block.shape == self.blocksize[self.blockAddresses.index((row,column))]: raise Exception("Size doesn't match pre-existing matrix. Unable to change size.")
                
                # If here, simply a case of using the change matrix command in the sparse version
                return self.spmat.changeBlock(self.blockIndex[self.blockAddresses.index((row, column))], block)
        else:
            # Change code.
            if self.blockAddresses.index((row,column)) == 0: raise KeyError("A block matrix doesn't yet exist in this location")
            
            # Verify shape is the same.
            if not block.shape == self.blocksize[self.blockAddresses.index((row,column))]: raise Exception("Size doesn't match pre-existing matrix. Unable to change size.")
            
            # If here, simply a case of using the change matrix command in the sparse version
            return self.spmat.changeBlock(self.blockIndex[self.blockAddresses.index((row, column))], block)
    
    cdef tuple getCoordinates(self, int row, int column, tuple shape):
        """
            Finds, calculates and returns the coordinates from block form to
            a relative/normal matrix form
        """
        cdef int r_row, r_col
        
        r_col = 0
        r_row = 0
        
        if row > 0:
            if row > len(self.rowInstance):
                r_row = -2
            elif row == len(self.rowInstance):
                r_row = self.spmat.getStartingPoint(self.blockIndex[self.rowInstance[row - 1]])[0] + self.blocksize[self.rowInstance[row -1 ]][0]
            else:
                r_row = self.spmat.getStartingPoint(self.blockIndex[self.rowInstance[row]])[0]
        if row < len(self.rowInstance):
            if shape[0] != self.blocksize[self.rowInstance[row]][0]:
                r_row = -1
        if column > 0:
            if column > len(self.colInstance):
                r_col = -2
            elif column == len(self.colInstance):
                r_col = self.spmat.getStartingPoint(self.blockIndex[self.colInstance[column - 1]])[1] + self.blocksize[self.colInstance[column - 1]][1]
            else:
                r_col = self.spmat.getStartingPoint(self.blockIndex[self.colInstance[column]])[1]
        if column < len(self.colInstance):
            if shape[1] != self.blocksize[self.colInstance[column]][1]:
                r_col = -1
        return (r_row, r_col)
    
    def __copy__(self):
        """
            Copy the variable
        """
        return self.copy()
    
    cpdef copy(self):
        """
            Copy the object
        """
        cdef nsblock tempVar
        
        tempVar = nsblock(self.rows, self.cols)
        tempVar.__quickSet(self.spmat.copy(), self.blockAddresses, self.blockIndex, self.blocksize, self.rowInstance, self.colInstance)
        
        return tempVar
    
    def __add__(self, y):
        """
            Override the addition function for block matrices
        """
        tempVar = self.copy()
        tempVar._add(y)
        return tempVar
    
    cpdef _add(self, y):
        """
            Add an object to the matrix.
        """
        if isinstance(y, type(self)):
            # Another block type
            print "another block"
        elif isinstance(y, ndarray):
            print "np array"
        else:
            # Anything else: forward to sparse calculations
            self.spmat = self.spmat + y
    
    def __radd__(self, y):
        """
            Override the reflective addition function for block matrices
        """
        tempVar = self.copy()
        tempVar._radd(y)
        return tempVar
    
    cpdef _radd(self, y):
        if isinstance(y, type(self)):
            # Another block type
            print "another block"
        elif isinstance(y, ndarray):
            print "np array"
        else:
            # Anything else: forward to sparse calculations
            self.spmat = y + self.spmat
    
    def __sub__(self, y):
        """
            Override the subtraction function for block matrices
        """
        tempVar = self.copy()
        tempVar._sub(y)
        return tempVar
    
    cpdef _sub(self, y):
        """
            Subtract an object from the matrix
        """
        if isinstance(y, type(self)):
            # Another block type
            print "another block"
        elif isinstance(y, ndarray):
            # Numpy array
            print "np array"
        else:
            # Anything else: forward to sparse calculations
            self.spmat = self.spmat - y
    
    def __rsub__(self, y):
        """
            Override the subtraction function for block matrices
        """
        tempVar = self.copy()
        tempVar._rsub(y)
        return tempVar
    
    cpdef _rsub(self, y):
        """
            Subtract an object from the matrix
        """
        if isinstance(y, type(self)):
            # Another block type
            print "another block"
        elif isinstance(y, ndarray):
            # Numpy array
            print "np array"
        else:
            # Anything else: forward to sparse calculations
            self.spmat = y - self.spmat
    
    def __mul__(self, y):
        """
            Override the multiplication function for block matrices
        """
        tempVar = self.copy()
        tempVar._mul(y)
        return tempVar
    
    cpdef _mul(self, y):
        """
            Multiply an object with the current matrix
        """
        if isinstance(y, type(self)):
            # Another block type
            print "another block"
        elif isinstance(y, ndarray):
            # Numpy array
            print "np array"
        else:
            # Anything else: forward to sparse calculations
            self.spmat = self.spmat * y
    
    def __rmul__(self, y):
        """
            Override the multiplication function for block matrices
        """
        tempVar = self.copy()
        tempVar._rmul(y)
        return tempVar
    
    cpdef _rmul(self, y):
        """
            Multiply an object with the current matrix
        """
        if isinstance(y, type(self)):
            # Another block type
            print "another block"
        elif isinstance(y, ndarray):
            # Numpy array
            print "np array"
        else:
            # Anything else: forward to sparse calculations
            self.spmat = y * self.spmat
    
    def __div__(self, y):
        """
            Override the division function for block matrices
        """
        tempVar = self.copy()
        tempVar._div(y)
        return tempVar
    
    cpdef _div(self, y):
        """
            Divide an object from the current matrix
        """
        if isinstance(y, type(self)):
            # Another block type
            print "another block"
        elif isinstance(y, ndarray):
            # Numpy array
            print "np array"
        else:
            # Anything else: forward to sparse calculations
            self.spmat = self.spmat / y
    
    def __rdiv__(self, y):
        """
            Override the division function for block matrices
        """
        tempVar = self.copy()
        tempVar._rdiv(y)
        return tempVar
    
    cpdef _rdiv(self, y):
        """
            Divide an object from the current matrix
        """
        if isinstance(y, type(self)):
            # Another block type
            print "another block"
        elif isinstance(y, ndarray):
            # Numpy array
            print "np array"
        else:
            # Anything else: forward to sparse calculations
            self.spmat = y / self.spmat
    
    def __pow__(self, y, z):
        """
            Override the power function for block matrices
        """
        tempVar = self.copy()
        tempVar._pow(y, z)
        return tempVar
    
    cpdef _pow(self, y, z):
        """
            Put the current matrix to the power of y
        """
        self.spmat._pow(y, z)
    
    def __mod__(self, y):
        """
            Override the mod function (a % b) for block matrices
        """
        tempVar = self.copy()
        tempVar._mod(y)
        return tempVar
    
    cpdef _mod(self, y):
        self.spmat._mod(y)
    
    def __richcmp__(self, y, op):
        """
            Override the comparison function
        """
        tempVar = self.copy()
        tempVar._richcmp(y, op)
        return tempVar
    
    cpdef _richcmp(self, y, op):
        if op == 0:
            self.spmat = self.spmat._lt(y)
        elif op == 1:
            self.spmat = self.spmat._le(y)
        elif op == 2:
            self.spmat = self.spmat._eq(y)
        elif op == 3:
            self.spmat = self.spmat._ne(y)
        elif op == 4:
            self.spmat = self.spmat._gt(y)
        elif op == 5:
            self.spmat = self.spmat._ge(y)
        else:
            return None
    
    def __abs__(self):
        """
            Override the absolute function for block matrices
        """
        tempVar = self.copy()
        tempVar._abs()
        return tempVar
    
    cpdef _abs(self):
        self.spmat._abs()
    
    def __repr__(self):
        """
            Override the repr function for block matrices
        """
        return self._repr()
    
    cpdef _repr(self):
        return self.spmat.__repr__()
    
    def __str__(self):
        """
            Override the string function for block matrices
        """
        return self._str()
    
    cpdef _str(self):
        return self.spmat.__str__()
    
    cpdef T(self):
        """
            a.T()
            
            Returns the transpose of the block matrix
        """
        tempVar = self.copy()
        tempVar.transpose()
        return tempVar
    
    cpdef transpose(self):
        """
            a.transpose()
            
            Transpose of the block matrix
        """
        cdef list tempList
        
        self.spmat.transpose()
        
        tempList = self.rowInstance
        self.rowInstance = self.colInstance
        self.colInstance = tempList
        
        tempList = [(element[1], element[0]) for element in self.blockAddresses]
        self.blockAddresses = tempList

        tempList = [(element[1], element[0]) for element in self.blocksize]
        self.blocksize = tempList        
    
    cpdef tuple shape(self):
        """
            Tuple of block dimensions
        """
        cdef int rows
        cdef int cols
        
        rows = self.rows
        if rows == -1: rows = len(self.rowInstance)
        cols = self.cols
        if cols == -1: cols = len(self.colInstance)
        
        return tuple((rows, cols))
    
    cpdef tuple blockshape(self):
        """
            Tuple of block element dimensions
        """
        return self.spmat.shape()
    
    cpdef ndarray diagonal(self):
        """
            a.diagonal()
            
            Returns the diagonal of the matrix
        """
        return self.spmat.diagonal()
        
    cpdef ndarray getBlock(self, tuple coordinates):
        """
            a.getBlock(tuple coordinates)
            
            Get the block at the specified coordinates
        """
        cdef int rows, cols
                
        if self.blockAddresses.count(coordinates) == 1:
            # Block exists. simply return it
            return self.spmat.getBlock(self.blockIndex[self.blockAddresses.index(coordinates)])
        else:
            # No block exists at this point, but it could be within constraints
            rows = self.rows
            if rows == -1: rows = len(self.rowInstance)
            cols = self.cols
            if cols == -1: cols = len(self.colInstance)
            
            if coordinates[0] >= rows: raise IndexError("The row index must be within the given matrix size")
            if coordinates[1] >= cols: raise IndexError("The column index must be within the given matrix size")
            
            # Create a background space
            return numpy.ones((self.blocksize[self.rowInstance[coordinates[0]]][0], self.blocksize[self.colInstance[coordinates[1]]][1])) * self.spmat.getOffset()
    
    cpdef list blockDiagonal(self):
        """
            a.blockDiagonal()
            
            Returns a list of blocks that lie on the diagonal
        """
        cdef int toPos, currPos
        cdef list result
        
        result = []
        toPos = min(self.shape())
        
        for currPos in range(toPos):
            if self.blockAddresses.count((currPos, currPos)) == 0:
                # Append a padded result
                result.append(numpy.zeros((self.blocksize[self.rowInstance[currPos]][0], self.blocksize[self.colInstance[currPos]][1])))
            else:
                # Append a retrieved result
                result.append(self.spmat.getBlock(self.blockAddresses.index((currPos, currPos))))
        
        return result
    
    cpdef blockmat exportMatrix(self):
        """
            a.exportMatrix()
            
            Returns the block matrix in type blockmat
            
        """
        return self.spmat
    
    cpdef ndarray __array__(self):
        return self.spmat.__array__()
    
    def __getitem__(self, key):
        """
            Override the index 
        """
        if isinstance(key, tuple):
            return self.getBlock(key)
        else:
            raise TypeError("Unrecognised index type")
    
    def __setitem__(self, key, value):
        """
            Set the value given this index
        """
        if isinstance(key, tuple):
            self.changeBlock(value, key[0], key[1], True)
        else:
            raise TypeError("Unrecognised index type")
    

cpdef blockmat np2bm(ndarray array, bint autoFind = True, float sparseValue = 0.0):
    """
        np2bm(array)

        Convert a Numpy array to a sparse block matrix
    """
    cdef blockmat output
    cdef list candidates

    if autoFind:
        # Identify the most commonly associated values 
        elementList = numpy.unique(array)
        candidates = sorted([(numpy.sum(array == currVal), currVal) for currVal in elementList], reverse = True)
        # Check first two to see if they're equal
        if candidates[0][0] == candidates[1][0]:
            # There's a chance that there are more than two with this 
            candidates = [currVal[1] for currVal in candidates if currVal[0] == candidates[0][0]]
            
            sparseValue = candidates[numpy.argmin(abs(numpy.array(candidates) - numpy.mean(array)))]
        else:
            # One candidate stood out. Elect them
            sparseValue = candidates[0][1]
            
    output = blockmat(offset = sparseValue)

    return output
