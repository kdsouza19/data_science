# Start with importing numpy 
import numpy as np 
import array 
 
# Fixed type arrays 
#convert list in python to array 
L = list(range(15))
#print(L)
A = array.array('i',L) # i' is a type code indicating the contents are integers.
print(A)


#Creating Arrays from Python Lists

# integer array 
np.array([1,3,5,7])

#python lists can have elements of different data types
#numpy is constrained to arrays that all contain the same type. If types do not match, NumPy will upcast if possible

np.array([3.14, 4, 2, 3])

# if you wish to set the data type,use the dtype keyword

np.array([1, 2, 3, 4], dtype='float64')

# numpy arrays are multidimensional thru nested lists 
 # the i, i+3 is the number of elemnets in each nested list
np.array([range(i,i+4) for i in [1,2,3]])


# Creating arrays from scratch
# can do so using functionalities in numpy 

# create a 15 length float array or int array 
np.ones(15,dtype=float)
np.zeros(10,dtype= int)

# create a floating point array filled with ones
np.ones((3,4) , dtype=float)

# create a floating point or int array filled with custom numbers 
np.full((4,5), 20)  # use full - to fill with single value

# create an array filled with range of numnbers using arange
# this is similar to the built-in range() function
np.arange(0, 20, 3)

#create array of five values equally spaced
np.linspace(0, 1, 5)

#create array of random values between 0 and 1 but you specify the size
np.random.random((4,5))

# Create a 3x3 array of normally distributed random values with mean 0 and standard deviation 1
np.random.normal(0,1,(3,3))

# Create a 3x3 array of random integers between any interval 
np.random.randint(3,10,(4,4))

# Create an identity matrix
np.eye(5)

# Create an uninitialized array of three integers
# The values will be whatever happens to already exist at that memory location
np.empty(3)



### BASIC ARRAY MANIPULATIONS
# Attributes of arrays: Determining the size, shape, memory consumption, and data types of arrays
# Indexing of arrays: Getting and setting the value of individual array elements
# Slicing of arrays: Getting and setting smaller subarrays within a larger array
# Reshaping of arrays: Changing the shape of a given array
# Joining and splitting of arrays: Combining multiple arrays into one, and splitting one array into many ##



np.random.seed(0)  # seed for reproducibility,NumPy's random number generator uses a pseudorandom number generator (PRNG) algorithm. Seeding ensures the PRNG produces the same sequence of numbers.

x1 = np.random.randint(10, size=5)  # One-dimensional array
x2 = np.random.randint(10, size=(4, 4))  # Two-dimensional array
x3 = np.random.randint(10, size=(4, 5, 6))  # Three-dimensional array

#Array attributes are 
# number of dimensions - ndim, 
# size of dimension - shape, 
# total size of the array - size
# data type - dtype
# size in bytes  of each array element -itemsize
# total size (in bytes) of the array - nbytes --- (itemsize * size)
print("x3 ndim: ", x3.ndim)
print("x3 shape:", x3.shape)
print("x3 size: ", x3.size)
print("dtype:", x3.dtype)
print("itemsize:", x3.itemsize, "bytes")
print("nbytes:", x3.nbytes, "bytes")

#Array indexing 
# accessing single elements for 1d array
x1 # consider a 1d array 
x1[0]  # positive indices 
x1[3]
x1[-1] # negative indices 
x1[-2]

#accessing elements of multi dim array 
x2 #a 2d array
x2[0, 0]
x2[3, 2]
x2[2, -1]

#modifying values using indexing 
x2[0, 0] = 10
x2


#Array Slicing to access subarrays 

#1d arrays - like lists ?
x = np.arange(15)
x
x[:7]  # first 7 elements
x[7:]  # elements after index 7
x[4:7]  # middle sub-array
x[::2]  # every other element
x[1::3]  # every other element, starting at index 1
x[::-1]  # all elements, reversed
x[6::-2]  # reversed every other from index 6

#multi dim arrays - 
#2d
x2
x2[:1, :3]  # 1 row, 3 columns
x2[:2, :3] 
x2[:3, ::2]  # all rows, every other column
x2[::-1, ::-1] # reversed at once

#3d
x3[0, 1, 2]  # Access a single element
x3[0, :, :]  # Access 2D slice at x=0
x3[0, 1, :]  # Access 1D slice at x=0, y=1

# Access multiple elements using lists
indices_x = [0, 1]
indices_y = [1, 2]
indices_z = [0, 1]
x3[indices_x, indices_y, indices_z]  

#***array slices is that they return views rather than copies of the array data unlike list slices 
x2
x2_sub = x2[:2, :2]
print(x2_sub)
print(x2)

#Creating copies of arrays
x2_sub_copy = x2[:2, :2].copy()
print(x2_sub_copy)

x2_sub_copy[0, 0] = 42
print(x2_sub_copy)

print(x2)


#Array Reshaping
# using reshape - size of the initial array must match the size of the reshaped array
grid = np.arange(5, 15).reshape((3, 3))
print(grid)

x = np.array([9, 2, 7])

# row vector via reshape
x.reshape(1,3)

# row vector via newaxis
x[np.newaxis, :]

# column vector via reshape
x.reshape((3, 1))

# column vector via newaxis
x[:, np.newaxis]



# Array Concatenation and Splitting
# Concatenation
# using np.concatenate - usually for 1d, sometimes multi dim (takes a tuple or list of arrays as its first argument), 
# np.vstack, np.hstack - for multi dim
#1d arrays
a = np.array([8, 2, 7])
y = np.array([4, 3, 1])
np.concatenate([a, y])

z = [99, 99, 99]
print(np.concatenate([a, y, z])) #concatenate more than two arrays at once:

#2d arrays

grid = np.array([[1, 2, 3],
                 [4, 5, 6]])

np.concatenate([grid, grid])# concatenate along the first axis

np.concatenate([grid, grid], axis=1)# concatenate along the second axis (zero-indexed)

# for multi dim
x = np.array([1, 2, 3])
grid = np.array([[9, 8, 7],
                 [6, 5, 4]])


np.vstack([x, grid])# vertically stack the arrays


y = np.array([[99],
              [99]])
np.hstack([grid, y])# horizontally stack the arrays


#Splitting
# using np.split, np.hsplit, and np.vsplit and need to give split points 
#1d split
x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3, 5])
print(x1, x2, x3)

#multidim split
grid = np.arange(16).reshape((4, 4))
grid

upper, lower = np.vsplit(grid, [2])
print(upper)
print(lower)

left, right = np.hsplit(grid, [2])
print(left)
print(right)