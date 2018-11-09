<p align="center">           

<h1><b>Project 2018</b><br></h1>
<h2>Programming for Data Analysis - Mary McDonagh</h2>
<h2>numpy.random package in Python</h2>
</br>
</p>

![image](https://user-images.githubusercontent.com/36244887/48290610-690ea080-e46b-11e8-85b7-5a05ac318afd.JPG)

## Table of Contents

#### 1.0 Invetigation
#### 2.0 Question 1: Explain the overall purpose of the package.
#### 3.0 Question 2: Explain the use of the "Simple Random Data" and the Permutations functions.
#### 4.0 Question 3: Explain the use and purpose of at least five “Distributions” functions.
#### 5.0 Question 4: Explain the use of seeds in generating pseudorandom numbers.
#### 6.0 Summary
#### 7.0 References


## 1. Investigation

*Write some Python code to investigate numpy.*

Initial steps carried out:
- import required libraries 

Some of the popular libraries used for simple random data anslysis, permutations, distributions and seeds include the following:
- numpy
- Matplotlib
- Seaborn
- Pandas
- Plotly

## Project Plan

![image](https://user-images.githubusercontent.com/36244887/48293640-07086800-e478-11e8-8d98-45742c833038.JPG)

## 2.0 Question 1: Explain the overall purpose of the package

NumPy is the fundamental package for scientific computing with Python. It contains among other things:
- a powerful N-dimensional array object 
- sophisticated (broadcasting) functions
- tools for integrating C/C++ and Fortran code
- useful linear algebra, Fourier transform, and random number capabilities

NumPy array is a central data structure of the numpy library. An array contains information about raw data, how to locate and interpret elements. Numpy is an open source add ón to python and provides fast functions for pre compiled mathematical and numerical routines. The NumPy (Numeric Python) package allows for basic routines for manipulating large arrays and matrices of numeric data.
Python supports real numbers and integers. It allows us to create complicated data structures with lists/sets. It is very easy to write algorithms to solve scientific problems. Python is a dynamic language where almost all functions and operators are polymorphic. Python doesn't understand what has to be done on a hardware level. As a result this rules out any optimisations that can be made by rearranging operations to take advantage of how they are stored in memory and cache.
Polymorphism is one property of Python that causes a problem. Python needs to check the operands of any operator or function to see what type it is, decide whether this particular function can handle the data types, then use the correct form of the operand/function to do the actual operation. Generally this would not be an issue as computers run very fast. Saying this, in many scientific algorithms, this means applying the same operations to thousands or even millions of data points. Numpy comes in very useful at this point. Numpy adds extra overloading functions for the common operators and functions to help optimize uses of arrays. NumPY also uses external standard, optimized libraries written in C or FORTRAN to handle many of the actual manipulations on these array data types. This is handled by libraries like BLAS or lapack. Python simply does an initial check of each of the arrays and then hands them as a single object to the external library. The external library does all of the hard work and outputs a single object containing the result. This removes the need for Python to check each element of code. NumPY module provides an excellent foundation allowing the use of complex scientific workflows.
NumPy is the basic package for scientific computing in Python that provides multi-dimensional arrays and matrices, broadcasting functions, tools for integrating C/C++, Fortran code, mathematical, logical, shape manipulation, sorting, selecting, I/O, useful linear algebra, Fourier transform, random number capabilities, basic statistical operations and much more. NumPy can be used as an efficient multi-dimensional container of generic data and it is licensed under the BSD license. The Mayavi package basically integrates seamlessly with numpy and scipy for 3D plotting and can even be used in IPython interactively, similarly to Matplotlib. The h5py : The h5py package is a Pythonic interface to the HDF5 binary data format that uses straightforward NumPy as well as Python metaphors, such as dictionary and NumPy array syntax. PyTables : PyTables is a package for managing hierarchical datasets as well as designed to efficiently and easily cope with extremely large amounts of data. PyTables is built on top of the HDF5 library, that using Python language and the NumPy package.
One of NumPy’s very important goals is compatibility, which tries to retain all features supported by either of its predecessors. NumPy contains few linear algebra functions, even though these more properly belong in SciPy. SciPy contains more fully-featured versions of the linear algebra modules, and many other numerical algorithms.

## 3.0 Question 2: Explain the use of the "Simple Random Data" and the Permutations functions.


#### Using np.random functions to generate simple data
The examples below outline the use of random functions to generate random numbers based on the paremeters set. Most functions depend on the basic function random() , which generates a random float uniformly in the semi-open range 0.0, 1.0. Python uses the Mersenne twister as the core generator. The Mersenne Twister is one of the most extensively tested random number generators in existence.

Display a table to show the simple random data functions and the description of each.

Input:
#Import the pandas package 
import pandas as pd
data = [['np.random.rand','Generates random values of given shape.'], 
        ['np.random.randint','Returns random integers from low to high.'],
        ['np.randn','Generates a random normal distribution.' ],
        ['np.random.uniform','Sample generated from a uniform distribution.' ],  
        ['np.random.choice','Sample generated from a given 1-D array.'],
       ['np.random.shuffle', 'Modifies the sequence by shuffling its contents.']]
pd.DataFrame(data, columns=["Simple random data", "Description"])

#### Random.random()
This is the basic way to create a random number(s) which will be imported from the random module.

Input:
#Use the random function in numpy (  # Random float:  0.0 <= x < 1.0)
#random.random() returns the next random floating point number in the range [0.0, 1.0).
#a single float will be the output if no argument is defined.
np.random.rand()

Out[5]:
0.27611980460718133

In machine learning we can use libraries such as scikit-learn and Keras. Using these libraries make use of NumPy, which is a very efficient library when working with vectors and matrices of numbers. NumPy also has its own implementation of a pseudorandom number generator and convenience wrapper functions. NumPy also implements the Mersenne Twister pseudorandom number generator.

Input:
#### Create a 1D array
np.random.rand(5)

Output:
array([0.88439232, 0.14191216, 0.87363317, 0.33119369, 0.41399057])

#### Create a 2D array
#Use a simple random function in numpy by setting values to create an array with 3 rows and 2 columns of random data.
np.random.rand(3, 2)

Output:
array([[0.11103511, 0.50725747],
       [0.80666428, 0.37929833],
       [0.62064798, 0.80827445]])
       
       
#### Use a simple random function in numpy by setting values to create an array with 3 rows and 4 columns of random data.
Input:
import numpy as np
np.random.rand(3, 4)

Output:
array([[0.62037717, 0.62329214, 0.81521343, 0.81478059],
       [0.33778278, 0.81295036, 0.94827048, 0.17030383],
       [0.58024849, 0.94107027, 0.15841619, 0.92230211]])
       
#### Create a 3D array
Input:
np.random.rand(3, 2, 2)

Output:
array([[[0.7641413 , 0.35916858],
        [0.05360667, 0.40886018]],

       [[0.35139341, 0.66013656],
        [0.13342679, 0.51347319]],

       [[0.58961022, 0.19239563],
        [0.68682527, 0.66106053]]])
        
#### Create a 3x3 array with random values
Input:
x = np.random.random((3,3,3))
print(x)

Output:
[[[5.84763038e-01 2.02976715e-01 6.87070201e-01]
  [3.65327607e-02 8.11746984e-01 3.48927082e-01]
  [2.11288285e-01 5.39528196e-04 7.88791758e-01]]

 [[6.19415540e-01 1.21836408e-01 1.53446978e-01]
  [4.02598817e-01 2.83738482e-01 4.07223091e-01]
  [7.51015902e-01 4.48356796e-01 4.60034185e-01]]

 [[5.18144234e-01 2.90678555e-01 9.44121761e-01]
  [9.64408666e-01 5.54696322e-01 1.95098184e-01]
  [5.30336471e-01 1.63101966e-01 6.35840077e-01]]]
  
#### Create a 10x10 array with random values and show the minimum and maximum values
Input:
x = np.random.random((10,10))
xmin, xmax = x.min(), x.max()
print(xmin, xmax)

Output:
0.013928171204602657 0.9070049707597532

#### Create a random vector of size 20 and show the mean value
Input:
x = np.random.random(20)
m = x.mean()
print(m)

Output:
0.4811550867638397

#### Swop two rows in an array. Author: Eelco Hoogendoorn
Input:
A = np.arange(25).reshape(5,5)
A[[0,1]] = A[[1,0]]
print(A)

Output:
[[ 5  6  7  8  9]
 [ 0  1  2  3  4]
 [10 11 12 13 14]
 [15 16 17 18 19]
 [20 21 22 23 24]]
 
#### Create a null vector of size 10
Input:
import numpy as np
x = np.zeros(10)
print(x)

Output:
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

#### Update the 5th value to 12
Input:
x[5]=12
print(x)

Output:
[ 0.  0.  0.  0.  0. 12.  0.  0.  0.  0.]

### Reverse a vector making the first element the last
Inp#ut:
x = np.arange(50)
x = x[::-1]
print(x)

Output:
[49 48 47 46 45 44 43 42 41 40 39 38 37 36 35 34 33 32 31 30 29 28 27 26
 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10  9  8  7  6  5  4  3  2
  1  0]
  
#### Set the parameter of 100 to define te dimensions of the array
Input:
x = np.random.rand(100)
x

Output:
array([0.86540077, 0.98276959, 0.03667397, 0.03908545, 0.76306268,
       0.97823441, 0.17552903, 0.35982625, 0.22271826, 0.33536785,
       0.75601985, 0.55331428, 0.94986799, 0.4071542 , 0.16516529,
       0.60760606, 0.11271274, 0.46205383, 0.1317771 , 0.04162354,
       0.48710629, 0.31975592, 0.93366526, 0.90606962, 0.65587773,
       0.54667384, 0.71973796, 0.16577249, 0.18804313, 0.86101582,
       0.46772639, 0.79351625, 0.57349306, 0.75739267, 0.40925102,
       0.49567829, 0.68850439, 0.1559254 , 0.98412105, 0.86607902,
       0.97613959, 0.83295452, 0.71201569, 0.1956789 , 0.6733361 ,
       0.91514592, 0.65658572, 0.0613386 , 0.23615896, 0.85161728,
       0.89197691, 0.78387653, 0.8548011 , 0.85412558, 0.52123363,
       0.68388397, 0.07715682, 0.76617622, 0.58191623, 0.60169632,
       0.88423461, 0.20013454, 0.93410184, 0.40433598, 0.61056597,
       0.64575524, 0.2727244 , 0.67405491, 0.48842212, 0.43544837,inset
       0.28786624, 0.36944197, 0.55099399, 0.43400621, 0.65408598,
       0.79803034, 0.14980369, 0.18371265, 0.08011524, 0.09417568,
       0.24887452, 0.38511078, 0.6897537 , 0.13251993, 0.74806729,
       0.82086757, 0.38827852, 0.25595118, 0.57871643, 0.55606781,
       0.18724072, 0.25552766, 0.43429069, 0.31390278, 0.69675783,
       0.8611739 , 0.24111164, 0.10220595, 0.93824736, 0.43383612])
       
#### Import matplotlib to plot a graph
Input:
#Import matplotlib package
import matplotlib.pyplot as plt
import numpy as np
x = plt.hist(np.random.rand(100))
plt.show() 


Using the matplotlib package above allows for the creation of a graph. It has plotted the random data generated from the .rand function. The random numbers are generated across the plot.

#### Normalise a 5x5 random matrix
Input:
x = np.random.random((5,5))
xmax, xmin = x.max(), x.min()
x = (x - xmin)/(xmax - xmin)
print(x)

Output:
[[0.57223788 1.         0.66306175 0.         0.66262592]
 [0.47921363 0.8544876  0.43021538 0.09753426 0.69014733]
 [0.47824371 0.30896032 0.40947359 0.24555078 0.76279143]
 [0.07421018 0.64466399 0.8477277  0.32922389 0.42274363]
 [0.94338165 0.52829323 0.48272391 0.2286424  0.91006357]]

#### Create a random array of size 10 and sort it
Input:
x = np.random.random(10)
x.sort()
print(x)

Output:
[0.03104232 0.25998345 0.27770799 0.27902061 0.32969548 0.62796397
 0.72135428 0.76010992 0.78164068 0.91131259]
 
x = np.random.rand
Input:
np.random.rand(3, 4) + 3

Output:
array([[3.25912672, 3.86000573, 3.64816683, 3.03381164],
       [3.30956289, 3.93256392, 3.02405371, 3.0050986 ],
       [3.39671306, 3.22835336, 3.10635108, 3.98686132]])

#### Using Randint

#### Use a simple random randint function in numpy to list an array of integers.
#### Display in the range of 0-10 with 20 digits displayed.
Input:
np.random.randint(0, 10, 20)

array([3, 9, 1, 3, 4, 9, 6, 9, 4, 7, 1, 1, 6, 1, 4, 7, 2, 2, 8, 9])

#### Use a simple random function to list an array of random integers in the range 1-7 with 10 rows and 3 columns.
Input:
x = np.random.randint(1, 7, (10,3))
x

Output:
array([[3, 3, 5],
       [5, 5, 2],
       [4, 4, 6],
       [2, 5, 6],
       [5, 2, 1],
       [6, 6, 4],
       [2, 5, 1],
       [2, 5, 2],
       [2, 3, 6],
       [1, 4, 6]])
       
#### x was defined in the previous input. Now I will use shape to define how many rows and columns exist in x.
Input:
x.shape

Output:
(10, 3)


x = np.sum(x, axis=2)

#### Plot a histogram to display x output.
plt.hist(x);

insert histogram graph

#### Calculate the sum of x.
Input:
np.sum(x)

Output:
104

#### Calculate the sum of the rows
np.sum(x, axis=0)

array([47, 29, 47])

#### Calclate the sum of the columns
np.sum(x, axis=1)

array([16,  9, 10, 13, 14, 14, 10, 12, 12, 13])

y = np.sum(x, axis=1)
y

array([16,  9, 10, 13, 14, 14, 10, 12, 12, 13])

### Data Visualisation
![image](https://user-images.githubusercontent.com/36244887/48292682-35d00f80-e473-11e8-8859-550f6f56c70f.JPG)

Matplotlib produces publication-quality figures in a variety of formats, and interactive environments across Python platforms. Another advantage is that Pandas comes equipped with useful wrappers around several matplotlib plotting routines, allowing for quick and handy plotting of Series and DataFrame objects.
The matplotlib library supports a large number of plot types useful for data visualization. Some common types of of plot types include: scatter plots, bar plots, contour plots, and histograms.
A scatter plot is used to visualize the relationship between variables measured in the same dataset. It is easy to plot a simple scatter plot, using the plt.scatter() function, that requires numeric columns for both the x and y axis:


Input:
#### import matplotlib to plot a graph
import matplotlib.pyplot as plt
X = np.random.normal(0, 1, 1000)
Y = np.random.normal(0, 1, 1000)
plt.scatter(X, Y, c = ['b', 'g', 'k', 'r', 'c'])
plt.show()

Output:
insert scatter2 graph

Input:
#### import matplotlib to plot a graph
import matplotlib.pyplot as plt
plt.hist(x)
plt.show() 

Input:
import matplotlib.pyplot as plt
x = np.random.randint(1,7, (100,10))
y = np.sum(x, axis=1)
plt.hist(y);

Input:
x = np.random.randint(1,7, (100000,10))
y = np.sum(x, axis=1)
plt.hist(y);

#### Using Randn
Using randn to return a random sample for the normal distribution
randn will generate a random sample from the normal distributionn set of data. A normal distribution has a mean of 0 and a standard deviation of 1.

Input:
#### Return a sample for the standard normal distribution. n referring to the normal distribution.
np.random.randn(3)

Output:
array([0.61750326, 0.29389352, 0.20851485])

Input:
#### Return random integers from low (inclusive) to high(exclusive).
Return random integers from the “discrete uniform” distribution of the specified dtype in the “half-open” interval [low, high). If high is None (the default), then results are from [0, low).
np.random.randint(5, size=20)

Output:
array([4, 2, 2, 4, 1, 0, 0, 2, 0, 0, 1, 2, 2, 3, 4, 3, 3, 0, 1, 3])

Input:
#### Randint creating a 5 by 4 array of ints between 0 and 4: 
np.random.randint(5, size=(5, 4))

Output:
array([[0, 1, 4, 3],
       [3, 3, 2, 4],
       [3, 1, 2, 0],
       [2, 0, 4, 1],
       [2, 2, 1, 3]])
       
Input:
import matplotlib.pyplot as plt
x = np.arange(0.0, 10.0, 0.01)
y = 3.0 * x + 1.0
noise = np.random.normal(0.0, 1.0, len(x))

plt.plot(x, y + noise, 'r.')
plt.plot(x, y, 'b-')

plt.show

Input:
#### import matplotlib for graphs
import matplotlib.pyplot as plt 

x = np.arange(0.0, 10.0, 0.01)
y = 3.0 * x + 1.0
noise = np.random.normal(0.0, 1.0, len(x))

plt.plot(x, y + noise, 'r.', label="Model")
plt.plot(x, y, 'b-', label="Actual")

plt.title("Sample Plot")
plt.xlabel("Weight")
plt.ylabel("Mass")
plt.legend()

plt.show


#### Plot 2 graphs 
Input:
plt.subplot(1, 2, 1)
x =  np.random.normal(0.0, 10.0, 10000)
plt.hist(x)
plt.subplot(1, 2, 2)
x =  np.random.uniform(-20, .2, 2000)
plt.hist(x)
plt.show

#### Using Uniform
random.uniform(a, b) Return a random floating point number N such that a <= N <= b for a <= b and b <= N <= a for b < a.~
The end-point value b may or may not be included in the range depending on floating-point rounding in the equation a + (b-a) * random().
Essentially, random.uniform() is used to generate a random number between two numbers other than zero and 1. Using random.uniform we must specify the low and high numbers.

Input:
#### generates random numbers from a uniform distribution (lower, higher, size)
np.random.uniform(-10, 8, 6)

Output:
array([-4.49293858,  7.20663687,  4.0094406 , -9.53500723,  4.99508228,
       -0.57488354])
       
#### Using Choice
#### Using choice within a random sample

Input:
#### Generate a random sample within the range 1-5
np.random.choice(range(1,5))

Output:
3


Input: 
#### Generates a random sample from a given 1-D array. Choose 3 random numbers between 1 & 5.
np.random.choice(5 , 3)

Output:
array([1, 4, 2])

Input: 
#### Generate randomly from either M or O.
np.random.choice(["M", "O"])

Output:
'O'

Input: 
#### Create a 2d array with 1 on the border and 0 on the inside
x = np.ones((10,10))
x[1:-1,1:-1] = 0
print(x)

Output:
[[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]
 
#### Using Shuffle
#### Using the shuffle function to shuffle the random array

Input:
#### Modify an array sequence by shuffling its contents. Shuffles the first axis of a multi-dimensional array.
arr = np.arange(10)
np.random.shuffle(arr)
arr

Output:
array([6, 1, 9, 0, 2, 8, 7, 5, 3, 4])

Input:
#### Create a 4x4 arrange of shuffled values
arr = np.arange(16).reshape((4, 4))
np.random.shuffle(arr)
arr

Output:
array([[ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [ 0,  1,  2,  3],
       [12, 13, 14, 15]])
       
#### Simple Statistics

I can use python and numpy as a tool to describe my sets of data in the following ways:
- Max/Min
- Length
- Mean
- Standard Deviation

Input:
import numpy as np
my_list = [12, 6, 13, 5, 4, 8, 3, 9, 15]

#### Displays the min and max values from my list
print("min: ", min(my_list))
print("max: ", max(my_list))

Output:
min:  3
max:  15

Input:
print("length: ", len(my_list))

Output:
length:  9

Input:
#### Displays the mean.
def mean(x):
    return sum(x) / len(x)

print("average: ", mean(my_list))

Output:
average:  8.333333333333334

Input:
#### Displays the standard deviation.
print("Standard Deviation: ", np.std(my_list))

Output:
Standard Deviation:  4.0

### 2. Explain the use of Permutation Functions
A permutation is the arrangement of a set of items in different order. Using permutation functions allows for the random permutation of a sequence, or to return a permuted range. First import itertools package to implement permutations method in python. This method takes a list as an input and return an object list of tuples that contain all permutation in a list form. Permutation is an arrangement of objects in a specific order. Order of arrangement of object is very important. The number of permutations on a set of n elements is given by n!. For example, there are 2! = 21 = 2 permutations of {1, 2}, namely {1, 2} and {2, 1}, and 3! = 32*1 = 6 permutations of {1, 2, 3}, namely {1, 2, 3}, {1, 3, 2}, {2, 1, 3}, {2, 3, 1}, {3, 1, 2} and {3, 2, 1}.

Input:
#### import permutations
#### list items in the range of 1-4 and print
from itertools import permutations 
x = list(permutations(range(1, 4))) 
print(x) 

Output:
[(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]

We can also rearrange characters in a word to create other words. If all the n characters are unique, we should get n! unique permutations. We can make a list of words unique by converting it to a set.

Input:
#### re-arrange charaters in the word dog to create other words
from itertools import permutations 
for x in permutations('dog'):
    print("".join(x))
    
Output:
[(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1


We can also rearrange characters in a word to create other words. If all the n characters are unique, we should get n! unique permutations. We can make a list of words unique by converting it to a set.

Input:
#### re-arrange charaters in the word dog to create other words
from itertools import permutations 
for x in permutations('dog'):
    print("".join(x))
    
Output:
[(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]


We can also rearrange characters in a word to create other words. If all the n characters are unique, we should get n! unique permutations. We can make a list of words unique by converting it to a set.

Input:
#### Re-arrange charaters in the word dog to create other words
from itertools import permutations 
for x in permutations('dog'):
    print("".join(x))
    
Output:
dog
dgo
odg
ogd
gdo
god

Input:
np.random.permutation(6)

Output:
array([0, 3, 5, 4, 2, 1])

Input:
np.random.permutation([4, 7, 8, 2, 1])

Output:
array([4, 1, 8, 2, 7])

Input:
#### Generates the permutations for all of the elements of the set input (abcd)
from itertools import permutations 
["".join(a) for a in permutations("abcd", 2)]

Output:
['ab', 'ac', 'ad', 'ba', 'bc', 'bd', 'ca', 'cb', 'cd', 'da', 'db', 'dc']

Input:
sample = np.random.permutation(9)
print("Sample = ", sample)

Output:
Sample =  [8 3 6 1 7 5 4 0 2]

Input:
#### Shuffle the sample from above output
shuffled = np.random.permutation(sample)
print("Permuted = ",shuffled)

Output:
Permuted =  [4 0 6 2 5 3 8 7 1]

Input:
#### Example of reshuffling an array
np.random.permutation([1, 78, 39, 2, 5])

Output:
array([ 5, 78,  1,  2, 39])

Input:
#### Create a multi dimensional array
arr = np.arange(9).reshape((3, 3))
arr

Output:
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])
       
Input:
#### Example of reshuffling a multidimensional array 
np.random.permutation(arr)

Output:
array([[6, 7, 8],
       [0, 1, 2],
       [3, 4, 5]])
       
Input:
#### Create a 3x3 matrix with numbers ranging from 3 to 10
x= np.arange(2, 11).reshape(3,3)
print(x)

Output:
[[ 2  3  4]
 [ 5  6  7]
 [ 8  9 10]]
 
### Linear Algebra
#### Using numpy's linear algebra function to do simple maths on an array. 
 
Input:
x = np.array([2,3,4], float)
y = np.array([5,6,7], float)
print (x)
print (y)

Output:
[2. 3. 4.]
[5. 6. 7.]

Input:
print(x+y)

Output:
[ 7.  9. 11.]

Input:
print(x-y)

Output:
[-3. -3. -3.]

Input:
print(x*y)

Output:
[10. 18. 28.]

Input:
print(y/x)

Output:
[2.5  2.   1.75]

Input:
print(y ** x)

Output:
[  25.  216. 2401.]

### Question 3: Explain the use and purpose of at least five “Distributions” functions.

A probabillity disctibution is a mathematical function that can provide the probability of occurence of different outcomes.
-Continuous Distributions e.g normal (every normal density is non zero for all real numbers) -Multivariate Distributions -Discrete Distributions -Statistical Distributions -Contingency table functions

Types of Distributions: -Binominal (is a discrete distribution having two parameters viz. sample size (n) and probability of success (p).) -Poisson (used to model events which take place continuously and at any given time independent of each other) -Normal (s a continuous distribution. It has the shape of a bell shaped curve.) -Exponential () -Uniform

#### Binomial distribution
Binomial distribution is the discrete probability distribution of the number of successes in a sequence of n independent binary (yes/no) experiments, each of which yields success with probability p. Such a success/failure experiment is also called a Bernoulli experiment or Bernoulli trial. In fact, when n = 1, the binomial distribution is a Bernoulli distribution. This draws samples from a binomial distribution.

Input:
#### numpy.random.binomial(n,p, size=none)
#### Draw samples from a binomial distribution
n,p = 10, .5 # The number of trials, probability of each trial
x = np.random.binomial(n, p, 1000)

Output:
#### 20,000 trials of the model, and count the number that generate zero positive results.
sum(np.random.binomial(9, 0.1, 20000) == 0)/20000.

Output:
0.38575

### Poisson distribution
Poisson distribution is a discrete probability distribution that expresses the probability of a number of events occurring in a fixed period of time if these events occur with a known average rate and independently of the time since the last event. For example, if you receive 3 calls on average between 8am-5pm each day, then the number of calls you will receive tomorrow between 8am-5pm should follow a Poisson distribution with parameter 3 λ = . This is under the assumption that the chance to receive a call at any time point between 8am-5pm is the same.

Input:
#### numpy.random.poisson(lam=1.0, size=None)
#### Draw samples from a Poisson distribution. The Poisson distribution is the limit of the binomial distribution for large N.
np.random.poisson(3, 100)

Output:
array([5, 2, 4, 1, 1, 5, 4, 4, 2, 2, 2, 2, 5, 4, 2, 6, 5, 2, 6, 3, 4, 2,
       4, 3, 1, 2, 0, 4, 5, 1, 3, 1, 4, 1, 3, 2, 2, 5, 0, 2, 0, 2, 2, 1,
       4, 8, 3, 2, 1, 6, 2, 3, 5, 2, 2, 5, 5, 3, 1, 3, 4, 3, 3, 4, 5, 4,
       1, 2, 3, 4, 4, 5, 3, 3, 1, 4, 0, 5, 2, 2, 3, 2, 1, 3, 1, 2, 3, 2,
       4, 2, 4, 3, 2, 3, 0, 4, 3, 5, 5, 6])
       
       
Input:
import matplotlib.pyplot as plt
    count, bins, ignored = plt.hist(12, density=True)
    plt.show()

Output:
insert hist144

### Exponential distributions
Exponential distributions are a class of continuous probability distribution. An exponential distribution arises naturally when modeling the time between independent events that happen at a constant average rate. For example, if you receive 3 calls on average between 8am-5pm each day, then the hours you wait for the first call since 8am tomorrow should follow an exponential distribution with parameter 3 /9 1/3 calls hrs λ = = . The average time you wait for the new call since last call is the expectation of the distribution: 1/ 3hrs λ = 

### Uniform Distribution
#### Discrete uniform distribution
Discrete uniform distribution is a discrete probability distribution: If a random variable has any of n possible values k1, k2, …, kn that are equally probable, then it has a discrete uniform distribution. The probability of any outcome ki is 1/n. A simple example of the discrete uniform distribution is throwing a fair die. The possible values of k are 1, 2, 3, 4, 5, 6; and each time the die is thrown, the probability of a given score is 1/6. 

#### Continuous uniform distribution
Continuous uniform distribution is a family of probability distributions such that for each member of the family, all intervals of the same length on the distribution's support are equally probable. The support is defined by the two parameters, a and b, which are its minimum and maximum values. The distribution is often abbreviated U(a,b), e.g., U(0,1) is a member of this family and so is U(1,2).

Input:
#### Draw samples from the distribution
np.random.uniform(-1,0,1000)

Input:
#### All values are within the given interval
np.all(s >= -1)
True
    np.all(s < 0)
True

Output:
True

### Normal Distribution
A Normal Distribution (Gaussian) is a continuous probability distribution. The normal distribution is often referred to as a bell curve. Properties of a normal distribution are outlined below:
- The normal curve is symmetric about the mean and bell shaped.
- Mean, mode and median is zero and positioned at the centre of the curve.
- Approximately 68% of the data will be between -1 and +1 (i.e. within 1 standard deviation from the mean), 95% between -2 and +2 (within 2 SD from the mean) and 99.7% between -3 and 3 (within 3 SD from the mean)


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

Input:
#Get median, mean, standard deviation and var
print(norm.median(), norm.mean(), norm.std(), norm.var())
Output:
0.0 0.0 1.0 1.0

### Beta Distribution
Beta distribution is well known distribution for probabilities. Beta distribution is a continuous distribution taking values from 0 to 1. Alpha and Beta are the two parameters which define it. Depending on the values of alpha and beta, the distributions can differ.

### Question 4: Explain the use of seeds in generating pseudorandom numbers.

'random.seed' gives a value to random value generator ('random.randint()') which generates these values based on this seed. A property of random numbers is that they should be reproducible. Once I use the same seed I get the same pattern of random numbers in the examples below. random is actually pseudo-random. Given a seed, it will generate numbers with an equal distribution. Using the same seed, it will generate the same number sequences every time. In order to change it, I would have to change my seed. Seed is often used based on the current time.
Class Random can be subclassed in order to use a different basic generator: override the random(), seed(), getstate(), setstate() and jumpahead() methods.

Input:
random.seed(100)
random.randint(1, 10)

Output:
3

Input:
#### Using seed will allow me to produce the same random integer each time
random.seed(100)
random.randint(1, 10)

Output:
3

Input:
#### Same results as above
random.seed(100)
random.randint(1, 10)

Output:
3

Input:
#### Result has now changed as I have changed the see number to 101
random.seed(101)
random.randint(1, 10)

Output:
10

Input:
#### Same results as above
random.seed(101)
random.randint(1, 10)

Output:
10

Input:
#### prints 6 random numbers between 1 and 50
random.seed(10)
for i in range(6):
        print(random.randint(1,50))
        
Output:
37
3
28
31
37
1

Input:
#### Prints the same output as above as it is the same seed
random.seed(10)
for i in range(6):
        print(random.randint(1,50))
        
Output:
37
3
28
31
37
1

Input:
#### Change the seed to change the output random integer values
random.seed(20)
for i in range(6):
        print(random.randint(1,50))

Output:
47
44
50
10
17
44

Input:
#### use seed to create a random number
random.seed(30)
random.random()

Output:
0.5390815646058106

Input:
#### Do not use seed
random.random()

Output:
0.2891964436397205

Input:
#### Use seed again and it will output the same random number as previous seed(30) above
random.seed(30)
random.random()

Output:
0.5390815646058106


### 6.0 Summary


### 7.0 References
https://www.packtpub.com/mapt/book/big_data_and_business_intelligence/9781785285110/2/ch02lvl1sec16/numpy-random-numbers
https://machinelearningmastery.com/how-to-generate-random-numbers-in-python/
https://www.investopedia.com/terms/s/simple-random-sample.asp
https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.uniform.html
https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.choice.html
https://github.com/RitRa/Numpy_random/blob/master/numpy-random.ipynb
https://www.quora.com/Probability-statistics-What-is-difference-between-binominal-poisson-and-normal-distribution
https://www.stat.berkeley.edu/~hhuang/STAT141/More_on_%20distributions.pdf
https://www.healthknowledge.org.uk/public-health-textbook/research-methods/1b-statistical-methods/statistical-distributions
https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.binomial.html
https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.uniform.html
https://stackoverflow.com/questions/22639587/random-seed-what-does-it-do
https://www.w3resource.com/python-exercises/numpy/index.php
https://www.datasciencedata.com/2018/09/using-python-numpy-random-modules.html
https://www.geeksforgeeks.org/numpy-random-rand-python/
https://docs.python.org/2/library/random.html
http://cmdlinetips.com/2018/03/probability-distributions-in-python/
https://medium.com/@balamurali_m/normal-distribution-with-python-793c7b425ef0
http://pytolearn.csd.auth.gr/d1-hyptest/11/norm-distro.html
