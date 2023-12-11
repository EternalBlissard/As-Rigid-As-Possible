There are 2 files in this folder start.cpp and start1.cpp

Start.cpp is my implementation of the algorithm 

Start1.cpp is library implementation of the algorithm I found relevant to my work for comparisons

Follow these steps:

Ensure the same path is being used to define img and image in start.cpp and image in start1.cpp and check their respective path if 
you opt to keep them in the main folder please add '../' to the image name '.format' and vice versa for others 

mkdir build
cd build
cmake ..
make

2 executables will be generated DisplayImage-for my implementation and 
DisplayImage1 - for library implementation.
