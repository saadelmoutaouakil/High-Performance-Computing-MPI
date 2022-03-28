This project is the implementation of a scalable parallel version of the Lattice Boltzmann algorithm using MPI.  
The starting point is a serially optimised implementation in C.  
The experiments are designed to be run in the Blue Crystal Supercomputer V4 with 4 nodes, each with 2 sockets yielding a total of 112 cores.  
For a better software/hardware compatibility, the Intel MPIICC compiler will be used with the flags -Ofast and -xAVX -qopenmp.  
