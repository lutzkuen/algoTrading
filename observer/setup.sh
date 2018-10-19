cp controller.py controller_cython.pyx
cython controller_cython.pyx
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -o controller_cython.so controller_cython.c
