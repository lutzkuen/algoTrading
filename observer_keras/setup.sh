#cp controller.py controller_cython.pyx
#cython -X language_level=3 controller_cython.pyx
#gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python3.6m -o controller_cython.so controller_cython.c

cp estimator_keras.py estimator_keras_cython.pyx
cython -X language_level=3 estimator_keras_cython.pyx
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python3.6m -o estimator_keras_cython.so estimator_keras_cython.c
