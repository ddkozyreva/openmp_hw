
#!/bin/bash
g++ -O3 -fopenmp 2.3b-fourier.cpp -o filename
for (( num_threads=1; num_threads <= 8; num_threads++ ))
do
    export OMP_NUM_THREADS=$num_threads; ./filename < input.txt
done