
#!/bin/bash
echo "2.3b"
g++ -O3 -fopenmp 2.3b-fourier.cpp -o filename
for (( num_threads=1; num_threads <= 8; num_threads++ ))
do
    export OMP_NUM_THREADS=$num_threads; ./filename < input.txt
done

echo "2.2a"
g++ -O3 -fopenmp 2.2a-buffon_pi.cpp -o filename
for (( num_threads=1; num_threads <= 8; num_threads++ ))
do
    echo "number of cpu's:"
    echo $num_threads
    export OMP_NUM_THREADS=$num_threads
    echo "42000000" | ./filename
done

echo "____"
for (( numbers=1000; numbers <= 1000000000; numbers*=10 ))
do
    echo "Input number is $numbers"
    export OMP_NUM_THREADS=8
    echo $numbers | ./filename
done