#!/bin/bash
#SBATCH --job-name=hw2                    # Название задачи
#SBATCH --error=output/results-%j.err     # Файл для вывода ошибок
#SBATCH --output=output/results.txt       # Файл для вывода результатов

#SBATCH --account=proj_1339        # Название проекта (курса)
#SBATCH --ntasks=1                 # Количество MPI процессов
#SBATCH --nodes=1                  # Требуемое кол-во узлов
#SBATCH --gpus=0                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=32         # Требуемое кол-во CPU
#SBATCH --constraint="type_a|type_b|type_c|type_d"
#SBATCH --time=2

g++ -O3 -fopenmp laplace2d.cpp -o filename
echo "3000 2000" | srun ./filename
for n_threads in 2 4 8 16 32; 
do
    echo "CPUs: $n_threads"
    export OMP_NUM_THREADS=$n_threads
    echo "3000 2000" | srun ./filename
done