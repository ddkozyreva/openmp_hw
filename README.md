# Практикум по суперкомпьютерному моделированию

## Запуск кода

- Для начала подключимся к кластеру hse и в терминале введем команду ниже - ею мы зарезервируем 8 ядер и впоследствии не будем ждать, когда же освободятся нужные для запуска сpu. Proj_1339 был обнаружен с помощью команды `mp`, отражающей, какие квоты (проекты) на суперкомпьютер есть.

```bash
srun -A proj_1339 --pty --cpus-per-task=8 bash
```

- Следующий шаг - запустить скрипт run.sh (создан для удобства). Он компилирует исполняемый файл filename из кода файла *.cpp и затем в цикле прогоняет его на разном количестве ядер - от 1 до 8. Ввод перенаправлен и сделан из файла input.txt.

```bash
#!/bin/bash
echo "2.2a"
g++ -O3 -fopenmp 2.2a-buffon_pi.cpp -o filename
echo "With different thread's number"
for (( num_threads=1; num_threads <= 8; num_threads++ ))
do
    echo "number of cpu's:"
    echo $num_threads
    export OMP_NUM_THREADS=$num_threads
    echo "42000000" | ./filename
done

echo "With different input number"
for (( numbers=1000; numbers <= 1000000000; numbers*=10 ))
do
    echo "Input number is $numbers"
    export OMP_NUM_THREADS=8
    echo $numbers | ./filename
done


echo "2.3b"
g++ -O3 -fopenmp 2.3b-fourier.cpp -o filename
for (( num_threads=1; num_threads <= 8; num_threads++ ))
do
    export OMP_NUM_THREADS=$num_threads; ./filename < input.txt
done
```
- По окончании работы на выделенных ядрах использую команду `exit`
____
## Теория
### Используемые классы переменных
|  | |
-|-|
|SHARED| Применяется к переменным, которые необходимо сделать общими.|
|PRIVATE|Применяется к переменным, которые необходимо сделать приватными. При входе в параллельную область для каждой нити создается отдельный экземпляр переменной, который не имеет никакой связи с оригинальной переменной вне параллельной области. |
| NOWAIT | Параметр для отмены синхронизации при завершении директивы. |

____
## 2.2а.
### Условие
Задача оценивает число π методом Бюффона: генерируется отрезок длины 1, середина
которого находится в начале координат, повернутый под случайным углом, и горизонтальная
прямая y=y0, где y0 случайно равномерно распределена на интервале [-1; 1). Вероятность
пересечения отрезка с этой прямой равна 1/π (см.
https://ru.wikipedia.org/wiki/Задача_Бюффона_о_бросании_иглы). Параллельная версия плохо
масштабируется, а также результат не сходится к теоретическому значению при увеличении
числа сгенерированных случайных чисел. Исправить функцию mc_pi.

- Объяснить и исправить ошибки, допущенные при параллелизации задачи (1 балл
за каждый номер).
- Определить размер входных данных, при котором в однопоточном режиме время
выполнения составляет 5-10 секунд. Убедиться, что параллельная версия
ускоряется на числе потоков больше двух, а также выдает одинаковый результат
при любом числе потоков. (0.5 балла за каждую задачу).
### Решение

_Исходная функция_

```cpp
double mc_pi(ptrdiff_t niter, size_t seed)
{
    double num_crosses = 0;
    std::mt19937_64 rng(seed);
    std::normal_distribution<> rand_nrm(0.0, 1.0);
    std::uniform_real_distribution<double> rand_un(-1.0, 1.0);
#pragma omp parallel
    {
        #pragma omp for reduction(+: num_crosses)
        for (ptrdiff_t i = 0; i < niter; ++i)
        {
            // generate a unit vector with a uniform rotation
            double x = rand_nrm(rng), y = rand_nrm(rng);
            double l = std::hypot(x, y);
            
            y *= 0.5 / l;

            // check if a horizontal line crosses a needle
            double y_line = rand_un(rng);
            num_crosses += std::abs(y_line) < std::abs(y);
        }
    }
    // p = 2L / (r * pi) = 1 / pi if 2L = r
    // r is width of uniform distribution (2 if it is from -1 to 1)
    // L is length of the needle (1 in our case)
    double pi_est = niter / num_crosses;
    return pi_est;
}
```
_Исправленная функция_

```cpp
double mc_pi(ptrdiff_t niter, size_t seed)
{
    double num_crosses = 0, x, l, y, y_line;
    #pragma omp parallel
    {
        std::mt19937_64 rng(seed + omp_get_thread_num());
        std::normal_distribution<> rand_nrm(0.0, 1.0);
        std::uniform_real_distribution<double> rand_un(-1.0, 1.0);
        #pragma omp for private(x, l, y, y_line) reduction(+: num_crosses) nowait
        for (ptrdiff_t i = 0; i < niter; ++i)
        {
            // generate a unit vector with a uniform rotation
            x = rand_nrm(rng);
            y = rand_nrm(rng);
            l = std::hypot(x, y);
            
            y *= 0.5 / l;

            // check if a horizontal line crosses a needle
            y_line = rand_un(rng);
            num_crosses += std::abs(y_line) < std::abs(y);
        }
    }
    // p = 2L / (r * pi) = 1 / pi if 2L = r
    // r is width of uniform distribution (2 if it is from -1 to 1)
    // L is length of the needle (1 in our case)
    double pi_est = niter / num_crosses;
    return pi_est;
}
```

_Что произошло_

1) Генераторы были внесены в тело директивы `#pragma omp parallel` и поправлен seed для ликвидации одинаковых генераторов в разных потоках. Таким образом, ошибки вычисления были убраны и с увеличением входных данных π аппроксимировалось лучше.

2) Директива `#pragma omp for reduction(+: num_crosses)` была заменена на `#pragma omp for private(x, l, y, y_line) reduction(+: num_crosses) nowait`. Главное изменение здесь касается добавление `nowait` - с ним многопоточная версия стала работать быстрее.
___
## 2.3б. 
### Условие

(fourier.cpp) Программа вычисляет косинусное дискретное преобразование Фурье
массива по наивному алгоритму с квадратичной сложностью. При увеличении числа потоков
результат отличается от OMP_NUM_THREADS=1. Исправить функцию fourier

- Объяснить и исправить ошибки, допущенные при параллелизации задачи (1 балл
за каждый номер).
- Определить размер входных данных, при котором в однопоточном режиме время
выполнения составляет 5-10 секунд. Убедиться, что параллельная версия
ускоряется на числе потоков больше двух, а также выдает одинаковый результат
при любом числе потоков. (0.5 балла за каждую задачу).
### Решениe

_Исходная функция_

```cpp
void cosine_dft(vec<double> f, vec<double> x)
{    
    ptrdiff_t i, k, nf = f.length(), nx = x.length();
    double omega = 0;
    #pragma omp parallel
    {
        for (k=0; k < nf; k++)
        {
            f(k) = 0;
        }
        #pragma omp for
        for (i=0; i < nx; i++)
        {
            for (k = 0; k < nf; k++)
            {
                omega = 2 * PI * k / nx;
                f(k) += x(i) * cos(omega * i);
            }
        }
    }
}
```
_Исправленная функция_

```cpp
void cosine_dft(vec<double> f, vec<double> x)
{
    // f, x, nf, nx omega по умолчанию shared
    // i, k по умолчанию private, так как внешний цикл для pragma
    ptrdiff_t i, k, nf = f.length(), nx = x.length();
    double omega = 0;
    #pragma omp parallel
    for (k=0; k < nf; k++)
        f(k) = 0;

    #pragma omp parallel
    for (i=0; i < nx; i++) {  
        #pragma omp for private(omega) nowait 
        for (k = 0; k < nf; k++) {
            omega = 2 * PI * k / nx;
            f(k) += x(i) * cos(omega * i);
        }
    }
}
```

_Что произошло_

1) Директива `#pragma omp parallel` была разделена на два отдельных блока, так по логике функции сначала нам нужно приравнять

2) Директива `#pragma omp for` была дополнена до `#pragma omp for private(omega) nowait`, где `omega` была объявлена как приватная переменная (то есть для каждого процесса будет создаваться копия данной переменной и использоваться именно она), `nowait` для того, чтобы не ждать когда все параллельные итерации завершатся.