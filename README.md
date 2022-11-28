# Практикум по суперкомпьютерному моделированию

## Запуск кода

- Для начала подключимся к кластеру hse и в терминале введем команду ниже - ею мы зарезервируем 8 ядер и впоследствии не будем ждать, когда же освободятся нужные для запуска сpu.

```bash
srun -A proj_1339 --pty --cpus-per-task=8 bash
```

- Следующий шаг - запустить скрипт run.sh (создан для удобства). Он компилирует исполняемый файл filename из кода файла *.cpp и затем в цикле прогоняет его на разном количестве ядер - от 1 до 8. Ввод перенаправлен и сделан из файла input.txt.

```bash
#!/bin/bash
g++ -O3 -fopenmp 2.3b-fourier.cpp -o filename
for (( num_threads=1; num_threads <= 8; num_threads++ ))
do
    export OMP_NUM_THREADS=$num_threads; ./filename < input.txt
done
```
- По окончании работы на выделенных ядрах использую команду `exit`
## 2.2а.
## Условие
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
## Решение

___
## 2.3б. 
## Условие

(fourier.cpp) Программа вычисляет косинусное дискретное преобразование Фурье
массива по наивному алгоритму с квадратичной сложностью. При увеличении числа потоков
результат отличается от OMP_NUM_THREADS=1. Исправить функцию fourier

- Объяснить и исправить ошибки, допущенные при параллелизации задачи (1 балл
за каждый номер).
- Определить размер входных данных, при котором в однопоточном режиме время
выполнения составляет 5-10 секунд. Убедиться, что параллельная версия
ускоряется на числе потоков больше двух, а также выдает одинаковый результат
при любом числе потоков. (0.5 балла за каждую задачу).
## Решениe

### Используемые классы переменных
|  | |
-|-|
|SHARED| Применяется к переменным, которые необходимо сделать общими.|
|PRIVATE|Применяется к переменным, которые необходимо сделать приватными. При входе в параллельную область для каждой нити создается отдельный экземпляр переменной, который не имеет никакой связи с оригинальной переменной вне параллельной области. |
| NOWAIT | Параметр для отмены синхронизации при завершении директивы. |

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

2) Директива `#pragma omp for` была дополнена до `#pragma omp for private(omega) nowait`, где `omega` была объявлена как приватная переменная (то есть для каждого процесса будет создаваться копия данной переменной и сипользоваться именно она), `nowait` для того, чтобы не ждать когда все параллельные итерации завершатся