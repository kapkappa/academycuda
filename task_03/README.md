# Отчет по третьему заданию

В качестве дивергенции добавлен цикл for - каждая нить обрабатывает не один, а несколько элементов вектора (параметр задан константой), а также if-else инструкция, где от четности индекса зависит операция над элементами векторов.

1. size = 10000
> Gpu time : 0.059872 
2. size = 100000
> Gpu time : 0.029632
3. size = 1000000
> Gpu time : 0.09328
4. size = 10000000
> Gpu time : 0.4848

Если сравнить с заданием номер 1, видно, что эффективность упала на несколько порядков, при тех же размерах входных векторов.

Собрана метрика:
> ==129395== NVPROF is profiling process 129395, command: ./prog_gpu 100000000
>
> ==129395== Profiling application: ./prog_gpu 100000000
>
> ==129395== Profiling result:
>
> ==129395== Metric result:
>
> Invocations                               Metric Name                        Metric Description         Min         Max         Avg
>
> Device "Tesla P100-SXM2-16GB (0)"
>
>     Kernel: vecAdd(double*, double*, double*, int)
>
>           1                 warp_execution_efficiency                 Warp Execution Efficiency     100.00%     100.00%     100.00%

Причём, программа компилировалась с флагом -O0.
