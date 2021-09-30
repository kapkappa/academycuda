# Отчёт по 10 заданию

Реализовано разделение задачи сложения векторов на потоки, с целью повысить эффективность работы программы за счёт асинхронных операций копирования данных с HtoD/DtoH, параллельно с вычислениями.
В информации из профилировщика видно, что главная задача для оптимизации - работа с памятью, и в резльтате, при использовании разделения на потоки, отношение Kernel / Memcpy получилось увеличить в два раза.

1. size = 500
> Common time: 0.056224

> Stream time: 0.118784
2. size = 2000
> Common time: 0.081856

> Stream time: 0.187200
3. size = 10 000
> Common time: 0.070912

> Stream time: 0.119616
4. size = 100 000
> Common time: 0.257088

> Stream time: 0.167840
5. size = 500 000
> Common time: 1.079232

> Stream time: 0.500256
6. size = 2 000 000
> Common time: 4.025952

> Stream time: 1.223328
7. size = 50 000 000
> Common time: 99.785599

> Stream time: 38.355457
8. size = 150 000 000
> Common time: 277.548126

> Stream time: 96.103325

Замеры проводились для копирования данных и исполнения ядра. По результатам видно, что при таком разделении на потоки получается добиться ускорения в 2,5 раза.