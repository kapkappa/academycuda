# Отчёт по 9 заданию

Реализована pinned/unified memory, выполнены замеры времени выполнения в каждом случае.

1. size = 500
> USUAL (Memcpy+Kernel) time: 0.081984

> PINNED (Memcpy+Kernel) time: 0.035808

> UNIFIED (Kernel only) time: 0.092480
2. size = 2000
> USUAL (Memcpy+Kernel) time: 0.049696

> PINNED (Memcpy+Kernel) time: 0.047264

> UNIFIED (Kernel only) time: 0.084960
3. size = 10 000
> USUAL (Memcpy+Kernel) time: 0.066688

> PINNED (Memcpy+Kernel) time: 0.034496

> UNIFIED (Kernel only) time: 0.112192
4. size = 100 000
> USUAL (Memcpy+Kernel) time: 0.251328

> PINNED (Memcpy+Kernel) time: 0.057376

> UNIFIED (Kernel only) time: 0.326528
5. size = 500 000
> USUAL (Memcpy+Kernel) time: 1.022432

> PINNED (Memcpy+Kernel) time: 0.175392

> UNIFIED (Kernel only) time: 0.621376
6. size = 2 000 000
> USUAL (Memcpy+Kernel) time: 3.760288

> PINNED (Memcpy+Kernel) time: 0.574272

> UNIFIED (Kernel only) time: 2.108896
7. size = 50 000 000
> USUAL (Memcpy+Kernel) time: 93.641121

> PINNED (Memcpy+Kernel) time: 14.563392

> UNIFIED (Kernel only) time: 54.195553
8. size = 150 000 000
> USUAL (Memcpy+Kernel) time: 297.638641

> PINNED (Memcpy+Kernel) time: 40.867073

> UNIFIED (Kernel only) time: 151.249313

Для UNUFIED MEMORY ускорение составляет до 100%, но на исключительно на больших векторах, на малых векторах, время исполнения этого примера больше, чем обычного.
Для PINNED MEMORY ускорение составляет от 200 до 500%, стабильно лучше обычного способа распределения памяти для почти всех размеров векторов.
