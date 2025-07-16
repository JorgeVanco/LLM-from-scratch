The results for the `all reduce` benchmark.

|    | backend   |   data_size (MB) |   time_taken |    std_time |
|---:|:----------|-----------------:|-------------:|------------:|
|  0 | gloo      |                1 |  0.00260709  | 0.000921305 |
|  1 | gloo      |               10 |  0.00858294  | 0.000962009 |
|  2 | gloo      |              100 |  0.0737034   | 0.00614307  |
|  3 | gloo      |             1024 |  0.912003    | 0.139844    |
|  4 | nccl      |                1 |  4.75724e-05 | 3.10624e-05 |
|  5 | nccl      |               10 |  9.55087e-05 | 5.64021e-06 |
|  6 | nccl      |              100 |  0.000521944 | 6.47293e-05 |
|  7 | nccl      |             1024 |  0.00456288  | 4.72274e-05 |

The results for the simple `Individual DDP` benchmark.

|    | model_size   | backend   |   time_per_step |   std_time_step |   time_per_sync |   std_time_sync |   fraction_sync |
|---:|:-------------|:----------|----------------:|----------------:|----------------:|----------------:|----------------:|
|  0 | xl           | nccl      |        0.408564 |     0.000637803 |       0.0102613 |     0.000398993 |       0.0251155 |