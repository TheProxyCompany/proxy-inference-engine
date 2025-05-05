# PIE Benchmark Results

## System Information

| Property | Value |
| --- | --- |
| Date | 2025-05-05 13:27:36 |
| System | Darwin |
| Release | 24.5.0 |
| Machine | arm64 |
| Processor | arm |
| Python | 3.12.7 |
| Memory | 128.0 GB |


## Summary

- Total benchmark categories: 1


## Benchmark Results

### BM_PageAllocator

| Benchmark | Real Time | CPU Time | Time Unit | Iterations | Threads |
| --- | --- | --- | --- | --- | --- |
| SingleThreadedAllocation/2000/32/80/real_time | 51.624 | 51.621 | us | 13569 | 1 |
| SingleThreadedAllocation/2000/32/80/real_time_mean | 51.707 | 51.704 | us | 3 | 1 |
| SingleThreadedAllocation/2000/32/80/real_time_median | 51.701 | 51.701 | us | 3 | 1 |
| SingleThreadedAllocation/2000/32/80/real_time_stddev | 0.086 | 0.085 | us | 3 | 1 |
| SingleThreadedAllocation/2000/32/80/real_time_cv | 0.002 | 0.002 | us | 3 | 1 |
| SingleThreadedAllocation/5000/32/128/real_time | 132.543 | 132.187 | us | 5440 | 1 |
| SingleThreadedAllocation/5000/32/128/real_time_mean | 130.068 | 129.940 | us | 3 | 1 |
| SingleThreadedAllocation/5000/32/128/real_time_median | 129.060 | 129.041 | us | 3 | 1 |
| SingleThreadedAllocation/5000/32/128/real_time_stddev | 2.155 | 1.959 | us | 3 | 1 |
| SingleThreadedAllocation/5000/32/128/real_time_cv | 0.017 | 0.015 | us | 3 | 1 |
| SingleThreadedAllocation/10000/40/128/real_time | 262.953 | 262.938 | us | 2590 | 1 |
| SingleThreadedAllocation/10000/40/128/real_time_mean | 262.611 | 262.573 | us | 3 | 1 |
| SingleThreadedAllocation/10000/40/128/real_time_median | 262.953 | 262.938 | us | 3 | 1 |
| SingleThreadedAllocation/10000/40/128/real_time_stddev | 1.096 | 1.098 | us | 3 | 1 |
| SingleThreadedAllocation/10000/40/128/real_time_cv | 0.004 | 0.004 | us | 3 | 1 |
| SingleThreadedAllocation/20000/60/128/real_time | 533.620 | 533.544 | us | 1274 | 1 |
| SingleThreadedAllocation/20000/60/128/real_time_mean | 531.742 | 531.666 | us | 3 | 1 |
| SingleThreadedAllocation/20000/60/128/real_time_median | 532.207 | 532.119 | us | 3 | 1 |
| SingleThreadedAllocation/20000/60/128/real_time_stddev | 2.148 | 2.140 | us | 3 | 1 |
| SingleThreadedAllocation/20000/60/128/real_time_cv | 0.004 | 0.004 | us | 3 | 1 |
| MultiThreadedAllocation/2000/0/32/80/real_time/threads:1 | 0.057 | 0.057 | ms | 12465 | 1 |
| MultiThreadedAllocation/2000/0/32/80/real_time/threads:1_mean | 0.056 | 0.056 | ms | 3 | 1 |
| MultiThreadedAllocation/2000/0/32/80/real_time/threads:1_median | 0.056 | 0.056 | ms | 3 | 1 |
| MultiThreadedAllocation/2000/0/32/80/real_time/threads:1_stddev | 0.000 | 0.000 | ms | 3 | 1 |
| MultiThreadedAllocation/2000/0/32/80/real_time/threads:1_cv | 0.003 | 0.003 | ms | 3 | 1 |
| MultiThreadedAllocation/2000/0/32/80/real_time/threads:2 | 0.275 | 0.275 | ms | 2030 | 2 |
| MultiThreadedAllocation/2000/0/32/80/real_time/threads:2_mean | 0.289 | 0.289 | ms | 3 | 2 |
| MultiThreadedAllocation/2000/0/32/80/real_time/threads:2_median | 0.291 | 0.291 | ms | 3 | 2 |
| MultiThreadedAllocation/2000/0/32/80/real_time/threads:2_stddev | 0.013 | 0.013 | ms | 3 | 2 |
| MultiThreadedAllocation/2000/0/32/80/real_time/threads:2_cv | 0.046 | 0.046 | ms | 3 | 2 |
| MultiThreadedAllocation/2000/0/32/80/real_time/threads:4 | 0.614 | 0.613 | ms | 1432 | 4 |
| MultiThreadedAllocation/2000/0/32/80/real_time/threads:4_mean | 0.631 | 0.631 | ms | 3 | 4 |
| MultiThreadedAllocation/2000/0/32/80/real_time/threads:4_median | 0.614 | 0.613 | ms | 3 | 4 |
| MultiThreadedAllocation/2000/0/32/80/real_time/threads:4_stddev | 0.045 | 0.044 | ms | 3 | 4 |
| MultiThreadedAllocation/2000/0/32/80/real_time/threads:4_cv | 0.071 | 0.070 | ms | 3 | 4 |
| MultiThreadedAllocation/2000/0/32/80/real_time/threads:8 | 1.224 | 1.222 | ms | 760 | 8 |
| MultiThreadedAllocation/2000/0/32/80/real_time/threads:8_mean | 1.229 | 1.228 | ms | 3 | 8 |
| MultiThreadedAllocation/2000/0/32/80/real_time/threads:8_median | 1.224 | 1.223 | ms | 3 | 8 |
| MultiThreadedAllocation/2000/0/32/80/real_time/threads:8_stddev | 0.009 | 0.009 | ms | 3 | 8 |
| MultiThreadedAllocation/2000/0/32/80/real_time/threads:8_cv | 0.007 | 0.007 | ms | 3 | 8 |
| MultiThreadedAllocation/2000/0/32/80/real_time/threads:16 | 1.951 | 1.851 | ms | 400 | 16 |
| MultiThreadedAllocation/2000/0/32/80/real_time/threads:16_mean | 1.925 | 1.831 | ms | 3 | 16 |
| MultiThreadedAllocation/2000/0/32/80/real_time/threads:16_median | 1.918 | 1.833 | ms | 3 | 16 |
| MultiThreadedAllocation/2000/0/32/80/real_time/threads:16_stddev | 0.024 | 0.020 | ms | 3 | 16 |
| MultiThreadedAllocation/2000/0/32/80/real_time/threads:16_cv | 0.012 | 0.011 | ms | 3 | 16 |
| MultiThreadedAllocation/5000/0/32/128/real_time/threads:1 | 0.064 | 0.064 | ms | 9751 | 1 |
| MultiThreadedAllocation/5000/0/32/128/real_time/threads:1_mean | 0.066 | 0.066 | ms | 3 | 1 |
| MultiThreadedAllocation/5000/0/32/128/real_time/threads:1_median | 0.065 | 0.064 | ms | 3 | 1 |
| MultiThreadedAllocation/5000/0/32/128/real_time/threads:1_stddev | 0.002 | 0.002 | ms | 3 | 1 |
| MultiThreadedAllocation/5000/0/32/128/real_time/threads:1_cv | 0.037 | 0.037 | ms | 3 | 1 |
| MultiThreadedAllocation/5000/0/32/128/real_time/threads:2 | 0.389 | 0.389 | ms | 2086 | 2 |
| MultiThreadedAllocation/5000/0/32/128/real_time/threads:2_mean | 0.395 | 0.393 | ms | 3 | 2 |
| MultiThreadedAllocation/5000/0/32/128/real_time/threads:2_median | 0.389 | 0.389 | ms | 3 | 2 |
| MultiThreadedAllocation/5000/0/32/128/real_time/threads:2_stddev | 0.014 | 0.011 | ms | 3 | 2 |
| MultiThreadedAllocation/5000/0/32/128/real_time/threads:2_cv | 0.036 | 0.029 | ms | 3 | 2 |
| MultiThreadedAllocation/5000/0/32/128/real_time/threads:4 | 0.808 | 0.807 | ms | 736 | 4 |
| MultiThreadedAllocation/5000/0/32/128/real_time/threads:4_mean | 1.003 | 1.001 | ms | 3 | 4 |
| MultiThreadedAllocation/5000/0/32/128/real_time/threads:4_median | 1.064 | 1.063 | ms | 3 | 4 |
| MultiThreadedAllocation/5000/0/32/128/real_time/threads:4_stddev | 0.172 | 0.172 | ms | 3 | 4 |
| MultiThreadedAllocation/5000/0/32/128/real_time/threads:4_cv | 0.172 | 0.172 | ms | 3 | 4 |
| MultiThreadedAllocation/5000/0/32/128/real_time/threads:8 | 1.811 | 1.809 | ms | 432 | 8 |
| MultiThreadedAllocation/5000/0/32/128/real_time/threads:8_mean | 1.730 | 1.728 | ms | 3 | 8 |
| MultiThreadedAllocation/5000/0/32/128/real_time/threads:8_median | 1.698 | 1.697 | ms | 3 | 8 |
| MultiThreadedAllocation/5000/0/32/128/real_time/threads:8_stddev | 0.070 | 0.070 | ms | 3 | 8 |
| MultiThreadedAllocation/5000/0/32/128/real_time/threads:8_cv | 0.041 | 0.040 | ms | 3 | 8 |
| MultiThreadedAllocation/5000/0/32/128/real_time/threads:16 | 3.326 | 3.238 | ms | 160 | 16 |
| MultiThreadedAllocation/5000/0/32/128/real_time/threads:16_mean | 3.415 | 3.327 | ms | 3 | 16 |
| MultiThreadedAllocation/5000/0/32/128/real_time/threads:16_median | 3.456 | 3.366 | ms | 3 | 16 |
| MultiThreadedAllocation/5000/0/32/128/real_time/threads:16_stddev | 0.077 | 0.077 | ms | 3 | 16 |
| MultiThreadedAllocation/5000/0/32/128/real_time/threads:16_cv | 0.023 | 0.023 | ms | 3 | 16 |
| MultiThreadedAllocation/10000/0/40/128/real_time/threads:1 | 0.079 | 0.079 | ms | 8779 | 1 |
| MultiThreadedAllocation/10000/0/40/128/real_time/threads:1_mean | 0.080 | 0.080 | ms | 3 | 1 |
| MultiThreadedAllocation/10000/0/40/128/real_time/threads:1_median | 0.080 | 0.080 | ms | 3 | 1 |
| MultiThreadedAllocation/10000/0/40/128/real_time/threads:1_stddev | 0.000 | 0.000 | ms | 3 | 1 |
| MultiThreadedAllocation/10000/0/40/128/real_time/threads:1_cv | 0.002 | 0.002 | ms | 3 | 1 |
| MultiThreadedAllocation/10000/0/40/128/real_time/threads:2 | 0.473 | 0.472 | ms | 1500 | 2 |
| MultiThreadedAllocation/10000/0/40/128/real_time/threads:2_mean | 0.469 | 0.468 | ms | 3 | 2 |
| MultiThreadedAllocation/10000/0/40/128/real_time/threads:2_median | 0.471 | 0.471 | ms | 3 | 2 |
| MultiThreadedAllocation/10000/0/40/128/real_time/threads:2_stddev | 0.005 | 0.005 | ms | 3 | 2 |
| MultiThreadedAllocation/10000/0/40/128/real_time/threads:2_cv | 0.011 | 0.011 | ms | 3 | 2 |
| MultiThreadedAllocation/10000/0/40/128/real_time/threads:4 | 0.580 | 0.580 | ms | 1084 | 4 |
| MultiThreadedAllocation/10000/0/40/128/real_time/threads:4_mean | 0.603 | 0.603 | ms | 3 | 4 |
| MultiThreadedAllocation/10000/0/40/128/real_time/threads:4_median | 0.604 | 0.604 | ms | 3 | 4 |
| MultiThreadedAllocation/10000/0/40/128/real_time/threads:4_stddev | 0.023 | 0.023 | ms | 3 | 4 |
| MultiThreadedAllocation/10000/0/40/128/real_time/threads:4_cv | 0.038 | 0.038 | ms | 3 | 4 |
| MultiThreadedAllocation/10000/0/40/128/real_time/threads:8 | 1.466 | 1.466 | ms | 480 | 8 |
| MultiThreadedAllocation/10000/0/40/128/real_time/threads:8_mean | 1.334 | 1.334 | ms | 3 | 8 |
| MultiThreadedAllocation/10000/0/40/128/real_time/threads:8_median | 1.327 | 1.327 | ms | 3 | 8 |
| MultiThreadedAllocation/10000/0/40/128/real_time/threads:8_stddev | 0.130 | 0.130 | ms | 3 | 8 |
| MultiThreadedAllocation/10000/0/40/128/real_time/threads:8_cv | 0.097 | 0.097 | ms | 3 | 8 |
| MultiThreadedAllocation/10000/0/40/128/real_time/threads:16 | 3.308 | 3.289 | ms | 160 | 16 |
| MultiThreadedAllocation/10000/0/40/128/real_time/threads:16_mean | 3.331 | 3.298 | ms | 3 | 16 |
| MultiThreadedAllocation/10000/0/40/128/real_time/threads:16_median | 3.310 | 3.289 | ms | 3 | 16 |
| MultiThreadedAllocation/10000/0/40/128/real_time/threads:16_stddev | 0.038 | 0.050 | ms | 3 | 16 |
| MultiThreadedAllocation/10000/0/40/128/real_time/threads:16_cv | 0.011 | 0.015 | ms | 3 | 16 |
| ReferenceCountingScenario/2000/32/80/real_time | 0.047 | 0.047 | ms | 14952 | 1 |
| ReferenceCountingScenario/2000/32/80/real_time_mean | 0.047 | 0.047 | ms | 3 | 1 |
| ReferenceCountingScenario/2000/32/80/real_time_median | 0.047 | 0.047 | ms | 3 | 1 |
| ReferenceCountingScenario/2000/32/80/real_time_stddev | 0.000 | 0.000 | ms | 3 | 1 |
| ReferenceCountingScenario/2000/32/80/real_time_cv | 0.001 | 0.001 | ms | 3 | 1 |
| ReferenceCountingScenario/5000/32/128/real_time | 0.116 | 0.116 | ms | 6027 | 1 |
| ReferenceCountingScenario/5000/32/128/real_time_mean | 0.116 | 0.116 | ms | 3 | 1 |
| ReferenceCountingScenario/5000/32/128/real_time_median | 0.116 | 0.116 | ms | 3 | 1 |
| ReferenceCountingScenario/5000/32/128/real_time_stddev | 0.000 | 0.000 | ms | 3 | 1 |
| ReferenceCountingScenario/5000/32/128/real_time_cv | 0.001 | 0.001 | ms | 3 | 1 |
| ReferenceCountingScenario/10000/40/128/real_time | 0.231 | 0.231 | ms | 3017 | 1 |
| ReferenceCountingScenario/10000/40/128/real_time_mean | 0.231 | 0.231 | ms | 3 | 1 |
| ReferenceCountingScenario/10000/40/128/real_time_median | 0.231 | 0.231 | ms | 3 | 1 |
| ReferenceCountingScenario/10000/40/128/real_time_stddev | 0.000 | 0.000 | ms | 3 | 1 |
| ReferenceCountingScenario/10000/40/128/real_time_cv | 0.002 | 0.002 | ms | 3 | 1 |
| ReferenceCountingScenario/20000/60/128/real_time | 0.461 | 0.461 | ms | 1514 | 1 |
| ReferenceCountingScenario/20000/60/128/real_time_mean | 0.462 | 0.462 | ms | 3 | 1 |
| ReferenceCountingScenario/20000/60/128/real_time_median | 0.462 | 0.462 | ms | 3 | 1 |
| ReferenceCountingScenario/20000/60/128/real_time_stddev | 0.001 | 0.001 | ms | 3 | 1 |
| ReferenceCountingScenario/20000/60/128/real_time_cv | 0.002 | 0.002 | ms | 3 | 1 |
| SimulateLLMInference/4000/32/80/0/512/real_time/threads:4 | 3.198 | 1.780 | ms | 220 | 4 |
| SimulateLLMInference/4000/32/80/0/512/real_time/threads:4_mean | 3.206 | 1.782 | ms | 3 | 4 |
| SimulateLLMInference/4000/32/80/0/512/real_time/threads:4_median | 3.206 | 1.781 | ms | 3 | 4 |
| SimulateLLMInference/4000/32/80/0/512/real_time/threads:4_stddev | 0.008 | 0.003 | ms | 3 | 4 |
| SimulateLLMInference/4000/32/80/0/512/real_time/threads:4_cv | 0.002 | 0.002 | ms | 3 | 4 |
| SimulateLLMInference/4000/32/80/0/512/real_time/threads:8 | 6.939 | 3.874 | ms | 96 | 8 |
| SimulateLLMInference/4000/32/80/0/512/real_time/threads:8_mean | 6.985 | 3.910 | ms | 3 | 8 |
| SimulateLLMInference/4000/32/80/0/512/real_time/threads:8_median | 6.987 | 3.917 | ms | 3 | 8 |
| SimulateLLMInference/4000/32/80/0/512/real_time/threads:8_stddev | 0.045 | 0.033 | ms | 3 | 8 |
| SimulateLLMInference/4000/32/80/0/512/real_time/threads:8_cv | 0.006 | 0.008 | ms | 3 | 8 |
| SimulateLLMInference/4000/32/80/0/1024/real_time/threads:4 | 6.397 | 3.568 | ms | 108 | 4 |
| SimulateLLMInference/4000/32/80/0/1024/real_time/threads:4_mean | 6.402 | 3.564 | ms | 3 | 4 |
| SimulateLLMInference/4000/32/80/0/1024/real_time/threads:4_median | 6.404 | 3.563 | ms | 3 | 4 |
| SimulateLLMInference/4000/32/80/0/1024/real_time/threads:4_stddev | 0.004 | 0.004 | ms | 3 | 4 |
| SimulateLLMInference/4000/32/80/0/1024/real_time/threads:4_cv | 0.001 | 0.001 | ms | 3 | 4 |
| SimulateLLMInference/4000/32/80/0/1024/real_time/threads:8 | 13.886 | 7.736 | ms | 48 | 8 |
| SimulateLLMInference/4000/32/80/0/1024/real_time/threads:8_mean | 14.420 | 8.095 | ms | 3 | 8 |
| SimulateLLMInference/4000/32/80/0/1024/real_time/threads:8_median | 14.280 | 8.000 | ms | 3 | 8 |
| SimulateLLMInference/4000/32/80/0/1024/real_time/threads:8_stddev | 0.617 | 0.414 | ms | 3 | 8 |
| SimulateLLMInference/4000/32/80/0/1024/real_time/threads:8_cv | 0.043 | 0.051 | ms | 3 | 8 |
| SimulateLLMInference/8000/32/128/0/1024/real_time/threads:4 | 6.427 | 3.579 | ms | 108 | 4 |
| SimulateLLMInference/8000/32/128/0/1024/real_time/threads:4_mean | 6.405 | 3.568 | ms | 3 | 4 |
| SimulateLLMInference/8000/32/128/0/1024/real_time/threads:4_median | 6.405 | 3.570 | ms | 3 | 4 |
| SimulateLLMInference/8000/32/128/0/1024/real_time/threads:4_stddev | 0.022 | 0.012 | ms | 3 | 4 |
| SimulateLLMInference/8000/32/128/0/1024/real_time/threads:4_cv | 0.003 | 0.003 | ms | 3 | 4 |
| SimulateLLMInference/8000/32/128/0/1024/real_time/threads:8 | 14.037 | 7.876 | ms | 48 | 8 |
| SimulateLLMInference/8000/32/128/0/1024/real_time/threads:8_mean | 14.059 | 7.888 | ms | 3 | 8 |
| SimulateLLMInference/8000/32/128/0/1024/real_time/threads:8_median | 14.037 | 7.876 | ms | 3 | 8 |
| SimulateLLMInference/8000/32/128/0/1024/real_time/threads:8_stddev | 0.052 | 0.044 | ms | 3 | 8 |
| SimulateLLMInference/8000/32/128/0/1024/real_time/threads:8_cv | 0.004 | 0.006 | ms | 3 | 8 |
| SimulateLLMInference/16000/40/128/0/1024/real_time/threads:4 | 6.424 | 3.569 | ms | 108 | 4 |
| SimulateLLMInference/16000/40/128/0/1024/real_time/threads:4_mean | 6.439 | 3.578 | ms | 3 | 4 |
| SimulateLLMInference/16000/40/128/0/1024/real_time/threads:4_median | 6.431 | 3.574 | ms | 3 | 4 |
| SimulateLLMInference/16000/40/128/0/1024/real_time/threads:4_stddev | 0.020 | 0.011 | ms | 3 | 4 |
| SimulateLLMInference/16000/40/128/0/1024/real_time/threads:4_cv | 0.003 | 0.003 | ms | 3 | 4 |
| SimulateLLMInference/16000/40/128/0/1024/real_time/threads:8 | 14.116 | 7.907 | ms | 48 | 8 |
| SimulateLLMInference/16000/40/128/0/1024/real_time/threads:8_mean | 14.143 | 7.931 | ms | 3 | 8 |
| SimulateLLMInference/16000/40/128/0/1024/real_time/threads:8_median | 14.116 | 7.922 | ms | 3 | 8 |
| SimulateLLMInference/16000/40/128/0/1024/real_time/threads:8_stddev | 0.055 | 0.029 | ms | 3 | 8 |
| SimulateLLMInference/16000/40/128/0/1024/real_time/threads:8_cv | 0.004 | 0.004 | ms | 3 | 8 |
