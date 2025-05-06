## Quick Start Guide

For a complete benchmarking workflow:

```bash
# 1. Build benchmarks
cmake -S . -B build -DPIE_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target pie_benchmarks

# 2. Run all benchmarks and analyze results in one step
python3 scripts/benchmarks.py run --analyze --output benchmarks/baseline.json

# 3. Later, run benchmarks and compare against baseline
python3 scripts/benchmarks.py run --output current_results.json
python3 scripts/benchmarks.py compare --input current_results.json --baseline benchmarks/baseline.json

# 4. Generate a Markdown report with all benchmark results
python3 scripts/benchmarks.py report --input current_results.json --baseline benchmarks/baseline.json --output benchmarks/RESULTS.md --system-info
```

## Building the Benchmarks

By default, benchmarks are not built. To build them, use the `PIE_BUILD_BENCHMARKS` CMake option:

```bash
# Configure with benchmarks enabled
cmake -S . -B build -DPIE_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release

# Build just the benchmarks target
cmake --build build --target pie_benchmarks -j$(nproc)
```

For profiling with Tracy, add `-DPIE_USE_TRACY=ON` to the CMake command.

## Running the Benchmarks

The benchmarks executable will be built as `build/benchmarks/pie_benchmarks`.

```bash
# Run all benchmarks
./build/benchmarks/pie_benchmarks

# Filter to specific benchmarks
./build/benchmarks/pie_benchmarks --benchmark_filter=PageAllocator_SingleThreaded

# Run with multiple repetitions (for variance analysis)
./build/benchmarks/pie_benchmarks --benchmark_repetitions=5

# Output to JSON file for analysis
./build/benchmarks/pie_benchmarks --benchmark_format=json --benchmark_out=benchmarks/baseline.json
```

## Analyzing Results

The `scripts/benchmarks.py` script is a comprehensive benchmark harness that can run benchmarks, analyze results, and generate reports:

```bash
# Run specific benchmarks with filters
python3 scripts/benchmarks.py run --filter=PageAllocator_ConcurrentRefCount* --threads=2,4,8

# Analyze existing benchmark results for stability
python3 scripts/benchmarks.py analyze --input benchmarks/results.json

# Compare against baseline (detect regressions)
python3 scripts/benchmarks.py compare --input new_results.json --baseline benchmarks/baseline.json

# Generate comprehensive Markdown report
python3 scripts/benchmarks.py report --input benchmarks/results.json --output benchmarks/RESULTS.md
```

For CI integration, the script returns non-zero exit codes when regressions are detected. Use the `--fail-on-issues` or `--fail-on-regression` flags to fail CI when issues are found.

## Tracy Profiler Integration

For deeper performance analysis, Tracy profiler integration is available:

```bash
# Build with Tracy enabled
cmake -S . -B build -DPIE_BUILD_BENCHMARKS=ON -DPIE_USE_TRACY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target pie_benchmarks

# Run benchmarks while Tracy Profiler is running
./build/benchmarks/pie_benchmarks
```

The profiled allocator uses a template decorator pattern that only adds profiling instrumentation when Tracy is enabled, with zero overhead otherwise.
