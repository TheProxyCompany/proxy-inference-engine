"""
PIE Benchmark Harness

A comprehensive tool for running benchmarks, analyzing results, and generating reports.

Features:
- Run benchmarks with configurable parameters
- Generate terminal and Markdown reports
- Track benchmark trends over time

Usage examples:
  # Run all benchmarks and save results
  python3 benchmarks.py run --build-dir build --output benchmarks/results.json

  # Run specific benchmark with custom parameters
  python3 benchmarks.py run --filter=PageAllocator_ConcurrentRefCountContention --threads=4,8

  # Generate a report from benchmark results
  python3 benchmarks.py report --input benchmarks/results.json --output benchmarks/RESULTS.md
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Constants
DEFAULT_BUILD_DIR = "build"
DEFAULT_BENCHMARK_EXE = "benchmarks/cpp/pie_benchmarks"
DEFAULT_OUTPUT_DIR = "benchmarks"
DEFAULT_RESULT_FILE = "results.json"
DEFAULT_BASELINE_FILE = "baseline.json"
DEFAULT_REPORT_FILE = "RESULTS.md"
DEFAULT_REPETITIONS = 3
DEFAULT_CV_THRESHOLD = 0.05  # Coefficient of variation threshold (5%)

# Type aliases for clarity
BenchmarkName = str
BenchmarkMetrics = dict[str, Any]
BenchmarkResults = dict[BenchmarkName, BenchmarkMetrics]


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""

    build_dir: str
    exe_path: str
    filter_pattern: str | None
    repetitions: int
    output_file: str
    threads: list[int] | None
    real_time: bool
    cpu_time: bool
    verbose: bool
    extra_args: list[str]


@dataclass
class AnalysisConfig:
    """Configuration for benchmark analysis."""

    input_file: str
    cv_threshold: float
    verbose: bool


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    input_file: str
    output_file: str | None
    markdown: bool
    include_system_info: bool
    verbose: bool


class BenchmarkRunner:
    """Handles running the benchmark executable with specified parameters."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.exe_path = Path(config.build_dir) / config.exe_path

        # Validate executable exists
        if not os.path.isfile(self.exe_path):
            raise FileNotFoundError(
                f"Benchmark executable not found at: {self.exe_path}"
            )

    def run(self) -> None:
        """Run benchmarks and return the parsed JSON results."""
        # Construct command
        cmd = [str(self.exe_path)]

        # Add filters if specified
        if self.config.filter_pattern:
            cmd.append(f"--benchmark_filter={self.config.filter_pattern}")

        # Add repetitions
        if self.config.repetitions > 1:
            cmd.append(f"--benchmark_repetitions={self.config.repetitions}")

        # Add threading options
        if self.config.threads:
            for thread_count in self.config.threads:
                cmd.append(f"--benchmark_threads={thread_count}")

        # Add timing options
        if self.config.real_time:
            cmd.append("--benchmark_time_unit=ms")
            cmd.append("--benchmark_use_real_time")

        # Set JSON output format and file
        cmd.append("--benchmark_format=json")
        cmd.append(f"--benchmark_out={self.config.output_file}")

        # Add any extra arguments
        cmd.extend(self.config.extra_args)

        # Print command if verbose
        if self.config.verbose:
            print(f"Running command: {' '.join(cmd)}")

        # Run command
        start_time = time.time()
        print(f"Starting benchmarks at {datetime.now().strftime('%H:%M:%S')}...")

        try:
            # Run with output displayed to terminal
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as e:
                print(f"Error: Benchmark execution failed: {e}")
                raise

            elapsed = time.time() - start_time
            print(f"Benchmarks completed in {elapsed:.2f} seconds.")
            print(f"Results saved to: {self.config.output_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error running benchmarks: {e}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error parsing benchmark results: {e}")
            sys.exit(1)


class BenchmarkAnalyzer:
    """Analyzes benchmark results to check stability and detect regressions."""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.current_results = self._load_results(config.input_file)
        self.baseline_results: dict[str, Any] | None = None

    def _load_results(self, file_path: str) -> dict[str, Any]:
        """Load benchmark results from a JSON file."""
        try:
            with open(file_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading benchmark results from {file_path}: {e}")
            sys.exit(1)

    def extract_metrics(self, results: dict[str, Any]) -> BenchmarkResults:
        """Extract key metrics from benchmark JSON data."""
        metrics: BenchmarkResults = {}

        for benchmark in results.get("benchmarks", []):
            name = benchmark.get("name")

            if not name:
                continue

            # Extract core metrics
            metrics[name] = {
                "real_time": float(benchmark.get("real_time", 0)),
                "cpu_time": float(benchmark.get("cpu_time", 0)),
                "time_unit": benchmark.get("time_unit", "ns"),
                "iterations": int(benchmark.get("iterations", 0)),
                "threads": int(benchmark.get("threads", 1)),
                "counters": {},
            }

            # Extract any custom counters
            for counter_name, counter_value in benchmark.get("counters", {}).items():
                metrics[name]["counters"][counter_name] = float(counter_value)

        return metrics

    def group_by_base_name(
        self, metrics: BenchmarkResults
    ) -> dict[str, list[tuple[str, BenchmarkMetrics]]]:
        """Group benchmarks by their base name (without repetition suffix)."""
        groups: defaultdict[str, list[tuple[str, BenchmarkMetrics]]] = defaultdict(list)

        for name, data in metrics.items():
            # Split by repetition marker if present
            base_name = name.split("/", 1)[0]
            groups[base_name].append((name, data))

        return dict(groups)


class ReportGenerator:
    """Generates terminal and Markdown reports from benchmark results."""

    def __init__(self, config: ReportConfig):
        self.config = config
        self.analyzer = BenchmarkAnalyzer(
            AnalysisConfig(
                input_file=config.input_file,
                cv_threshold=DEFAULT_CV_THRESHOLD,
                verbose=config.verbose,
            )
        )

    def generate_system_info(self) -> dict[str, str]:
        """Gather system information for the report."""
        system_info = {
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "System": platform.system(),
            "Release": platform.release(),
            "Machine": platform.machine(),
            "Processor": platform.processor(),
            "Python": platform.python_version(),
        }

        # Try to get CPU info on Linux
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo") as f:
                    cpuinfo = f.read()
                match = re.search(r"model name\s+:\s+(.*)", cpuinfo)
                if match:
                    system_info["CPU"] = match.group(1)

                match = re.search(r"cpu cores\s+:\s+(.*)", cpuinfo)
                if match:
                    system_info["CPU Cores"] = match.group(1)
            except Exception:
                pass

        # Try to get memory info
        try:
            import psutil

            mem = psutil.virtual_memory()
            system_info["Memory"] = f"{mem.total / (1024**3):.1f} GB"
        except ImportError:
            system_info["Memory"] = "Unknown (psutil not installed)"

        return system_info

    def summarize_benchmark_results(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Summarize and categorize benchmark results."""
        metrics = self.analyzer.extract_metrics(self.analyzer.current_results)
        # Group by benchmark family (e.g., PageAllocator_SingleThreaded)
        families: defaultdict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

        for name, data in metrics.items():
            # Extract family name (before first underscore)
            parts = name.split("_")
            if len(parts) >= 2:
                family = "_".join(parts[:2])
                benchmark_name = "_".join(parts[2:]) if len(parts) > 2 else name
                families[family][benchmark_name] = data

        return dict(families)

    def generate_markdown_report(self) -> str:
        """Generate a detailed Markdown report of benchmark results."""
        # Get benchmark data and analysis results
        summary = self.summarize_benchmark_results()
        # Start building the Markdown report
        report = []
        report.append("# PIE Benchmark Results\n")

        # Add date and system info
        if self.config.include_system_info:
            system_info = self.generate_system_info()
            report.append("## System Information\n")
            report.append("| Property | Value |")
            report.append("| --- | --- |")

            for key, value in system_info.items():
                report.append(f"| {key} | {value} |")

            report.append("\n")

        # Summary of benchmarks
        report.append("## Summary\n")
        report.append(f"- Total benchmark categories: {len(summary)}")
        report.append("\n")

        # Detailed benchmark results by family
        report.append("## Benchmark Results\n")

        for family, benchmarks in summary.items():
            report.append(f"### {family}\n")

            # Create a table for this family
            report.append(
                "| Benchmark | Real Time | CPU Time | Time Unit | Iterations | Threads |"
            )
            report.append("| --- | --- | --- | --- | --- | --- |")

            for name, data in benchmarks.items():
                # Format the row
                report.append(
                    f"| {name} | "
                    f"{data['real_time']:.3f} | "
                    f"{data['cpu_time']:.3f} | "
                    f"{data['time_unit']} | "
                    f"{data['iterations']} | "
                    f"{data['threads']} |"
                )

            # Add custom counters if any
            has_counters = any(
                bool(data.get("counters")) for data in benchmarks.values()
            )

            if has_counters:
                report.append("\n#### Performance Counters\n")

                # Collect all counter names across all benchmarks in this family
                counter_names = set()
                for data in benchmarks.values():
                    counter_names.update(data.get("counters", {}).keys())

                if counter_names:
                    # Create a table for counters
                    header = "| Benchmark | " + " | ".join(counter_names) + " |"
                    report.append(header)
                    report.append(
                        "| --- | " + " | ".join(["---"] * len(counter_names)) + " |"
                    )

                    for name, data in benchmarks.items():
                        row = f"| {name} |"
                        for counter in counter_names:
                            value = data.get("counters", {}).get(counter, "N/A")
                            if isinstance(value, int | float):
                                row += f" {value:.2f} |"
                            else:
                                row += f" {value} |"

                        report.append(row)

            report.append("\n")

        # Return the report as a string
        return "\n".join(report)

    def generate_report(self):
        """Generate benchmark report based on config, returns True if issues found."""
        # Print terminal report
        print("\n=== BENCHMARK REPORT ===\n")
        if self.config.include_system_info:
            system_info = self.generate_system_info()
            print("System Information:")
            for key, value in system_info.items():
                print(f"  {key}: {value}")

        # Summary section
        benchmark_families = self.summarize_benchmark_results()
        print(f"\nFound {len(benchmark_families)} benchmark categories")

        # Generate Markdown report if requested
        if self.config.markdown:
            markdown_report = self.generate_markdown_report()

            if self.config.output_file:
                with open(self.config.output_file, "w") as f:
                    f.write(markdown_report)
                print(f"\nMarkdown report saved to: {self.config.output_file}")
            else:
                print("\n=== MARKDOWN REPORT ===\n")
                print(markdown_report)

        return False


def run_command(args):
    """Run benchmarks with specified parameters."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Configure and run benchmarks
    config = BenchmarkConfig(
        build_dir=str(args.build_dir),
        exe_path=args.exe_path,
        filter_pattern=args.filter,
        repetitions=args.repetitions,
        output_file=args.output,
        threads=[int(t) for t in args.threads.split(",")] if args.threads else None,
        real_time=args.real_time,
        cpu_time=args.cpu_time,
        verbose=args.verbose,
        extra_args=args.extra_args if args.extra_args else [],
    )

    runner = BenchmarkRunner(config)
    runner.run()


def report_command(args):
    """Generate a report from benchmark results."""
    config = ReportConfig(
        input_file=str(args.input),
        output_file=args.output,
        markdown=True,
        include_system_info=args.system_info,
        verbose=args.verbose,
    )

    generator = ReportGenerator(config)
    has_issues = generator.generate_report()

    return 1 if has_issues and args.fail_on_issues else 0


def main():
    parser = argparse.ArgumentParser(
        description="PIE Benchmark Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run benchmarks")
    run_parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(DEFAULT_BUILD_DIR),
        help=f"Build directory containing benchmark executable (default: {DEFAULT_BUILD_DIR})",
    )
    run_parser.add_argument(
        "--exe-path",
        type=str,
        default=DEFAULT_BENCHMARK_EXE,
        help=f"Path to benchmark executable relative to build directory (default: {DEFAULT_BENCHMARK_EXE})",
    )
    run_parser.add_argument(
        "--filter", help="Filter pattern for benchmark names (e.g., 'PageAllocator_*')"
    )
    run_parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
        help=f"Number of repetitions for each benchmark (default: {DEFAULT_REPETITIONS})",
    )
    run_parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR) / DEFAULT_RESULT_FILE,
        help=f"Output file for benchmark results (default: {os.path.join(DEFAULT_OUTPUT_DIR, DEFAULT_RESULT_FILE)})",
    )
    run_parser.add_argument(
        "--threads",
        help="Comma-separated list of thread counts to benchmark (e.g., '1,2,4,8')",
    )
    run_parser.add_argument(
        "--real-time", action="store_true", help="Measure real time instead of CPU time"
    )
    run_parser.add_argument(
        "--cpu-time", action="store_true", help="Measure CPU time (default)"
    )
    run_parser.add_argument(
        "--analyze", action="store_true", help="Analyze results after running"
    )
    run_parser.add_argument(
        "--baseline", type=Path, help="Baseline file to compare against"
    )
    run_parser.add_argument(
        "--cv-threshold",
        type=float,
        default=DEFAULT_CV_THRESHOLD,
        help=f"Coefficient of variation threshold for stability analysis (default: {DEFAULT_CV_THRESHOLD})",
    )
    run_parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        help="Additional arguments to pass to benchmark executable",
    )
    run_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze benchmark results")
    analyze_parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input file containing benchmark results",
    )
    analyze_parser.add_argument(
        "--cv-threshold",
        type=float,
        default=DEFAULT_CV_THRESHOLD,
        help=f"Coefficient of variation threshold (default: {DEFAULT_CV_THRESHOLD})",
    )
    analyze_parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help="Exit with non-zero status if issues are found",
    )
    analyze_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare benchmark results")
    compare_parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input file containing current benchmark results",
    )
    compare_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate benchmark report")
    report_parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input file containing benchmark results",
    )
    report_parser.add_argument(
        "--output", type=Path, help="Output file for report (default: print to stdout)"
    )
    report_parser.add_argument(
        "--baseline", type=Path, help="Baseline file to compare against"
    )
    report_parser.add_argument(
        "--system-info",
        action="store_true",
        help="Include system information in report",
    )
    report_parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help="Exit with non-zero status if issues are found",
    )
    report_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    if args.command == "run":
        return run_command(args)
    elif args.command == "report":
        return report_command(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
