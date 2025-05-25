from ..benchmark import Benchmark
from .create_benchmark import create_benchmark

BENCHMARK_ROOT = "src/benchmark/socbenchd"


def get_socbenchd(count_benchmarks: int = 5) -> list[Benchmark]:
    assert count_benchmarks > 0
    return [
        create_benchmark(BENCHMARK_ROOT, f"socbenchd_{i + 1}")
        for i in range(count_benchmarks)
    ]
