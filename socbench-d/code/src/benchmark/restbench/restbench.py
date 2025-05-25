import json
from socrag.file import read_file
from ..benchmark import Benchmark, Query, Queryset


def load_benchmark(filename: str) -> list:
    benchmark = json.loads(read_file(filename))
    for benchmark_case in benchmark:
        benchmark_case["solution"] = list(dict.fromkeys(benchmark_case["solution"]))
    return benchmark

def _get_restbench_query(query) -> Query:
    return Query(query=query["query"], solution=query["solution"])


def _get_restbench_queries(queries) -> list[Query]:
    return [_get_restbench_query(query) for query in queries]

def get_restbench() -> Benchmark:
    benchmark_spotify = load_benchmark("./src/benchmark/restbench/data/datasets/spotify.json")
    benchmark_tmdb = load_benchmark("./src/benchmark/restbench/data/datasets/tmdb.json")
    openapi_spotify = read_file(f"./src/benchmark/restbench/data/specs/spotify_oas.json")
    openapi_tmdb = read_file(f"./src/benchmark/restbench/data/specs/tmdb_oas.json")
    queries_spotify = _get_restbench_queries(benchmark_spotify)
    queries_tmdb = _get_restbench_queries(benchmark_tmdb)
    queryset = Queryset(name="restbench", queries=queries_spotify+queries_tmdb, openapis=[openapi_spotify, openapi_tmdb])
    return Benchmark(name="restbench", queries=[queryset])
