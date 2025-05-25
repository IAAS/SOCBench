import math
import json
from pydantic import BaseModel
from typing import List, Optional
from llama_index.core.schema import NodeWithScore
from llama_index.core.base.base_retriever import BaseRetriever
from progress.bar import IncrementalBar
from functools import reduce
from benchmark.benchmark import Query, Queryset
from benchmark.encoding import count_token


def _compute_f1(recall, precision):
    if precision + recall == 0:
        return math.nan
    return (2 * precision * recall) / (precision + recall)


class RetrievedEndpoint(BaseModel):
    endpoint: str
    confidence: float


class BenchmarkMeasurements(BaseModel):
    recall: float
    precision: float
    f1: float
    token: dict[str, float]

class PreciseTokenCount(BaseModel):
    prompt_token: int
    completion_token: int
    response_token: int

class BenchmarkEntry(BaseModel):
    query: str
    solution: list[str]
    retrieved_endpoints: list[RetrievedEndpoint]
    recall: float
    precision: float | None
    token: int | PreciseTokenCount
    response: Optional[str] = None


class BenchmarkResult(BaseModel):
    name: str
    measurements: BenchmarkMeasurements
    entries: list[BenchmarkEntry]


class BenchmarkResultSet(BaseModel):
    name: str
    results: list[BenchmarkResult]


def _get_endpoints_from(documents: list[NodeWithScore]) -> list[RetrievedEndpoint]:
    endpoints = {}
    for document in documents:
        for endpoint in json.loads(document.metadata["endpoints"]):
            if endpoint in endpoints:
                endpoints[endpoint].confidence = max(
                    endpoints[endpoint].confidence, document.score
                )
            else:
                endpoints[endpoint] = RetrievedEndpoint(
                    endpoint=endpoint, confidence=document.score
                )
    return list(endpoints.values())


class Evaluator:
    def __init__(self):
        self.count_total = 0
        self.count_true_positive = 0
        self.count_solution = 0
        self.count_prompt_token = 0
        self.count_completion_token = 0

    def add_total(self, value):
        self.count_total += value

    def add_true_positive(self, value):
        self.count_true_positive += value

    def add_solution(self, value):
        self.count_solution += value

    def add_prompt_token(self, value):
        self.count_prompt_token += value

    def add_completion_token(self, value):
        self.count_completion_token += value

    def get_recall(self):
        if self.count_solution == 0:
            return math.nan
        return self.count_true_positive / self.count_solution

    def get_precision(self):
        if self.count_total == 0:
            return math.nan
        return self.count_true_positive / self.count_total

    def get_f1(self):
        return _compute_f1(self.get_recall(), self.get_precision())

    def get_prompt_token_count(self):
        return self.count_prompt_token

    def get_completion_token_count(self):
        return self.count_completion_token

    def get_total_token_count(self):
        return self.get_prompt_token_count() + self.get_completion_token_count()

    def get_result(self, len_benchmark) -> BenchmarkMeasurements:
        return BenchmarkMeasurements(
            recall=self.get_recall(),
            precision=self.get_precision(),
            f1=self.get_f1(),
            token= {
                "prompt": self.get_prompt_token_count() / len_benchmark,
                "completion": self.get_completion_token_count() / len_benchmark,
                "total": self.get_total_token_count() / len_benchmark,
            })

def _compute_token_count(documents: List[NodeWithScore]):
    return reduce(
        lambda count, document: count + count_token(document.get_content()),
        documents,
        0,
    )

def _run_rag_benchmark(
    benchmark_name: str, queryset_name: str, retriever: BaseRetriever, queries: list[Query]
) -> BenchmarkResult:
    evaluator = Evaluator()
    result_data = []
    name = f"{benchmark_name}/{queryset_name}"
    for query in IncrementalBar(f"Processing {name:>34}").iter(queries):
        result_documents = retriever.retrieve(query.query)
        retrieved_endpoints = _get_endpoints_from(result_documents)
        endpoints = set(endpoint.endpoint for endpoint in retrieved_endpoints)
        true_positives = endpoints.intersection(query.solution)
        evaluator.add_total(len(endpoints))
        evaluator.add_true_positive(len(true_positives))
        evaluator.add_solution(len(query.solution))
        number_of_token = _compute_token_count(result_documents)
        evaluator.add_prompt_token(number_of_token)
        entry_dict = BenchmarkEntry(
            query=query.query,
            solution=query.solution,
            retrieved_endpoints=retrieved_endpoints,
            recall=len(true_positives) / len(query.solution),
            precision=(
                len(true_positives) / len(endpoints) if len(endpoints) > 0 else None
            ),
            token=number_of_token)
        result_data.append(entry_dict)
    return BenchmarkResult(
        name=queryset_name, measurements=evaluator.get_result(len(queries)), entries=result_data
    )


def run_rag_benchmark(
    benchmark_name: str, queryset: Queryset, retriever: BaseRetriever
) -> BenchmarkResult:
    return _run_rag_benchmark(benchmark_name, queryset.name, retriever, queryset.queries)
