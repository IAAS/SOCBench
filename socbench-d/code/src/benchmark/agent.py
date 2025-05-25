from progress.bar import IncrementalBar
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.callbacks import TokenCountingHandler
from llama_index.core import Settings
from benchmark import Queryset
from benchmark.encoding import get_encoding, count_token
from benchmark.llama_index import Evaluator, BenchmarkEntry, BenchmarkResult, PreciseTokenCount, RetrievedEndpoint
from socrag.agent import (
    OpenAiSocQueryAgent,
    OpenAiSocSummaryAgent,
    ReactSocQueryAgent,
    ReactSocSummaryAgent,
    SocAgent,
)
from socrag.file import extract_endpoints


def _get_endpoints_from(response: str, all_endpoints: list[str]) -> list[RetrievedEndpoint]:
    return [RetrievedEndpoint(endpoint=endpoint, confidence=1) for endpoint in all_endpoints if endpoint in response]

def _get_all_endpoints_from(queryset: Queryset) -> list[str]:
    return [endpoint for openapi in queryset.openapis for endpoint in extract_endpoints(openapi)]

def _run_agent_benchmark(benchmark_name: str, queryset: Queryset, agent: SocAgent) -> BenchmarkResult:
    token_counter = TokenCountingHandler(tokenizer=get_encoding().encode)
    token_counter.reset_counts()
    Settings.callback_manager.add_handler(token_counter)
    evaluator = Evaluator()
    result_data = []
    name = f"{benchmark_name}/{queryset.name}"
    all_endpoints = _get_all_endpoints_from(queryset)
    for query in IncrementalBar(f"Processing {name:>34}").iter(queryset.queries):
        agent.reset()
        token_counter.reset_counts()
        response = agent.query(query.query)
        retrieved_endpoints = _get_endpoints_from(response, all_endpoints)
        endpoints = set(endpoint.endpoint for endpoint in retrieved_endpoints)
        true_positives = endpoints.intersection(query.solution)
        evaluator.add_total(len(endpoints))
        evaluator.add_true_positive(len(true_positives))
        evaluator.add_solution(len(query.solution))
        number_of_token = count_token(response)
        evaluator.add_prompt_token(number_of_token)
        entry_dict = BenchmarkEntry(
            query=query.query,
            solution=query.solution,
            retrieved_endpoints=retrieved_endpoints,
            recall=len(true_positives) / len(query.solution),
            precision=(
                len(true_positives) / len(endpoints) if len(endpoints) > 0 else None
            ),
            token=PreciseTokenCount(prompt_token=token_counter.prompt_llm_token_count, completion_token=token_counter.completion_llm_token_count, response_token=number_of_token),
            response=response,
        )
        result_data.append(entry_dict)
        evaluator.add_completion_token(token_counter.completion_llm_token_count)
        evaluator.add_prompt_token(number_of_token + token_counter.prompt_llm_token_count)
    Settings.callback_manager.remove_handler(token_counter)
    return BenchmarkResult(
        name=queryset.name, measurements=evaluator.get_result(len(queryset.queries)), entries=result_data
    )

def run_openai_agent_benchmark_query(benchmark_name: str, queryset: Queryset, query_engine: BaseQueryEngine, llm) -> BenchmarkResult:
    agent = OpenAiSocQueryAgent(query_engine, llm)
    return _run_agent_benchmark(benchmark_name, queryset, agent)


def run_react_agent_benchmark_query(benchmark_name: str, queryset: Queryset, query_engine: BaseQueryEngine, llm) -> BenchmarkResult:
    agent = ReactSocQueryAgent(query_engine, llm)
    return _run_agent_benchmark(benchmark_name, queryset, agent)


def run_openai_agent_benchmark_summary(benchmark_name: str, queryset: Queryset, dimension, embed_model, llm, top_k) -> BenchmarkResult:
    agent = OpenAiSocSummaryAgent(queryset.openapis, dimension, embed_model, llm, top_k)
    return _run_agent_benchmark(benchmark_name, queryset, agent)


def run_react_agent_benchmark_summary(benchmark_name: str, queryset: Queryset, dimension, embed_model, llm, top_k) -> BenchmarkResult:
    agent = ReactSocSummaryAgent(queryset.openapis, dimension, embed_model, llm, top_k)
    return _run_agent_benchmark(benchmark_name, queryset, agent)
