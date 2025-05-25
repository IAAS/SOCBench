import os.path
from enum import Enum
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingMode
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from benchmark.llama_index import BenchmarkResultSet
from socrag.file import write_file
from socrag.index import CHUNKING_STRATEGIES, get_retriever
import benchmark
import logging
import sys

logging.basicConfig(
    filename="data/socrag.log",
    filemode='a',
    level=logging.INFO)
logging.getLogger("httpx").disabled = True
class Model(Enum):
    OpenAI = 1
    BGE = 2
    Nvidia = 3

def setup_model(model: Model):
    if model == Model.OpenAI:
        embed_model = OpenAIEmbedding(
            model="text-embedding-3-large", mode=OpenAIEmbeddingMode.SIMILARITY_MODE
        )
        EMBEDDING_DIMENSIONS = 3072
        MODEL_NAME = "oai"
    elif model == Model.BGE:
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        logger = logging.getLogger("sentence_transformers.SentenceTransformer")
        logger.setLevel(logging.WARNING)
        EMBEDDING_DIMENSIONS = 384
        MODEL_NAME = "bge_small"
    elif model == Model.Nvidia:
        embed_model = HuggingFaceEmbedding(
            model_name="nvidia/NV-Embed-v2",
            trust_remote_code=True
        )
        logger = logging.getLogger("sentence_transformers.SentenceTransformer")
        EMBEDDING_DIMENSIONS = 4096
        MODEL_NAME = "nvidia"
    else:
        raise ValueError(f"{model} is not supported")
    llm = OpenAI(model="gpt-4o-2024-11-20")
    return embed_model, EMBEDDING_DIMENSIONS, MODEL_NAME, llm

(embed_model, EMBEDDING_DIMENSIONS, MODEL_NAME, llm) = setup_model(Model.OpenAI)
Settings.embed_model = embed_model
Settings.llm = llm
Settings.chunk_overlap = 20

restbench = benchmark.get_restbench()
# COUNT_SOCBENCHD_BENCHMARKS = 5
COUNT_SOCBENCHD_BENCHMARKS = 5
socbenchd = benchmark.get_socbenchd(COUNT_SOCBENCHD_BENCHMARKS)

def compute_restbench_results(top_k, chunking_strategy_name: str, chunking_strategy_path: str):
    assert len(restbench.queries) == 1
    rest_retriever = get_retriever(
        restbench.name, MODEL_NAME, restbench.queries[0].name, chunking_strategy_name, restbench.queries[0].openapis, embed_model, EMBEDDING_DIMENSIONS, top_k
    )
    rest_results = benchmark.run_rag_benchmark(restbench.name, restbench.queries[0], rest_retriever)
    write_restbench_results(chunking_strategy_path, rest_results, top_k)

def compute_restbench_agent_results(top_k):
    assert len(restbench.queries) == 1
    rest_results = benchmark.run_openai_agent_benchmark_summary(restbench.name, restbench.queries[0], EMBEDDING_DIMENSIONS, embed_model, llm, top_k)
    write_restbench_results("agent", rest_results, top_k)

def write_restbench_results(chunking_strategy_path, rest_results, top_k):
    write_file(
        f"{restbench.name}/{MODEL_NAME}/{chunking_strategy_path}/results_{top_k}.json",
        rest_results.model_dump()
    )

def compute_socbench_results(top_k, chunking_strategy_name: str, chunking_strategy_path: str):
    assert len(socbenchd) == COUNT_SOCBENCHD_BENCHMARKS
    for socbenchd_instance in socbenchd:
        assert len(socbenchd_instance.queries) == 11
    socbenchd_retriever = [
        [
            (
                get_retriever(
                    socbenchd_instance.name,
                    MODEL_NAME,
                    queryset.name,
                    chunking_strategy_name,
                    queryset.openapis,
                    embed_model,
                    EMBEDDING_DIMENSIONS,
                    top_k,
                ),
                queryset,
                socbenchd_instance.name,
            )
            for queryset in socbenchd_instance.queries
        ]
        for socbenchd_instance in socbenchd
    ]
    assert len(socbenchd_retriever) > 0
    socbenchd_results = [
        BenchmarkResultSet(name=socbenchd_instance[0][2], results=[
                    benchmark.run_rag_benchmark(benchmark_name, queryset, retriever)
                    for (retriever, queryset, benchmark_name) in socbenchd_instance
            ]
        )
        for socbenchd_instance in socbenchd_retriever
    ]
    write_socbenchd_results(chunking_strategy_path, socbenchd_results, top_k)

def compute_socbench_agend_results(top_k):
    assert len(socbenchd) == COUNT_SOCBENCHD_BENCHMARKS
    for socbenchd_instance in socbenchd:
        assert len(socbenchd_instance.queries) == 11
    socbenchd_results = [
        BenchmarkResultSet(name=socbenchd_instance.name, results=[
                    benchmark.run_openai_agent_benchmark_summary(socbenchd_instance.name, queryset, EMBEDDING_DIMENSIONS, embed_model, llm, top_k)
                    for queryset in socbenchd_instance.queries
            ]
        )
        for socbenchd_instance in socbenchd
    ]
    write_socbenchd_results("agent", socbenchd_results, top_k)

def write_socbenchd_results(chunking_strategy_path, socbenchd_results, top_k):
    for socbenchd_instance, results in zip(socbenchd, socbenchd_results):
        write_file(
            f"{socbenchd_instance.name}/{MODEL_NAME}/{chunking_strategy_path}/results_{top_k}.json",
            results.model_dump()
        )


def compute_results(top_k, chunking_strategy_name: str):
    compute_restbench_results(top_k, chunking_strategy_name, CHUNKING_STRATEGIES[chunking_strategy_name].path)
    compute_socbench_results(top_k, chunking_strategy_name, CHUNKING_STRATEGIES[chunking_strategy_name].path)


# for top_k in [10]:
#     for (chunking_strategy_name, chunking_strategy) in CHUNKING_STRATEGIES.items():
#         compute_results(top_k, chunking_strategy_name)
# for strategy in ["ENDPOINT_SPLIT_THIN", "ENDPOINT_SPLIT_FIELD", "ENDPOINT_SPLIT_1024_0", "ENDPOINT_JSON", "QUERY_EXTRACTION", "SUMMARY", "ENDPOINT_DESCRIPTION", "ENDPOINT_NAME", "CRAFT"]:
#     for top_k in [5, 10, 20]:
#         compute_results(top_k, strategy)
compute_restbench_agent_results(100)
compute_socbench_agend_results(100)
