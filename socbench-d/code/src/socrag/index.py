import os.path

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Document,
)
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.node_parser import NodeParser
from llama_index.core.node_parser.text.token import TokenTextSplitter
from llama_index.core.node_parser.file.json import JSONNodeParser
from socrag.craft import CraftRetriever
from socrag.endpointmetadataparser import (
    EndpointMetadataParser,
    EndpointTokenTextSplitter,
)
from socrag.endpointparser import (
    EndpointDescriptionParser,
    EndpointJsonParser,
    EndpointNameParser,
    EndpointParser,
    FieldParser,
    ThinParser,
)
from socrag.llmparser import (
    QuestionParser,
    SummaryParser,
)
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
from os.path import join, isdir
import shutil
from pydantic import BaseModel
import logging

class ChunkingStrategy(BaseModel):
    path: str
    transformations: list[NodeParser]


CHUNKING_STRATEGIES = {
    "WHOLE_DOCUMENT_100_0": ChunkingStrategy(
        path="whole_document_100_0",
        transformations=[
            EndpointMetadataParser(),
            TokenTextSplitter(chunk_size=100, chunk_overlap=0),
            EndpointTokenTextSplitter(),
        ],
    ),
    "WHOLE_DOCUMENT_100_20": ChunkingStrategy(
        path="whole_document_100_20",
        transformations=[
            EndpointMetadataParser(),
            TokenTextSplitter(chunk_size=100, chunk_overlap=20),
            EndpointTokenTextSplitter(),
        ],
    ),
    "WHOLE_DOCUMENT_200_0": ChunkingStrategy(
        path="whole_document_200_0",
        transformations=[
            EndpointMetadataParser(),
            TokenTextSplitter(chunk_size=200, chunk_overlap=0),
            EndpointTokenTextSplitter(),
        ],
    ),
    "WHOLE_DOCUMENT_200_20": ChunkingStrategy(
        path="whole_document_200_20",
        transformations=[
            EndpointMetadataParser(),
            TokenTextSplitter(chunk_size=200, chunk_overlap=20),
            EndpointTokenTextSplitter(),
        ],
    ),
    # "WHOLE_DOCUMENT_1024_0": ChunkingStrategy(
    #     path="whole_document_1024_0",
    #     transformations=[
    #         EndpointMetadataParser(),
    #         TokenTextSplitter(chunk_size=1024, chunk_overlap=0),
    #         EndpointTokenTextSplitter(),
    #     ],
    # ),
    "ENDPOINT_SPLIT_1024_0": ChunkingStrategy(
        path="endpoint_split_1024_0",
        transformations=[
            EndpointParser(),
            TokenTextSplitter(chunk_size=1024, chunk_overlap=0),
        ],
    ),
    # "WHOLE_DOCUMENT_1024_50": ChunkingStrategy(
    #     path="whole_document_1024_50",
    #     transformations=[
    #         EndpointMetadataParser(),
    #         TokenTextSplitter(chunk_size=1024, chunk_overlap=50),
    #         EndpointTokenTextSplitter(),
    #     ],
    # ),
    "ENDPOINT_SPLIT_1024_20": ChunkingStrategy(
        path="endpoint_split_1024_20",
        transformations=[
            EndpointParser(),
            TokenTextSplitter(chunk_size=1024, chunk_overlap=20),
        ],
    ),
    # "WHOLE_DOCUMENT_2048_0": ChunkingStrategy(
    #     path="whole_document_2048_0",
    #     transformations=[
    #         EndpointMetadataParser(),
    #         TokenTextSplitter(chunk_size=2048, chunk_overlap=0),
    #         EndpointTokenTextSplitter(),
    #     ],
    # ),
    # "ENDPOINT_SPLIT_2048_0": ChunkingStrategy(
    #     path="endpoint_split_2048_0",
    #     transformations=[
    #         EndpointParser(),
    #         TokenTextSplitter(chunk_size=2048, chunk_overlap=0),
    #     ],
    # ),
    # "WHOLE_DOCUMENT_2048_50": ChunkingStrategy(
    #     path="whole_document_2048_50",
    #     transformations=[
    #         EndpointMetadataParser(),
    #         TokenTextSplitter(chunk_size=2048, chunk_overlap=50),
    #         EndpointTokenTextSplitter(),
    #     ],
    # ),
    # "ENDPOINT_SPLIT_2048_50": ChunkingStrategy(
    #     path="endpoint_split_2048_50",
    #     transformations=[
    #         EndpointParser(),
    #         TokenTextSplitter(chunk_size=2048, chunk_overlap=50),
    #     ],
    # ),
    # "WHOLE_DOCUMENT_4096_0": ChunkingStrategy(
    #     path="whole_document_4096_0",
    #     transformations=[
    #         EndpointMetadataParser(),
    #         TokenTextSplitter(chunk_size=4096, chunk_overlap=0),
    #         EndpointTokenTextSplitter(),
    #     ],
    # ),
    # "ENDPOINT_SPLIT_4096_0": ChunkingStrategy(
    #     path="endpoint_split_4096_0",
    #     transformations=[
    #         EndpointParser(),
    #         TokenTextSplitter(chunk_size=4096, chunk_overlap=0),
    #     ],
    # ),
    # "WHOLE_DOCUMENT_4096_50": ChunkingStrategy(
    #     path="whole_document_4096_50",
    #     transformations=[
    #         EndpointMetadataParser(),
    #         TokenTextSplitter(chunk_size=4096, chunk_overlap=50),
    #         EndpointTokenTextSplitter(),
    #     ],
    # ),
    # "ENDPOINT_SPLIT_4096_50": ChunkingStrategy(
    #     path="endpoint_split_4096_50",
    #     transformations=[
    #         EndpointParser(),
    #         TokenTextSplitter(chunk_size=4096, chunk_overlap=50),
    #     ],
    # ),
    # "WHOLE_DOCUMENT_8191_0": ChunkingStrategy(
    #     path="whole_document_8191_0",
    #     transformations=[
    #         EndpointMetadataParser(),
    #         TokenTextSplitter(chunk_size=8191, chunk_overlap=0),
    #         EndpointTokenTextSplitter(),
    #     ],
    # ),
    # "ENDPOINT_SPLIT_8191_0": ChunkingStrategy(
    #     path="endpoint_split_8191_0",
    #     transformations=[
    #         EndpointParser(),
    #         TokenTextSplitter(chunk_size=8191, chunk_overlap=0),
    #     ],
    # ),
    # "WHOLE_DOCUMENT_8191_50": ChunkingStrategy(
    #     path="whole_document_8191_50",
    #     transformations=[
    #         EndpointMetadataParser(),
    #         TokenTextSplitter(chunk_size=8191, chunk_overlap=50),
    #         EndpointTokenTextSplitter(),
    #     ],
    # ),
    # "ENDPOINT_SPLIT_8191_50": ChunkingStrategy(
    #     path="endpoint_split_8191_50",
    #     transformations=[
    #         EndpointParser(),
    #         TokenTextSplitter(chunk_size=8191, chunk_overlap=50),
    #     ],
    # ),
    "ENDPOINT_SPLIT_THIN": ChunkingStrategy(
        path="endpoint_split_thin", transformations=[ThinParser()]
    ),
    "ENDPOINT_SPLIT_FIELD": ChunkingStrategy(
        path="endpoint_split_field", transformations=[FieldParser()]
    ),
    "JSON_100_0": ChunkingStrategy(
        path="json_100_0",
        transformations=[
            EndpointMetadataParser(),
            JSONNodeParser(),
            TokenTextSplitter(chunk_size=100, chunk_overlap=0),
            EndpointTokenTextSplitter(),
        ],
    ),
    "JSON_100_20": ChunkingStrategy(
        path="json_100_20",
        transformations=[
            EndpointMetadataParser(),
            JSONNodeParser(),
            TokenTextSplitter(chunk_size=100, chunk_overlap=20),
            EndpointTokenTextSplitter(),
        ],
    ),
    # "JSON_1024_0": ChunkingStrategy(
    #     path="json_1024_0",
    #     transformations=[
    #         EndpointMetadataParser(),
    #         JSONNodeParser(),
    #         TokenTextSplitter(chunk_size=1024, chunk_overlap=0),
    #         EndpointTokenTextSplitter(),
    #     ],
    # ),
    # "JSON_2048_0": ChunkingStrategy(
    #     path="json_2048_0",
    #     transformations=[
    #         EndpointMetadataParser(),
    #         JSONNodeParser(),
    #         TokenTextSplitter(chunk_size=2048, chunk_overlap=0),
    #         EndpointTokenTextSplitter(),
    #     ],
    # ),
    # "JSON_4096_0": ChunkingStrategy(
    #     path="json_4096_0",
    #     transformations=[
    #         EndpointMetadataParser(),
    #         JSONNodeParser(),
    #         TokenTextSplitter(chunk_size=4096, chunk_overlap=0),
    #         EndpointTokenTextSplitter(),
    #     ],
    # ),
    # "JSON_8191_0": ChunkingStrategy(
    #     path="json_8191_0",
    #     transformations=[
    #         EndpointMetadataParser(),
    #         JSONNodeParser(),
    #         TokenTextSplitter(chunk_size=8191, chunk_overlap=0),
    #         EndpointTokenTextSplitter(),
    #     ],
    # ),
    # "JSON_1024_50": ChunkingStrategy(
    #     path="json_1024_50",
    #     transformations=[
    #         EndpointMetadataParser(),
    #         JSONNodeParser(),
    #         TokenTextSplitter(chunk_size=1024, chunk_overlap=50),
    #         EndpointTokenTextSplitter(),
    #     ],
    # ),
    # "JSON_2048_50": ChunkingStrategy(
    #     path="json_2048_50",
    #     transformations=[
    #         EndpointMetadataParser(),
    #         JSONNodeParser(),
    #         TokenTextSplitter(chunk_size=2048, chunk_overlap=50),
    #         EndpointTokenTextSplitter(),
    #     ],
    # ),
    # "JSON_4096_50": ChunkingStrategy(
    #     path="json_4096_50",
    #     transformations=[
    #         EndpointMetadataParser(),
    #         JSONNodeParser(),
    #         TokenTextSplitter(chunk_size=4096, chunk_overlap=50),
    #         EndpointTokenTextSplitter(),
    #     ],
    # ),
    # "JSON_8191_50": ChunkingStrategy(
    #     path="json_8191_50",
    #     transformations=[
    #         EndpointMetadataParser(),
    #         JSONNodeParser(),
    #         TokenTextSplitter(chunk_size=8191, chunk_overlap=50),
    #         EndpointTokenTextSplitter(),
    #     ],
    # ),
    "ENDPOINT_JSON": ChunkingStrategy(
        path="endpoint_json",
        transformations=[
            EndpointJsonParser(),
            TokenTextSplitter(chunk_size=1024, chunk_overlap=0)
        ],
    ),
    "QUERY_EXTRACTION": ChunkingStrategy(
        path="query_extraction", transformations=[QuestionParser()]
    ),
    "SUMMARY": ChunkingStrategy(path="summary", transformations=[SummaryParser()]),
    "ENDPOINT_DESCRIPTION": ChunkingStrategy(
        path="endpoint_description", transformations=[EndpointDescriptionParser()]
    ),
    "ENDPOINT_NAME": ChunkingStrategy(
        path="endpoint_name", transformations=[EndpointNameParser()]
    ),
    "CRAFT": ChunkingStrategy(path="craft", transformations=[]),
}


def _create_index(
    openapis: list[str], embed_model, dimension: int, path: str, transformations
):
    logging.info(f"Creating index {path}")
    if isdir(path):
        shutil.rmtree(path)
    faiss_index = faiss.IndexFlatL2(dimension)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    semantic_documents = [
        Document(text=specification, mimetype="application/json")
        for specification in openapis
    ]
    index = VectorStoreIndex.from_documents(
        semantic_documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=transformations,
    )
    index.storage_context.persist(path)
    return index


def _load_index(embed_model, path):
    logging.info(f"Loading index {path}")
    vector_store = FaissVectorStore.from_persist_dir(path)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=path
    )
    return load_index_from_storage(
        storage_context=storage_context, embed_model=embed_model
    )


def get_index(
    benchmark_name: str,
    model_name: str,
    queryset_name: str,
    chunking_strategy_name: str,
    openapis: list[str],
    embed_model,
    dimension: int,
) -> VectorStoreIndex:
    chunking_strategy = CHUNKING_STRATEGIES[chunking_strategy_name]
    assert chunking_strategy
    path = os.path.join("data", benchmark_name, model_name, chunking_strategy.path, queryset_name, "vector_store")
    return (
        _load_index(embed_model, path)
        if isdir(path)
        else _create_index(
            openapis,
            embed_model,
            dimension,
            path,
            chunking_strategy.transformations,
        )
    )


def get_retriever(
    benchmark_name: str,
    model_name: str,
    queryset_name: str,
    chunking_strategy_name: str,
    openapis: list[str],
    embed_model,
    dimension,
    top_k,
) -> BaseRetriever:
    if chunking_strategy_name == "CRAFT":
        summary_index = get_index(
            benchmark_name, model_name, queryset_name, "SUMMARY", openapis, embed_model, dimension
        )
        summary_retriever = summary_index.as_retriever(
            similarity_top_k=top_k, embed_model=embed_model
        )
        endpoint_name_index = get_index(
            benchmark_name, model_name, queryset_name, "ENDPOINT_NAME", openapis, embed_model, dimension
        )
        endpoint_name_retriever = endpoint_name_index.as_retriever(
            similarity_top_k=top_k, embed_model=embed_model
        )
        endpoint_description_index = get_index(
            benchmark_name, model_name, queryset_name, "ENDPOINT_DESCRIPTION", openapis, embed_model, dimension
        )
        endpoint_description_retriever = endpoint_description_index.as_retriever(
            similarity_top_k=top_k, embed_model=embed_model
        )
        return CraftRetriever(
            summary_retriever,
            endpoint_name_retriever,
            endpoint_description_retriever,
            top_k,
        )
    index = get_index(
        benchmark_name, model_name, queryset_name, chunking_strategy_name, openapis, embed_model, dimension
    )
    index.as_query_engine()
    return index.as_retriever(similarity_top_k=top_k, embed_model=embed_model)
