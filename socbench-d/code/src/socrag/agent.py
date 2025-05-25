import json
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.agent import ReActAgent, AgentRunner
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from pydantic import Field
from llama_index.core.schema import BaseNode, TextNode, Document
from llama_index.core.node_parser import NodeParser
from typing import List, Sequence, Any
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
import logging
from socrag.file import read_file, extract_endpoints
import os


def get_tools_query_engine(query_engine: BaseQueryEngine):
    query_engine_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="endpoint_fetcher",
            description="Provides the endpoint specification for a specific endpoint from the OpenAPI specification",
        ),
        resolve_input_errors=False,
    )
    return [query_engine_tool]


class SummaryIndexParser(NodeParser):
    stored_summary: dict = Field(default_factory=dict)
    def __init__(self, path=os.path.join("data", "summary.json")):
        super(SummaryIndexParser, self).__init__()
        stored = json.loads(read_file(path))
        stored_summary = {}
        for item in stored:
            stored_summary[item["endpoint"]] = item["answer"]
        assert stored_summary
        self.stored_summary = stored_summary

    def _parse_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        for node in nodes:
            all_nodes.extend(self._extract_summary(node))
        return all_nodes

    def _extract_summary(self, node: TextNode):
        openapi = node.get_content()
        endpoints = extract_endpoints(openapi)
        nodes = []
        for endpoint in endpoints:
            answer = self.stored_summary[endpoint]
            node = TextNode(text=f"{endpoint}: {answer}")
            node.embedding = Settings.embed_model.get_text_embedding(answer)
            nodes.append(node)
        return nodes


def create_summary_index(openapis: list[str], dimension, embed_model):
    faiss_index = faiss.IndexFlatL2(dimension)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    documents = [Document(text=openapi) for openapi in openapis]
    return VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[SummaryIndexParser()],
    )


def create_get_endpoint_details(openapis: list[str]):
    endpoints = {}
    for openapi in openapis:
        openapi_json = json.loads(openapi)
        for endpoint in extract_endpoints(openapi):
            [verb, path] = endpoint.split(" ")
            endpoints[endpoint] = json.dumps(openapi_json["paths"][path][verb.lower()])

    def get_endpoint_details(
        endpoint: str,  # = Field(description="The endpoint, formated like '<verb> <path>', e.g., GET /weather", exclude=True),
    ) -> str:
        """Returns the OpenAPI endpoint specification for a specific endpoint

        :param str endpoint: The endpoint, formated like '<verb> <path>', e.g., GET /weather
        """
        if endpoint in endpoints:
            return endpoints[endpoint]
        return f"The endpoint {endpoint} does not exist!"

    return get_endpoint_details


def get_tools_summary(openapis, dimension, embed_model, top_k):
    index = create_summary_index(openapis, dimension, embed_model)
    query_engine_tool = QueryEngineTool(
        query_engine=index.as_query_engine(
            similarity_top_k=top_k, embed_model=embed_model
        ),
        metadata=ToolMetadata(
            name="endpoint_summary",
            description="Gathers and provides concise summaries of service endpoints.",
        ),
        resolve_input_errors=False,
    )
    endpoint_details_fetcher = FunctionTool.from_defaults(
        create_get_endpoint_details(openapis)
    )
    return [query_engine_tool, endpoint_details_fetcher]


class SocAgent:
    agent: AgentRunner
    def reset(self):
        assert self.agent
        self.agent.reset()

    def query(self, task) -> str:
        assert self.agent
        prompt = f"""SUMMARY:
Determine the list of required endpoints to perform the task.

TASK:
{task}

EXAMPLE:
GET /weather
POST /data

INSTRUCTIONS:
Determine the needed endpoints to fulfill the TASK from the specified OpenAPIs. The endpoints are formatted as "VERB PATH". Make sure to include all relevant endpoints. You must not fulfill the task but return the endpoints that should be called to fulfill the task. Only return the endpoints and no explanation as you are in an automated setting.
"""
        try:
            return str(self.agent.chat(prompt))
        except ValueError as error:
            logging.warn("Error in agent", exc_info=error)
            return ""


class OpenAiSocQueryAgent(SocAgent):
    def __init__(self, query_engine, llm):
        tools = get_tools_query_engine(query_engine)
        self.agent: OpenAIAgent = OpenAIAgent.from_tools(tools, verbose=True, llm=llm)


class ReactSocQueryAgent(SocAgent):
    def __init__(self, query_engine, llm):
        tools = get_tools_query_engine(query_engine)
        self.agent: ReActAgent = ReActAgent.from_tools(
            tools, verbose=True, llm=llm, max_iterations=100
        )


class OpenAiSocSummaryAgent(SocAgent):
    def __init__(self, openapis: list[str], dimension: int, embed_model, llm, top_k: int):
        tools = get_tools_summary(openapis, dimension, embed_model, top_k)
        self.agent: OpenAIAgent = OpenAIAgent.from_tools(tools, verbose=True, llm=llm)


class ReactSocSummaryAgent(SocAgent):
    def __init__(self, openapis: list[str], dimension: int, embed_model, llm, top_k: int):
        tools = get_tools_summary(openapis, dimension, embed_model, top_k)
        self.agent: ReActAgent = ReActAgent.from_tools(
            tools, verbose=True, llm=llm, max_iterations=100
        )
