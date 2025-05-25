from pydantic import Field
from .openapiparser import OpenApiParser
from llama_index.core.schema import TextNode, BaseNode
from llama_index.core.llms import ChatMessage
from llama_index.core import Settings
from typing import Any, List, Sequence
import json
import logging
import os


class LlmParser(OpenApiParser):
    path: str = Field(default=None)
    answers: list = Field(default=[])
    existed: dict = Field(default=None)

    def __init__(self, path=os.path.join("data", "llm.json")):
        super(LlmParser, self).__init__()
        self.path = path
        self.answers = []
        self.existed = {}
        if os.path.exists(self.path):
            with open(self.path) as file:
                self.answers = json.loads(file.read())
                self.existed = {}
                for answer in self.answers:
                    self.existed[answer["content"]] = answer["answer"]
                file.close()

    def _create_endpoint_node(self, node: TextNode, endpoint: object):
        verb = node.metadata["verb"]
        path = node.metadata["path"]
        node.embedding = self._get_embeddings(f"{verb} {path}", endpoint)

    def _filter_endpoint(self, endpoint: object) -> None:
        pass

    def _get_embeddings(self, path, endpoint):
        result = self._load_embedding(path, endpoint)
        return Settings.embed_model.get_text_embedding(result)

    def _create_embeddings(self, path, endpoint):
        messages = [ChatMessage(role="user", content=self._get_message(endpoint))]
        response = Settings.llm.chat(messages=messages)
        assert response.message.role == "assistant"
        result = response.message.content
        logging.info(result)
        self.answers.append(
            {"endpoint": path, "content": json.dumps(endpoint), "answer": result}
        )
        with open(self.path, "w") as file:
            file.write(json.dumps(self.answers, indent=4))
            file.close()
        return result

    def _load_embedding(self, path, endpoint):
        key = json.dumps(endpoint)
        if key in self.existed:
            return self.existed[key]
        return self._create_embeddings(path, endpoint)

    def _get_message(self, endpoint) -> str: ...


class SummaryParser(LlmParser):
    def __init__(self, path=os.path.join("data", "summary.json")):
        super(SummaryParser, self).__init__(path)

    def _get_message(self, endpoint):
        return f"""Create a summary for the endpoint.
DOCUMENT:
{json.dumps(endpoint)}

EXAMPLE:
Input: A weather service endpoint.
Output: The endpoint GET /weather return the current weather with status code 200.

TASK:
Create a summary the endpoint given in the DOCUMENT section. The summary should cover all relevant information. The purpose of the summary is the usage in a semantic search engine, i.e., all usefull information should appear.

INSTRUCTIONS:
Answer the given task. Do not state any other information as described in the TASK. Return the summary as running text without formating.
"""


WEATHER_ENDPOINT = """"/weather": {
  "get": {
    "summary": "Get current weather by city",
    "parameters": [
      {
        "name": "city",
        "in": "query",
        "required": true,
        "schema": {
          "type": "string"
        }
      },
      {
        "name": "units",
        "in": "query",
        "required": false,
        "schema": {
          "type": "string",
          "enum": ["metric", "imperial", "standard"],
          "default": "metric"
        }
      }
    ],
    "responses": {
      "200": {
        "description": "Success"
      },
      "400": {
        "description": "Invalid request"
      },
      "500": {
        "description": "Server error"
      }
    }
  }
}
"""


class ConciseSummaryParser(LlmParser):
    def __init__(self):
        super(ConciseSummaryParser, self).__init__(
            os.path.join("data", "summary_parser_concise.json")
        )

    def _get_message(self, endpoint):
        return f"""Create a summary for the endpoint.
DOCUMENT:
{json.dumps(endpoint)}

EXAMPLE:
Input: {WEATHER_ENDPOINT}
Output: The weather endpoint /weather provides current weather information based on a specified city. It accepts two query parameters: city (required) to specify the city name, and units (optional) to determine the measurement units (metric, imperial, or standard). The response includes weather details like temperature, humidity, and wind speed. Common response codes are 200 for success, 400 for invalid requests, and 500 for server errors.

TASK:
Create a concise summary of the endpoint given in the DOCUMENT section. The summary should cover all relevant information. The purpose of the summary is the usage in a semantic search engine, i.e., all useful information should appear.

INSTRUCTIONS:
Answer the given task. Do not state any other information as described in the TASK. Return the summary as running text without formating.
"""


class ExtensiveSummaryParser(LlmParser):
    def __init__(self):
        super(ExtensiveSummaryParser, self).__init__(
            os.path.join("data", "summary_parser_extensive.json")
        )

    def _get_message(self, endpoint):
        return f"""Create a summary for the endpoint.
DOCUMENT:
{json.dumps(endpoint)}

EXAMPLE:
Input: {WEATHER_ENDPOINT}
Output: The /weather endpoint of the Weather Service API provides real-time weather information for a specified city. It supports a GET request with two query parameters: city (required) and units (optional). The city parameter specifies the location, while the units parameter determines the measurement system (metric, imperial, or standard). By default, temperatures are returned in Celsius if no unit is specified.

The response includes the city name, current temperature, humidity percentage, wind speed, and a short weather description (e.g., "clear sky"). Common status codes include 200 OK for successful requests, 400 Bad Request for invalid input (such as an incorrect city name), 401 Unauthorized for missing or invalid API keys, and 500 Internal Server Error for server-side issues.

This endpoint is ideal for applications that need dynamic, location-based weather data, supporting flexible units and secure access. Developers can use it in mobile or web apps to provide users with real-time weather updates. The endpoint includes clear error handling for debugging and managing API requests. Overall, it offers a reliable solution for integrating weather data into applications.

TASK:
Create an extensive summary of the endpoint as a running text without formating given in the DOCUMENT section. The summary should cover all relevant information. The purpose of the summary is the usage in a semantic search engine, i.e., all useful information should appear.

INSTRUCTIONS:
Answer the given task. Do not state any other information as described in the TASK. Return the summary as running text without formating.
"""


class QuestionParser(LlmParser):
    def __init__(self):
        super(QuestionParser, self).__init__(
            os.path.join("data", "question_parser.json")
        )

    def _get_message(self, endpoint):
        return f"""Create a question for the endpoint.
DOCUMENT:
{json.dumps(endpoint)}

EXAMPLE:
Input: A weather service endpoint.
Output: What is the weather in London?

TASK:
State a possible question that the endpoint given in the DOCUMENT section can fulfill.

INSTRUCTIONS:
Answer the given task. Do not state any other information as described in the TASK. Return a single question without formating.
"""
