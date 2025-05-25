from .openapiparser import OpenApiParser
from llama_index.core.schema import TextNode
from llama_index.core import Settings
from llama_index.core.node_parser.file.json import JSONNodeParser
import json

class EndpointParser(OpenApiParser):
    def _create_endpoint_node(self, node: TextNode, endpoint: object) -> None:
        pass

    def _filter_endpoint(self, endpoint: object) -> None:
        pass


class ThinParser(EndpointParser):
    def _filter_endpoint(self, endpoint: object) -> None:
        endpoint.pop("requestBody", None)
        self._remove_examples(endpoint)

    def _remove_examples(self, endpoint):
        if isinstance(endpoint, dict):
            for _, item in endpoint.items():
                self._remove_examples(item)
            if "examples" in endpoint:
                del endpoint["examples"]


class FieldParser(EndpointParser):
    def _create_endpoint_node(self, node: TextNode, endpoint: object) -> None:
        fields = f"""Title: {node.source_node.metadata["title"]}
Service description: {node.source_node.metadata["description"]}
Endpoint: {node.metadata["verb"]} {node.metadata["path"]}
Endpoint description: {endpoint["description"]}"""
        node.embedding = Settings.embed_model.get_text_embedding(fields)


class EndpointDescriptionParser(EndpointParser):
    def _create_endpoint_node(self, node: TextNode, endpoint: object) -> None:
        node.embedding = Settings.embed_model.get_text_embedding(
            endpoint["description"]
        )


class EndpointNameParser(EndpointParser):
    def _create_endpoint_node(self, node: TextNode, endpoint: object) -> None:
        node.embedding = Settings.embed_model.get_text_embedding(node.metadata["path"])

class EndpointJsonParser(EndpointParser):
    def _create_endpoint_node(self, node: TextNode, endpoint: object) -> None:
        parser = JSONNodeParser()
        node.set_content(json.dumps(endpoint))
        json_nodes = parser.get_nodes_from_node(node)
        assert len(json_nodes) == 1
        node.set_content(json_nodes[0].get_content())
