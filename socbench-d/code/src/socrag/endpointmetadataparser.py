from typing import List, Sequence, Any
from llama_index.core.schema import BaseNode, TextNode, NodeRelationship
from llama_index.core.node_parser import NodeParser
from llama_index.core.node_parser.text.token import TokenTextSplitter
from socrag.file import extract_endpoints
import json


class EndpointMetadataParser(NodeParser):
    def _parse_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        for node in nodes:
            self._add_endpoints(node)
        return nodes

    def _add_endpoints(self, root_node: BaseNode):
        endpoints = extract_endpoints(root_node.get_content())
        root_node.metadata.clear()
        root_node.metadata["endpoints"] = json.dumps(endpoints)
        root_node.excluded_embed_metadata_keys = ["endpoints"]
        root_node.excluded_llm_metadata_keys = ["endpoints"]


class EndpointTokenTextSplitter(NodeParser):
    def _parse_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        for node in nodes:
            self._filter_endpoints(node)
        return nodes

    def _filter_endpoints(self, chunk: BaseNode):
        all_endpoints = json.loads(chunk.metadata["endpoints"])
        actual_endpoints = [
            endpoint
            for endpoint in all_endpoints
            if endpoint.split()[1] in chunk.get_content()
        ]
        chunk.metadata["endpoints"] = json.dumps(actual_endpoints)
