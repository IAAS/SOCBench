from typing import List, Sequence, Any
from llama_index.core.schema import BaseNode, TextNode, NodeRelationship
from llama_index.core.node_parser import NodeParser
import json


class OpenApiParser(NodeParser):
    def _parse_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        for node in nodes:
            all_nodes.extend(self._split_node(node))
        return all_nodes

    def _split_node(self, root_node: BaseNode) -> List[TextNode]:
        nodes: List[TextNode] = []
        specification = json.loads(root_node.get_content())
        for path, path_content in specification["paths"].items():
            path_node = TextNode(
                text=json.dumps(path),
                relationships={
                    NodeRelationship.SOURCE: root_node.as_related_node_info()
                },
            )
            path_node.metadata["title"] = specification["info"]["title"]
            path_node.metadata["description"] = specification["info"]["description"]
            for verb, endpoint_content in path_content.items():
                if not "description" in endpoint_content:
                    continue
                self._filter_endpoint(endpoint_content)
                endpoint_node = TextNode(
                    text=f"Endpoint: {verb.upper()} {path}\nSpecification:\n{json.dumps(endpoint_content)}",
                    relationships={
                        NodeRelationship.SOURCE: path_node.as_related_node_info()
                    },
                )
                endpoint_node.metadata["verb"] = verb.upper()
                endpoint_node.metadata["path"] = path
                endpoint_node.metadata["endpoints"] = json.dumps(
                    [f"{verb.upper()} {path}"]
                )
                endpoint_node.excluded_embed_metadata_keys = ["endpoints"]
                endpoint_node.excluded_llm_metadata_keys = ["endpoints"]
                self._create_endpoint_node(endpoint_node, endpoint_content)
                nodes.append(endpoint_node)
        return nodes

    def _filter_endpoint(self, endpoint: object) -> None: ...

    def _create_endpoint_node(self, node: TextNode, endpoint: object) -> None: ...
