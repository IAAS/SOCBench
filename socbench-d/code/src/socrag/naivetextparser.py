from typing import List, Sequence, Any, Callable
from pydantic import Field
from llama_index.core.schema import BaseNode, TextNode, NodeRelationship
from llama_index.core.node_parser import NodeParser
import tiktoken

class NaiveTextParser(NodeParser):
    chunk_size: int = Field(default=1024)
    chunk_overlap: int = Field(default=1024)
    tokenizer: Callable = Field(default=None)
    decoder: Callable = Field(default=None)
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        super().__init__()
        assert chunk_size > 0
        assert chunk_overlap >= 0
        assert 2 * chunk_overlap < chunk_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o").encode
        self.decoder = tiktoken.encoding_for_model("gpt-4o").decode


    def _parse_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        for node in nodes:
            all_nodes.extend(self._split_node(node))
        return all_nodes

    def _split_node(self, root_node: BaseNode) -> List[TextNode]:
        text = root_node.get_content()
        nodes: List[TextNode] = []
        tokens = self.tokenizer(text)
        start_index = 0
        stop_index = self.chunk_size
        while stop_index < len(tokens):
            chunk_tokens = tokens[start_index:stop_index]
            nodes.append(self._create_node(root_node, chunk_tokens))
            start_index += self.chunk_size - self.chunk_overlap
            stop_index = start_index + self.chunk_size
        if start_index != stop_index:
            chunk_tokens = tokens[start_index:len(tokens)]
            nodes.append(self._create_node(root_node, chunk_tokens))
        return nodes

    def _create_node(self, root_node, chunk_tokens) -> TextNode:
        return TextNode(
            text=self.decoder(chunk_tokens),
            relationships={
                NodeRelationship.SOURCE: root_node.as_related_node_info()
            },
        )
