from typing import Dict, List
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.schema import NodeWithScore, QueryBundle
import itertools


class CraftRetriever(BaseRetriever):

    def __init__(
        self,
        summary_retriever: BaseRetriever,
        endpoint_name_retriever: BaseRetriever,
        endpoint_description_retriever: BaseRetriever,
        similarity_top_k,
        callback_manager: CallbackManager | None = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(callback_manager, None, None, verbose)
        self.summary_retriever = summary_retriever
        self.endpoint_name_retriever = endpoint_name_retriever
        self.endpoint_description_retriever = endpoint_description_retriever
        self.similarity_top_k = similarity_top_k

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        for i in itertools.count():
            results = self._retrieve_each_k_results(
                query_bundle, (i + 1) * self.similarity_top_k
            )
            if len(results) >= self.similarity_top_k:
                assert len(results) == self.similarity_top_k
                return [item.node for item in results]

    def _retrieve_each_k_results(self, query_bundle: QueryBundle, k: int):
        self.summary_retriever.similarity_top_k = k
        self.endpoint_name_retriever.similarity_top_k = k
        self.endpoint_description_retriever.similarity_top_k = k
        summary_retrieved = self.summary_retriever.retrieve(query_bundle)
        endpoint_name_retrieved = self.endpoint_name_retriever.retrieve(query_bundle)
        endpoint_description_retrieved = self.endpoint_description_retriever.retrieve(
            query_bundle
        )
        summary_results = set()
        endpoint_name_results = set()
        endpoint_description_results = set()
        results = []
        for summary, endpoint_name, endpoint_description in zip(
            summary_retrieved, endpoint_name_retrieved, endpoint_description_retrieved
        ):
            summary_results.add(ResultItem(summary))
            results = self._compute_results(
                summary_results, endpoint_name_results, endpoint_description_results
            )
            if len(results) >= self.similarity_top_k:
                return results
            endpoint_name_results.add(ResultItem(endpoint_name))
            results = self._compute_results(
                summary_results, endpoint_name_results, endpoint_description_results
            )
            if len(results) >= self.similarity_top_k:
                return results
            endpoint_description_results.add(ResultItem(endpoint_description))
            results = self._compute_results(
                summary_results, endpoint_name_results, endpoint_description_results
            )
            if len(results) >= self.similarity_top_k:
                return results
        return results

    def _compute_results(
        self, summary_results, endpoint_name_results, endpoint_description_results
    ) -> set:
        return (
            (summary_results & endpoint_name_results)
            | (summary_results & endpoint_description_results)
            | (endpoint_description_results | endpoint_name_results)
        )

    def _results_to_set(self, results: List[NodeWithScore]) -> set:
        result_items = [ResultItem(node) for node in results]
        return set(result_items)


class ResultItem:
    def __init__(self, node: NodeWithScore) -> None:
        self.node = node
        self.content = node.get_content()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ResultItem):
            return self.content == other.content
        return False

    def __hash__(self) -> int:
        return hash(self.content)
