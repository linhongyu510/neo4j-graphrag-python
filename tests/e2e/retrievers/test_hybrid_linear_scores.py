#  Copyright (c) "Neo4j"
#  Neo4j Sweden AB [https://neo4j.com]
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from typing import Iterator

import pytest
from neo4j import Driver
from neo4j_graphrag.indexes import (
    create_fulltext_index,
    create_vector_index,
    drop_index_if_exists,
)
from neo4j_graphrag.neo4j_queries import (
    _build_hybrid_search_clause_query,
    _build_hybrid_search_clause_query_linear,
    get_search_query,
)
from neo4j_graphrag.retrievers import HybridCypherRetriever, HybridRetriever
from neo4j_graphrag.types import HybridSearchRanker, SearchType
from neo4j_graphrag.utils.version_utils import supports_search_clause

from ..utils import await_index_online

VECTOR_INDEX = "linear-hybrid-vector"
FULLTEXT_INDEX = "linear-hybrid-fulltext"
NODE_LABEL = "LinearHybridDocument"


@pytest.fixture(scope="module")
def hybrid_score_documents(driver: Driver) -> Iterator[None]:
    create_vector_index(
        driver,
        VECTOR_INDEX,
        label=NODE_LABEL,
        embedding_property="embedding",
        dimensions=2,
        similarity_fn="cosine",
    )
    create_fulltext_index(
        driver, FULLTEXT_INDEX, label=NODE_LABEL, node_properties=["text"]
    )
    try:
        driver.execute_query(
            f"UNWIND $documents AS doc CREATE (n:{NODE_LABEL}) SET n = doc",
            documents=[
                {"name": "best", "text": "needle", "embedding": [1.0, 0.0]},
                {"name": "weaker", "text": "needle", "embedding": [0.0, 1.0]},
                {"name": "vector_only", "text": "other", "embedding": [0.0, 1.0]},
                {"name": "fulltext_only", "text": "needle"},
            ],
        )
        await_index_online(driver, VECTOR_INDEX)
        await_index_online(driver, FULLTEXT_INDEX)
        yield
    finally:
        driver.execute_query(f"MATCH (n:{NODE_LABEL}) DETACH DELETE n")
        drop_index_if_exists(driver, VECTOR_INDEX)
        drop_index_if_exists(driver, FULLTEXT_INDEX)


@pytest.mark.usefixtures("hybrid_score_documents")
@pytest.mark.parametrize("query_path", ["legacy", "procedure", "search"])
@pytest.mark.parametrize(
    ("ranker", "alpha"),
    [
        (HybridSearchRanker.LINEAR, 0.5),
        (HybridSearchRanker.LINEAR, 0.7),
        (HybridSearchRanker.NAIVE, None),
    ],
)
def test_hybrid_channel_score_fusion(
    driver: Driver,
    query_path: str,
    ranker: HybridSearchRanker,
    alpha: float | None,
) -> None:
    tail = "RETURN node.name AS name, score"
    if query_path == "search":
        if not supports_search_clause(driver):
            pytest.skip("SEARCH clause requires Neo4j >= 2026.01")
        builder = (
            _build_hybrid_search_clause_query_linear
            if ranker == HybridSearchRanker.LINEAR
            else _build_hybrid_search_clause_query
        )
        query = f"{builder(VECTOR_INDEX, FULLTEXT_INDEX, NODE_LABEL)} {tail}"
    else:
        query, _ = get_search_query(
            SearchType.HYBRID,
            retrieval_query=tail,
            neo4j_version_is_5_23_or_above=query_path == "procedure",
            ranker=ranker,
            alpha=alpha,
        )
    records, _, _ = driver.execute_query(
        query,
        vector_index_name=VECTOR_INDEX,
        fulltext_index_name=FULLTEXT_INDEX,
        query_text="needle",
        query_vector=[1.0, 0.0],
        top_k=4,
        effective_search_ratio=1,
        alpha=alpha,
    )

    # Unit-axis vectors keep the scores exact even with index quantization.
    # The first two documents have identical fulltext relevance.
    vector_scores = {
        "best": 1.0,
        "weaker": 0.5,
        "vector_only": 0.5,
        "fulltext_only": 0,
    }
    text_scores = {"best": 1.0, "weaker": 1.0, "vector_only": 0, "fulltext_only": 1.0}
    if alpha is not None:
        expected = {
            name: alpha * score + (1 - alpha) * text_scores[name]
            for name, score in vector_scores.items()
        }
    else:
        expected = {
            name: max(score, text_scores[name]) for name, score in vector_scores.items()
        }

    assert len(records) == len(expected)
    assert {row["name"]: row["score"] for row in records} == pytest.approx(
        expected, abs=1e-6
    )
    if alpha is not None:
        assert [row["name"] for row in records] == sorted(
            expected, key=lambda name: expected[name], reverse=True
        )


@pytest.mark.usefixtures("hybrid_score_documents")
@pytest.mark.parametrize("cypher_retriever", [False, True])
def test_hybrid_retriever_preserves_equal_contributions(
    driver: Driver, cypher_retriever: bool
) -> None:
    retriever: HybridCypherRetriever | HybridRetriever
    if cypher_retriever:
        retriever = HybridCypherRetriever(
            driver, VECTOR_INDEX, FULLTEXT_INDEX, "RETURN node, score"
        )
    else:
        retriever = HybridRetriever(driver, VECTOR_INDEX, FULLTEXT_INDEX)
    result = retriever.get_search_results(
        query_text="needle",
        query_vector=[1.0, 0.0],
        top_k=4,
        ranker="linear",
        alpha=0.5,
    )

    assert len(result.records) == 4
    assert result.records[0]["node"]["name"] == "best"
    assert result.records[0]["score"] == pytest.approx(1.0)
