"""Portable m-separation checks, including an independent path oracle."""
from itertools import combinations, product
import random

import pytest
from grapl.admg import ADMG
from grapl.dsl import GraplDSL
from tcb._graph import is_m_separated


@pytest.mark.parametrize("edges,condition,separated", [
    ("A -> B; B -> C;", (), False),
    ("A -> B; B -> C;", ("B",), True),
    ("B -> A; B -> C;", ("B",), True),
    ("A -> B; C -> B;", (), True),
    ("A -> B; C -> B;", ("B",), False),
    ("A <-> B; B <-> C;", (), True),
    ("A <-> B; B <-> C;", ("B",), False),
    ("A <-> B; B <-> C; B -> D;", ("D",), False),
    ("A <-> C;", (), False),
])
def test_separation_motifs(edges, condition, separated):
    graph = GraplDSL().readgrapl("A; B; C; D; " + edges)
    assert is_m_separated(graph, "A", "C", condition) is separated


def active_path_exists(graph, source, target, condition):
    """Enumerate simple mixed-edge paths and apply the collider definition."""
    ancestors = graph.an(set(condition))
    neighbors = {v: [] for v in graph.nodes()}
    for node in graph.nodes():
        for child in graph.ch({node}):
            neighbors[node].append((child, False, True))
            neighbors[child].append((node, True, False))
        for sibling in graph.bi({node}):
            neighbors[node].append((sibling, True, True))

    def walk(node, incoming_head, visited):
        if node == target:
            return True
        for other, outgoing_head, next_head in neighbors[node]:
            if other in visited:
                continue
            if incoming_head is not None:
                collider = incoming_head and outgoing_head
                if collider and node not in ancestors:
                    continue
                if not collider and node in condition:
                    continue
            if walk(other, next_head, visited | {other}):
                return True
        return False

    return walk(source, None, {source})


def check_graph(nodes, bits):
    pairs = list(combinations(nodes, 2))
    graph = ADMG()
    for node in nodes:
        graph.addvar(node)
    for (left, right), directed, bidirected in zip(pairs, bits[:len(pairs)], bits[len(pairs):]):
        if directed:
            graph.addedges(right, parents={left})
        if bidirected:
            graph.addedges(left, bidirects={right})
    graph.connect()
    for source, target in pairs:
        rest = [v for v in nodes if v not in (source, target)]
        for mask in product((False, True), repeat=len(rest)):
            condition = {v for v, chosen in zip(rest, mask) if chosen}
            expected = not active_path_exists(graph, source, target, condition)
            assert is_m_separated(graph, source, target, condition) == expected
            assert is_m_separated(graph, target, source, condition) == expected


def test_all_three_node_mixed_graphs_against_path_definition():
    for bits in product((False, True), repeat=6):
        check_graph(("A", "B", "C"), bits)


def test_four_node_mixed_graphs_against_path_definition():
    rng = random.Random(20260925)
    for _ in range(50):
        check_graph(("A", "B", "C", "D"), [rng.choice((False, True)) for _ in range(12)])
