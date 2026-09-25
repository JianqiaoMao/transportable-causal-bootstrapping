"""Graph separation using only the public grapl adjacency interface."""
from itertools import combinations


def is_m_separated(graph, source, target, condition=()):
    """Test m-separation via the canonical latent DAG's ancestral moral graph.

    Each bidirected edge is replaced by a distinct unobserved common parent.
    No graph mutation or locally patched grapl methods are required.
    """
    nodes = set(graph.nodes())
    condition = set(condition)
    if ({source, target} | condition) - nodes:
        raise ValueError("Separation query contains nodes absent from the graph.")
    if {source, target} & condition:
        raise ValueError("Separation endpoints must not be conditioned on.")

    parents = {node: set(graph.pa({node})) for node in nodes}
    seen_edges = set()
    for node in nodes:
        for sibling in graph.bi({node}):
            edge = frozenset((node, sibling))
            if edge not in seen_edges:
                seen_edges.add(edge)
                latent = object()  # Cannot collide with observed variable names.
                parents[latent] = set()
                parents[node].add(latent)
                parents[sibling].add(latent)

    ancestors = {source, target} | condition
    pending = list(ancestors)
    while pending:
        for parent in parents[pending.pop()]:
            if parent not in ancestors:
                ancestors.add(parent)
                pending.append(parent)

    neighbors = {node: set() for node in ancestors}
    for child in ancestors:
        family = parents[child] & ancestors
        for parent in family:
            neighbors[child].add(parent)
            neighbors[parent].add(child)
        for left, right in combinations(family, 2):
            neighbors[left].add(right)
            neighbors[right].add(left)

    visited = set(condition)
    pending = [source]
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        if node == target:
            return False
        visited.add(node)
        pending.extend(neighbors[node] - visited)
    return True
