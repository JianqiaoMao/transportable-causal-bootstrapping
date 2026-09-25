"""Adapter for the user's revised sID and its per-term do metadata."""
from __future__ import annotations

import contextlib
import copy
import io
from dataclasses import replace

from ._sid import sid
from ._graph import is_m_separated
from .exceptions import NotTransportableError, UnsupportedFormulaError
from .symbolic import Probability, TransportFormula, names


def simplify_formula(formula, causal_graph, selection=()):
    """Optional graph-checked simplifications; all applied steps are recorded.

    Remove unnecessary conditioning from the target outcome factor by
    m-separation. Exchange a bound source action do(Z) for observation Z
    only when rule 2's separation holds in G_bar(other_actions)_under(Z).
    Query interventions are retained as actions.
    """
    if isinstance(causal_graph, str):
        from grapl.dsl import GraplDSL
        causal_graph = GraplDSL().readgrapl(causal_graph)
    graph = copy.deepcopy(causal_graph).sub(causal_graph.nodes() - set(names(selection)))
    aliases = formula.aliases
    base = lambda v: aliases.get(v, v)
    steps = []

    def separated(g, left, right, cond):
        return all(is_m_separated(g, base(a), base(b), {base(v) for v in cond}) for a in left for b in right)

    def simplify(p):
        if p.domain == 'target' and set(formula.outcomes).issubset(p.variables):
            given = set(p.given)
            for variable in sorted(given):
                if separated(graph, p.variables, (variable,), given - {variable}):
                    given.remove(variable)
                    steps.append('m-separation: remove ' + variable + ' from target outcome conditioning')
            p = replace(p, given=names(given))
        elif p.domain == 'source':
            from ._sid.algorithm import remove_incoming, remove_outgoing
            actions, given = set(p.do), set(p.given)
            for variable in sorted(actions - set(formula.interventions)):
                mutilated = copy.deepcopy(graph)
                for other in actions - {variable}:
                    remove_incoming(mutilated, base(other))
                # Rule 2 removes outgoing directed edges, retaining bidirected edges.
                remove_outgoing(mutilated, base(variable))
                if separated(mutilated, p.variables, (variable,), given | (actions - {variable})):
                    actions.remove(variable)
                    given.add(variable)
                    steps.append('do-calculus rule 2: observe ' + variable + ' in source factor')
            p = replace(p, do=names(actions), given=names(given))
        return p

    num = tuple(simplify(p) for p in formula.numerator)
    den = tuple(simplify(p) for p in formula.denominator)
    provenance = formula.provenance + ('; ' + '; '.join(steps) if steps else '')
    return replace(formula, numerator=num, denominator=den, provenance=provenance)


def from_sid(equation, *, mode="standard", graph_nodes=()):
    """Read Eqn objects directly, preserving source/target and bound symbols."""
    if equation is None:
        raise NotTransportableError("sID did not identify a transport formula.")
    rhs = equation.rhs
    if not hasattr(rhs, "_num_dovs") or not hasattr(rhs, "_den_dovs"):
        raise TypeError("sID output must carry per-term do metadata (LocalDOExpr).")
    if len(rhs.num) != len(rhs._num_dovs) or len(rhs.den) != len(rhs._den_dovs):
        raise UnsupportedFormulaError("sID term/do metadata lengths do not match.")
    marginal = names(rhs.mrg)
    aliases = {}
    original_names = set(graph_nodes)
    for symbol in marginal:
        if symbol in original_names:
            continue
        base = symbol
        while base.endswith("'") and base not in original_names:
            base = base[:-1]
        if base != symbol:
            aliases[symbol] = base

    def factor(term, do):
        return Probability(names(term), do=names(do), domain="source" if do else "target")
    num = [factor(term, do) for term, do in zip(rhs.num, rhs._num_dovs)]
    den = [factor(term, do) for term, do in zip(rhs.den, rhs._den_dovs)]
    # Read matching source joint ratios as one conditional distribution so
    # users can supply P(Z | V, do(Y)) directly without supplying its two joints.
    # Also condition target terms; derive_weights expands them when needed.
    paired = []
    for term in sorted(num, key=lambda p: (-len(p.variables), str(p))):
        matches = [d for d in den if d.domain == term.domain and d.do == term.do
                   and set(d.variables) < set(term.variables)]
        if matches:
            parent = max(matches, key=lambda p: (len(p.variables), str(p)))
            den.remove(parent)
            term = Probability(names(set(term.variables) - set(parent.variables)),
                               given=parent.variables, do=term.do, domain=term.domain)
        paired.append(term)
    return TransportFormula(
        outcomes=names(equation.lhs.num[0]), interventions=names(equation.lhs.dov),
        numerator=tuple(paired), denominator=tuple(den), marginal=marginal,
        aliases=aliases, provenance="user revised sID: " + mode,
    )


def identify(causal_graph, outcomes, interventions, selection=(), *, mode="standard"):
    """Identify P*(outcomes | do(interventions)); return a structured formula.

    This preserves the supplied sID algorithm. It does not silently change
    source conditioning sets to match a manually derived example.
    """
    if isinstance(causal_graph, str):
        from grapl.dsl import GraplDSL
        graph = GraplDSL().readgrapl(causal_graph)
    else:
        graph = copy.deepcopy(causal_graph)
    outcomes, interventions, selection = map(names, (outcomes, interventions, selection))
    nodes = graph.nodes()
    if not outcomes or set(outcomes) & set(interventions):
        raise ValueError("Outcomes must be nonempty and disjoint from interventions.")
    if set(outcomes + interventions) & set(selection):
        raise ValueError("Selection nodes cannot be outcomes or interventions.")
    if set(outcomes + interventions + selection) - nodes:
        raise ValueError("Query contains nodes absent from the graph.")
    for node in selection:
        if graph.pa({node}) or graph.bi({node}):
            raise ValueError("Selection nodes must have no incoming or bidirected edges.")
    # Preserve warnings, but translate the legacy failure print into an error.
    with contextlib.redirect_stdout(io.StringIO()):
        equation = sid(set(outcomes), set(interventions), set(selection), graph, mode=mode)
    return from_sid(equation, mode=mode, graph_nodes=nodes)
