"""Structured probabilities; no parsing of rendered LaTeX or evaluation of code.

Chapter 4, equations (4.15)--(4.28): remove the unique target outcome
conditional, then divide the remaining integrand by p*(W_X) p*(W_R).
Symbols such as Z and Z' remain distinct until numerical row binding.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping, Tuple

from .exceptions import UnsupportedFormulaError


def names(value):
    if isinstance(value, str):
        value = (value,)
    result = tuple(sorted(value))
    if any(not isinstance(v, str) or not v for v in result):
        raise ValueError("Variable names must be nonempty strings.")
    if len(result) != len(set(result)):
        raise ValueError("Duplicate variable names.")
    return result


@dataclass(frozen=True)
class Probability:
    """A factor or distribution key, e.g. source P(Z | V, do(Y)).

    In a formula, names may be bound symbols (Z'). Distribution registry
    keys use base variable names (Z). Callback inputs are keyed by these
    base names and contain arrays of shape (number_of_points, dimension).
    """
    variables: Tuple[str, ...]
    given: Tuple[str, ...] = ()
    do: Tuple[str, ...] = ()
    domain: str = "target"

    def __post_init__(self):
        for key in ("variables", "given", "do"):
            object.__setattr__(self, key, names(getattr(self, key)))
        if self.domain not in ("target", "source"):
            raise ValueError("domain must be 'target' or 'source'.")
        sets = [set(self.variables), set(self.given), set(self.do)]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
            raise ValueError("Event, condition and intervention variables must be disjoint.")
        if self.domain == "target" and self.do:
            raise ValueError("Target interventional factors must first be identified.")

    @property
    def symbols(self):
        return set(self.variables + self.given + self.do)

    def mapped(self, aliases):
        return Probability(
            tuple(aliases.get(v, v) for v in self.variables),
            tuple(aliases.get(v, v) for v in self.given),
            tuple(aliases.get(v, v) for v in self.do), self.domain,
        )

    def __str__(self):
        prefix = "P*" if self.domain == "target" else "P_source"
        conditions = list(self.given)
        if self.do:
            conditions.append("do(" + ",".join(self.do) + ")")
        suffix = " | " + ",".join(conditions) if conditions else ""
        return prefix + "(" + ",".join(self.variables) + suffix + ")"


def _sort_probability(p):
    return p.domain, p.variables, p.given, p.do


def cancel_factors(numerator, denominator):
    num = Counter(p for p in numerator if p.variables)
    den = Counter(p for p in denominator if p.variables)
    common = num & den
    return (tuple(sorted((num - common).elements(), key=_sort_probability)),
            tuple(sorted((den - common).elements(), key=_sort_probability)))


@dataclass(frozen=True)
class TransportFormula:
    """Product/ratio integrand, with explicit outer marginalization scope.

    Use this to supply a paper-verified formula when an identifier's
    formula differs, or to connect a different sID implementation.
    It is a trusted mathematical input, not a proof of transportability.
    """
    outcomes: Tuple[str, ...]
    interventions: Tuple[str, ...]
    numerator: Tuple[Probability, ...]
    denominator: Tuple[Probability, ...] = ()
    marginal: Tuple[str, ...] = ()
    aliases: Mapping[str, str] = field(default_factory=dict)
    provenance: str = "user supplied"

    def __post_init__(self):
        for key in ("outcomes", "interventions", "marginal"):
            object.__setattr__(self, key, names(getattr(self, key)))
        for key in ("numerator", "denominator"):
            terms = tuple(getattr(self, key))
            if not all(isinstance(p, Probability) for p in terms):
                raise TypeError("Formula factors must be Probability objects.")
            object.__setattr__(self, key, terms)
        object.__setattr__(self, "aliases", dict(self.aliases))
        if not self.outcomes:
            raise ValueError("At least one outcome is required.")
        if set(self.outcomes) & set(self.interventions):
            raise ValueError("Outcomes and interventions must be disjoint.")
        if set(self.marginal) & set(self.outcomes + self.interventions):
            raise ValueError("Marginal symbols must be distinct from free query variables.")
        allowed = set(self.outcomes + self.interventions + self.marginal)
        used = set().union(*(p.symbols for p in self.numerator + self.denominator))
        if used - allowed:
            raise UnsupportedFormulaError(
                "Unbound symbols in transport formula: " + repr(sorted(used - allowed))
                + ". Check the identifier's marginal/do-variable bookkeeping."
            )
        for symbol, base in self.aliases.items():
            if symbol not in self.marginal or not isinstance(base, str) or not base:
                raise ValueError("Aliases must map bound marginal symbols to base variable names.")

    def to_latex(self):
        """Return the transport formula as LaTeX, using sum for marginalization."""
        def probability(p):
            prefix = r"P^{*}" if p.domain == "target" else r"P_{\mathrm{source}}"
            conditions = list(p.given)
            if p.do:
                conditions.append(r"\operatorname{do}(" + ",".join(p.do) + ")")
            condition = r"\mid " + ",".join(conditions) if conditions else ""
            return prefix + r"\left(" + ",".join(p.variables) + condition + r"\right)"

        num = r"\,".join(probability(p) for p in self.numerator) or "1"
        den = r"\,".join(probability(p) for p in self.denominator)
        rhs = r"\frac{" + num + "}{" + den + "}" if den else num
        if self.marginal:
            rhs = r"\sum_{" + ",".join(self.marginal) + r"}\left[" + rhs + r"\right]"
        lhs = r"P^{*}\left(" + ",".join(self.outcomes)
        if self.interventions:
            lhs += r"\mid\operatorname{do}(" + ",".join(self.interventions) + ")"
        return lhs + r"\right) = " + rhs

    def __str__(self):
        return self.to_latex()

    def _repr_latex_(self):
        """Render the formula directly in Jupyter rich displays."""
        return "$$" + self.to_latex() + "$$"


@dataclass(frozen=True)
class WeightExpression:
    formula: TransportFormula
    numerator: Tuple[Probability, ...]
    denominator: Tuple[Probability, ...]
    w_x: Tuple[str, ...]
    w_r: Tuple[str, ...]
    kernel_variables: Tuple[str, ...]

    @property
    def required_distributions(self):
        terms = {p.mapped(self.formula.aliases) for p in self.numerator + self.denominator}
        return tuple(sorted(terms, key=_sort_probability))

    @property
    def required_target_distributions(self):
        return tuple(p for p in self.required_distributions if p.domain == "target")

    @property
    def required_source_distributions(self):
        return tuple(p for p in self.required_distributions if p.domain == "source")

    def tostr(self, coupling="product"):
        if coupling not in ("product", "diagonal"):
            raise ValueError("coupling must be 'product' or 'diagonal'.")
        numerator = " * ".join(map(str, self.numerator)) or "1"
        denominator = " * ".join(map(str, self.denominator)) or "1"
        psi = "(" + numerator + ") / (" + denominator + ")"
        kernel = " * ".join("K_" + v + "(" + v + " - " + v + "_n)" for v in self.kernel_variables)
        if kernel:
            psi = kernel + " * " + psi
        if self.w_r and coupling == "product":
            return "w_n = 1/(N*M) * sum_{j=1..M} [" + psi + "]; W_X at row n, W_R at row j"
        return "w_n = 1/N * [" + psi + "]; " + (
            "W_X and W_R at row n (diagonal approximation)" if self.w_r else "W_X at row n"
        )

    def __str__(self):
        return self.tostr()


def derive_weights(formula):
    """Derive TCW under Chapter 4's unique target outcome-factor condition."""
    effect = set(formula.outcomes)
    candidates = [i for i, p in enumerate(formula.numerator)
                  if p.domain == "target" and effect.issubset(p.variables)]
    if len(candidates) != 1:
        raise UnsupportedFormulaError(
            "TCW requires one target observational factor containing all outcomes. "
            "Try standard sID mode or provide a verified transport formula."
        )
    i = candidates[0]
    anchor = formula.numerator[i]
    remaining = list(formula.numerator[:i] + formula.numerator[i+1:])
    if any(p.symbols & effect for p in tuple(remaining) + formula.denominator):
        raise UnsupportedFormulaError("The factors remaining after extracting the outcome must not depend on it.")
    other_events = set(anchor.variables) - effect
    w_x = set(anchor.given) | other_events
    # P*(X,A|B) = P*(X|A,B) P*(A|B).
    if other_events:
        remaining.append(Probability(tuple(other_events), given=anchor.given))
    w_r = set(formula.marginal) - w_x
    denominator = list(formula.denominator)
    if w_x:
        denominator.append(Probability(tuple(w_x)))
    if w_r:
        denominator.append(Probability(tuple(w_r)))
    # Expand target conditionals so the same joint estimator is reused at
    # observed and intervened values. Keep source conditionals atomic.
    def expand(p, num, den):
        if p.domain == "target" and p.given:
            num.append(Probability(p.variables + p.given))
            den.append(Probability(p.given))
        else:
            num.append(p)
    expanded_num, expanded_den = [], []
    for p in remaining:
        expand(p, expanded_num, expanded_den)
    for p in denominator:
        expand(p, expanded_den, expanded_num)
    num, den = cancel_factors(expanded_num, expanded_den)
    plan = WeightExpression(formula, num, den, names(w_x), names(w_r),
                            names(w_x & set(formula.interventions)))
    # Repeated symbols of one base variable in a joint carrier are not an
    # empirical joint of distinct columns. Do not silently tie them together.
    for carrier in (w_x, w_r):
        bases = [formula.aliases.get(v, v) for v in carrier]
        if len(bases) != len(set(bases)):
            raise UnsupportedFormulaError("A carrier contains two aliases of the same variable; explicit independent carriers are required.")
    _ = plan.required_distributions  # Validate mapped distribution keys now.
    return plan
