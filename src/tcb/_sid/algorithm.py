from . import expressions as exprext
from .._graph import is_m_separated
import copy as cp
import warnings
import grapl.eqn as eqn
import grapl.expr as expr

def remove_outgoing(G, node):
    """Remove outgoing directed edges from the node, retaining bidirected edges.
       This modifies the graph in-place.
    """
    # Remove this node from its children's parents
    for child in G.ch({node}):
        G.vars[child].parents = G.vars[child].parents.difference({node})
    G.vars[node].children = set()
    # for bidirect in G.bi({node}):
    #     G.vars[bidirect].bidirects = G.vars[bidirect].bidirects.difference({node})
    # G.vars[node].bidirects = set()
    return G

def remove_incoming(G, node):
    for parent in G.pa({node}):
        G.vars[parent].children = G.vars[parent].children.difference({node})
    G.vars[node].parents = set()
    for bidirect in G.bi({node}):
        G.vars[bidirect].bidirects = G.vars[bidirect].bidirects.difference({node})
    G.vars[node].bidirects = set()
    return G

def ismsep_set(G, source: set, target: set, cond = set()):
    
    """Check if the source set is m-separated from the target set given the condition set in the graph G.
       Returns True if m-separated, False otherwise.
    """
    for s in source:
        for t in target:
            if not is_m_separated(G, s, t, cond):
                return False

    return True

def isdsep_set(G, source: set, target: set, cond = set()):
    """Check if the source set is d-separated from the target set given the condition set in the graph G.
       Returns True if d-separated, False otherwise.
    """
    for s in source:
        for t in target:
            if not G.isdsep(s, t, cond):
                return False
    return True

def sid(Y, X, S, D, mode="standard"):
    """Return a transport formula (Eqn), or None when not transportable.

    Y, X, S are sets of outcome, intervention and selection nodes; D is
    the selection diagram. The input graph is not modified.

    mode="standard" preserves the original sID order: first decompose
    using target-domain observations, then try direct transport when
    the current graph has a single district.
    mode="more_do" tries direct transport first in every single-component
    subproblem, preferentially retaining source-domain do-expressions.
    Recursive calls use the same mode. The original simplification timing
    of each version is preserved.
    """
    if mode not in ("standard", "more_do"):
        raise ValueError("mode must be 'standard' or 'more_do'")

    Z = set()
    P_star = exprext.LocalDOExpr()  
    P_star.addvars(num = D.nodes().difference(S))
    D_original = cp.deepcopy(D)
    
    def sid_recur(Y, X, Z, S, P_star, D, D_original):
        
        V = D.nodes()
        if S.issubset(V):
            V = V.difference(S)
        G = cp.deepcopy(D)
        G = G.sub(V)
        
        # Line 1
        if len(X) == 0:
            VdY = V.difference(Y)
            P_star_copy = cp.deepcopy(P_star)
            P_star_copy.addvars(mrg = VdY.union(P_star_copy.mrg))
            P_star_copy.simplify()
            return P_star_copy, True

        Y_an_G = G.an(Y)
        VdY_an_G = V.difference(Y_an_G)
        #Line 2
        if len(VdY_an_G) > 0:
            P_star_copy = cp.deepcopy(P_star)
            P_star_copy.addvars(mrg = VdY_an_G.union(P_star_copy.mrg))
            G_sub_Y_an_G = cp.deepcopy(G)
            G_sub_Y_an_G = G_sub_Y_an_G.sub(Y_an_G)
            return sid_recur(Y, X.intersection(Y_an_G), Z, S, P_star_copy, G_sub_Y_an_G, D_original)
        
        VdX = V.difference(X)
        G_doX = cp.deepcopy(G)
        for x in X:
            G_doX = remove_incoming(G_doX, x)
        Y_an_G_doX = G_doX.an(Y)
        W = VdX.difference(Y_an_G_doX)
        # Line 3
        if len(W) > 0:
            return sid_recur(Y, X.union(W), Z, S, P_star, G, D_original)
        
        G_sub_dX = cp.deepcopy(G)
        G_sub_dX = G_sub_dX.sub(VdX)
        C = G_sub_dX.districts()
            
        # Line 4
        if len(C) > 1:
            I_star_exprs = []
            for c in C:
                P_star_copy = cp.deepcopy(P_star)
                I_star_expr_c, c_sidfixable = sid_recur(c, V.difference(c), Z, S, P_star_copy, G, D_original)
                if c_sidfixable:
                    I_star_mrg = list(I_star_expr_c.mrg)
                    for mrg_var in I_star_mrg:
                        new_var = mrg_var + chr(39)     # Avoid variable name clashes
                        I_star_expr_c.subsvar(mrg_var, new_var)
                    I_star_expr_c.simplify()
                    I_star_exprs.append(I_star_expr_c)
                else:
                    return None, False
            I_star_expr = exprext.LocalDOExpr()
            I_star_expr.addvars(mrg = V.difference(Y.union(X)))
            I_star_expr.combine(tuple(I_star_exprs))
            if mode == "more_do":
                I_star_expr.simplify()
            return I_star_expr, True

        # Line 5
        if len(C) == 1:
            
            districts = G.districts()
            # more_do tries direct transport before observational decomposition;
            # standard only tries it when decomposition is unavailable.
            if mode == "more_do" or len(districts) == 1:
                D_doX = cp.deepcopy(D_original)
                for x in X:
                    D_doX = remove_incoming(D_doX, x)

                S_ind_Y_on_X = ismsep_set(D_doX, Y, S, X)
                S_ind_Y_on_XandZ = ismsep_set(D_doX, Y, S, X.union(Z))
                # Line 10
                if S_ind_Y_on_X:
                    py_dox = exprext.LocalDOExpr()
                    py_dox.addvars(num = Y, dov = X)
                    return py_dox, True
                elif S_ind_Y_on_XandZ:
                    py_dox = exprext.LocalDOExpr()
                    py_dox.addvars(num = Y.union(Z), den = Z, dov = X)
                    return py_dox, True

            # Line 6
            if len(districts) <= 1:
                return None, False

            # line 7
            if C[0] in districts:
                mrg_vars = C[0].difference(Y)
                expr_c0 = exprext.LocalDOExpr() 
                expr_c0.addvars(mrg = mrg_vars)
                topo_ordering = G.topsort()
                for v_i in C[0]:
                    vi_topo_index = topo_ordering.index(v_i)
                    vi_topo_procd = set(topo_ordering[: vi_topo_index])
                    den_term = vi_topo_procd
                    num_term = {v_i}.union(den_term)
                    expr_c0.addvars(num = num_term, den = den_term)
                    expr_c0.simplify()
                expr_c = exprext.LocalDOExpr()
                expr_c.addvars(mrg = mrg_vars)
                for num in expr_c0.num:
                    mrg_vars_num = V.difference(num)
                    P_star_copy = cp.deepcopy(P_star)
                    P_star_copy.addvars(mrg = mrg_vars_num)
                    P_star_copy.simplify()
                    expr_c.combine((P_star_copy,))
                for den in expr_c0.den:
                    mrg_vars_den = V.difference(den)
                    P_star_copy = cp.deepcopy(P_star)
                    P_star_copy.addvars(mrg = mrg_vars_den)
                    P_star_copy.simplify()
                    P_star_copy_inv = exprext.LocalDOExpr(num=P_star_copy.den, den=P_star_copy.num, mrg=P_star_copy.mrg)
                    P_star_copy_inv.simplify()
                    expr_c.combine((P_star_copy_inv,))
                expr_c.simplify()
                return expr_c, True
            
            c0_in_c_flag = False
            for c in districts:
                if C[0].issubset(c):
                    c0_in_c_flag = True
                    c_prime = c
                    break
            # Line 8
            if c0_in_c_flag:
                c_expr = exprext.LocalDOExpr()
                topo_ordering = G.topsort()
                for v_i in c_prime:
                    vi_topo_index = topo_ordering.index(v_i)
                    vi_topo_procd = set(topo_ordering[: vi_topo_index])
                    vi_topo_procd_and_c_prime = vi_topo_procd.intersection(c_prime)
                    vidc_prime = vi_topo_procd.difference(c_prime)
                    den_term = vi_topo_procd_and_c_prime.union(vidc_prime)
                    num_term = {v_i}.union(den_term)
                    c_expr.addvars(num = num_term, den = den_term)
                # Paper: Z = C'/X, but the implementation seems shows Z = X/C'
                G_sub_c_prime = cp.deepcopy(G)
                G_sub_c_prime = G_sub_c_prime.sub(c_prime)
                return sid_recur(Y, X.intersection(c_prime), X.difference(c_prime), S, c_expr, G_sub_c_prime, D_original)  
    
        warnings.warn(
            "sID reached an unhandled recursive state; returning (None, False). "
            "Check the input graph and query. "
            f"mode={mode!r}, Y={Y!r}, X={X!r}, Z={Z!r}, S={S!r}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None, False

    rhs, transportable = sid_recur(Y, X, Z, S, P_star, D, D_original)
    if not transportable:
        print("Not transportable")
        return None
    if mode == "standard":
        rhs.simplify()
    lhs = expr.Expr()
    lhs.addvars(num=Y, dov=X)
    transport_formula = eqn.Eqn(lhs, rhs)
    return transport_formula
