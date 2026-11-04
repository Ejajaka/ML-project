"""Rule discovery and pruning for CISCaRL v1.1.

A rule is a conjunction of threshold conditions on the standardised
covariates, represented as a list of ``(feature_index, '<=' | '>', threshold)``.
"""
from __future__ import annotations

import numpy as np


def eval_rule(cond, X):
    m = np.ones(len(X), bool)
    for f, op, thr in cond:
        m &= (X[:, f] <= thr) if op == "<=" else (X[:, f] > thr)
    return m


def simplify(cond):
    """Collapse repeated conditions on the same feature to the binding one.

    A root-to-leaf path like ``cd40 <= 1 & cd40 <= 0 & cd40 <= -1`` is the same
    subset as ``cd40 <= -1``. Keeping the redundant conditions makes the output
    unreadable and inflates the apparent rule complexity.

    NOTE (fixed 2026-11-02): the binding bound for a conjunction of ``<=``
    conditions is the *minimum* threshold, and for ``>`` conditions it is the
    *maximum*. The earlier implementation used max/min respectively, which
    returned a superset whenever a path split twice on the same feature.
    """
    out, le, gt = [], {}, {}
    for f, op, thr in cond:
        if op == "<=":
            le[f] = thr if f not in le else min(le[f], thr)
        else:
            gt[f] = thr if f not in gt else max(gt[f], thr)
    for f in sorted(set(le) | set(gt)):
        if f in le:
            out.append((f, "<=", le[f]))
        if f in gt:
            out.append((f, ">", gt[f]))
    return out


def rule_key(cond):
    return " & ".join(f"{f}{op}{thr:.6f}" for f, op, thr in cond)


def rule_str(cond, names):
    return " & ".join(f"{names[f]} {op} {thr:.3f}" for f, op, thr in cond)


def extract_leaf_rules(tree, X, min_support=20):
    """All root-to-leaf paths of a fitted sklearn tree, as (cond, mask)."""
    out = []
    t = tree.tree_

    def rec(node, cond, mask):
        if t.feature[node] == -2:                   # leaf
            if cond and mask.sum() >= min_support:
                out.append((simplify(list(cond)), mask.copy()))
            return
        f, thr = t.feature[node], t.threshold[node]
        rec(t.children_left[node], cond + [(f, "<=", float(thr))],
            mask & (X[:, f] <= thr))
        rec(t.children_right[node], cond + [(f, ">", float(thr))],
            mask & (X[:, f] > thr))

    rec(0, [], np.ones(len(X), bool))
    return out


def jaccard(m1, m2):
    inter = np.sum(m1 & m2)
    union = np.sum(m1 | m2)
    return float(inter / union) if union else 0.0


def rules_from_ensemble(model, X, min_support=20, max_trees=100):
    """Leaf rules pooled across the trees of a fitted ensemble."""
    out = []
    for est in getattr(model, "estimators_", [])[:max_trees]:
        t = est
        if not hasattr(t, "tree_") and hasattr(t, "__len__"):
            t = t[0]
        if hasattr(t, "tree_"):
            out.extend(extract_leaf_rules(t, X, min_support))
    return out


def dedup(rules):
    seen, out = set(), []
    for cond, mask in rules:
        k = rule_key(cond)
        if k not in seen:
            seen.add(k)
            out.append((cond, mask))
    return out


def prune(rules, X, max_rules=5, overlap=0.8):
    """Greedy: keep the most specific non-redundant rules, drop near-duplicates.

    `rules` is a list of dicts with keys cond, mask, score (higher = better).
    A rule is dropped if it overlaps > `overlap` with an already kept rule.
    """
    kept = []
    for r in sorted(rules, key=lambda d: -d["score"]):
        if any(jaccard(r["mask"], k["mask"]) > overlap for k in kept):
            continue
        kept.append(r)
        if len(kept) >= max_rules:
            break
    return kept


def stability_scores(rules, subsample_rules, X, thr=0.8):
    """Fraction of subsamples containing a rule with Jaccard > thr."""
    scores = np.zeros(len(rules))
    for i, (_, mask) in enumerate(rules):
        hits = sum(1 for sub in subsample_rules
                   if any(jaccard(mask, m) > thr for _, m in sub))
        scores[i] = hits / max(len(subsample_rules), 1)
    return scores
