"""
Per-operator time attribution for PostgreSQL EXPLAIN (ANALYZE, ...) JSON plans.

One recursive tree walk. The invariant is:
    sum(operators.values()) == node_wall_time(plan_root)

Rules (confirmed against PG docs + FlameExplain):

    1. Wall-clock for a node = Actual Total Time * Actual Loops.
    2. Under a Gather / Gather Merge, descendants use the Gather's own
       Actual Loops as the multiplier instead of their own (PG reports
       per-worker bookkeeping otherwise).
    3. Hashed SubPlan roots fake loops > 1 as internal bookkeeping; treat
       them as loops = 1.
    4. InitPlan and SubPlan subtrees are NOT walked for accounting —
       their time is already folded into the referring node's own att.
       They ARE walked to detect vector-search work inside them; any VS
       time found is moved from the referring node's exclusive bucket
       into VectorSearch (preserves the total).
    5. Parent att is inclusive of children. self_time = wall - sum(children).
       No clamping needed when rules 1-4 are correct.

Verify correctness: `python verify_plan_parser.py --rep 5`
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

VEC_OPS = ("<->", "<=>", "<#>")

# Reset VectorSearch propagation here — these do real relational work.
PROPAGATION_STOPPERS = (
    "Hash Join", "Nested Loop", "Merge Join",
    "Aggregate", "HashAggregate", "GroupAggregate", "WindowAgg",
    "Unique", "Append", "MergeAppend",
)

# Tables whose bare Seq Scan inside a vec query counts as VectorSearch
# (ENN fallback — distance is computed during the scan).
VEC_RELATIONS = frozenset({"reviews", "images"})

SUBPLAN_REL = frozenset({"InitPlan", "SubPlan"})


# ---------------------------------------------------------------------------
# Classification — one node -> one bucket
# ---------------------------------------------------------------------------

def classify_node(node: dict) -> str:
    """Single-node category from its own fields (no subtree context)."""
    raw = node.get("Node Type", "")
    node_type = raw[len("Parallel "):] if raw.startswith("Parallel ") else raw

    sort_key = " ".join(node.get("Sort Key") or [])
    index_cond = node.get("Index Cond") or ""
    order_by = node.get("Order By")
    order_by = " ".join(order_by) if isinstance(order_by, list) else (order_by or "")

    if any(op in index_cond for op in VEC_OPS): return "VectorSearch"
    if any(op in order_by for op in VEC_OPS):   return "VectorSearch"
    if "Sort" in node_type and any(op in sort_key for op in VEC_OPS):
        return "VectorSearch"
    if node_type in ("Hash Join", "Nested Loop", "Merge Join"):         return "Join"
    if node_type in ("Aggregate", "HashAggregate", "GroupAggregate", "WindowAgg"):
        return "GroupBy"
    if "Sort" in node_type:                                             return "OrderBy"
    if node_type == "Limit":                                            return "Limit"
    if node_type == "Unique":                                           return "Distinct"
    if node_type in ("Seq Scan", "Index Scan", "Index Only Scan",
                     "Bitmap Index Scan", "Bitmap Heap Scan", "Tid Scan"):
        return "Filter"
    # CTE Scan reads the materialized result of a WITH clause. PG attributes
    # the InitPlan's materialization time (including any aggregation/filter
    # work inside the CTE) to the first CTE Scan that triggers it, so the
    # CTE Scan node is where the CTE's relational work visually lands. We
    # give it its own bucket so callers can choose where to display it
    # (e.g. plot 1 rolls it into Rel. Operators; plot 2 folds it into Other).
    if node_type == "CTE Scan":                                         return "CTE Scan"
    # Hash, Materialize, Memoize, Gather, Subquery Scan, Result, ...
    return "Other"


# ---------------------------------------------------------------------------
# Node wall-time with the four PG-accounting quirks
# ---------------------------------------------------------------------------

def node_wall_time(node: dict, gather_loops: Optional[float] = None) -> float:
    avg_ms = float(node.get("Actual Total Time", 0.0) or 0.0)
    if gather_loops is not None:
        # Rule 2: descendants of a Gather use the Gather's own loop count.
        return avg_ms * gather_loops
    if node.get("Subplan Name", "").lower().startswith("hashed"):
        # Rule 3: hashed SubPlan inner plan ran once; loops field is bookkeeping.
        return avg_ms * 1.0
    return avg_ms * float(node.get("Actual Loops", 1) or 1)  # Rule 1


# ---------------------------------------------------------------------------
# The only recursive walker
# ---------------------------------------------------------------------------

def _find_subplan_attribution(node: dict, gather_loops: Optional[float],
                              force_cat: Optional[str], in_vec: bool,
                              accum: dict) -> float:
    """
    Walk a subplan subtree (InitPlan/SubPlan) and accumulate per-category
    exclusive time into `accum`. Returns the node's wall so the caller can
    compute sibling sums. Does NOT contribute to main-tree accounting directly;
    the caller decides how much of this dict to redistribute.

    Classification mirrors `_walk` so VS propagation and vec-relation rules
    stay consistent between main and subplan traversals.
    """
    raw = node.get("Node Type", "")
    clean = raw[len("Parallel "):] if raw.startswith("Parallel ") else raw

    gather_here = float(node.get("Actual Loops", 1) or 1) if raw.startswith("Gather") else None
    child_gl = gather_here if gather_here is not None else gather_loops

    sort_key = " ".join(node.get("Sort Key") or [])
    index_cond = node.get("Index Cond") or ""
    if clean in PROPAGATION_STOPPERS:
        force_cat = None
        output_str = " ".join(node.get("Output") or [])
        if not any(op in output_str for op in VEC_OPS):
            in_vec = False
    if ("Sort" in clean and any(op in sort_key for op in VEC_OPS)) or \
       any(op in index_cond for op in VEC_OPS):
        in_vec = True

    if force_cat is not None:
        cat = force_cat
    elif in_vec and clean in ("Seq Scan", "Bitmap Heap Scan", "Index Only Scan") \
            and node.get("Relation Name", "") in VEC_RELATIONS:
        cat = "VectorSearch"
    else:
        cat = classify_node(node)

    next_force = "VectorSearch" if cat == "VectorSearch" and force_cat is None else force_cat

    wall = node_wall_time(node, gather_loops)

    # Same parent-cap / parallel-overlap handling as _walk (Rule 7): walk each
    # non-subplan child into a local accum; if children_wall exceeds the
    # parent's reported wall (parallel subtree), scale each child's
    # attribution by (wall / children_wall) so the subtree fits exactly.
    child_locals: list = []
    children_wall = 0.0
    for c in node.get("Plans", ()) or ():
        if c.get("Parent Relationship") in SUBPLAN_REL:
            _find_subplan_attribution(c, child_gl, None, False, accum)
        else:
            local: dict = defaultdict(float)
            w = _find_subplan_attribution(c, child_gl, next_force, in_vec, local)
            children_wall += w
            child_locals.append((w, local))

    if children_wall > wall and wall > 0.0:
        scale = wall / children_wall
        for _, local in child_locals:
            for k, v in local.items():
                accum[k] += v * scale
        children_wall = wall
    else:
        for _, local in child_locals:
            for k, v in local.items():
                accum[k] += v

    self_time = wall - children_wall  # always >= 0 here
    accum[cat] += self_time
    return wall


def _walk(node: dict, accum: dict, gather_loops: Optional[float],
          force_cat: Optional[str], in_vec: bool) -> float:
    """Main accounting walk. Returns the node's wall time."""
    raw = node.get("Node Type", "")
    clean = raw[len("Parallel "):] if raw.startswith("Parallel ") else raw

    gather_here = float(node.get("Actual Loops", 1) or 1) if raw.startswith("Gather") else None
    child_gl = gather_here if gather_here is not None else gather_loops

    sort_key = " ".join(node.get("Sort Key") or [])
    index_cond = node.get("Index Cond") or ""
    if clean in PROPAGATION_STOPPERS:
        force_cat = None
        # Only carry in_vec past a join/agg boundary if THIS node still
        # projects the vec expression — otherwise scans below are unrelated
        # query enumeration, not candidate generation for the vec sort.
        output_str = " ".join(node.get("Output") or [])
        if not any(op in output_str for op in VEC_OPS):
            in_vec = False
    if ("Sort" in clean and any(op in sort_key for op in VEC_OPS)) or \
       any(op in index_cond for op in VEC_OPS):
        in_vec = True

    if force_cat is not None:
        cat = force_cat
    elif in_vec and clean in ("Seq Scan", "Bitmap Heap Scan", "Index Only Scan") \
            and node.get("Relation Name", "") in VEC_RELATIONS:
        cat = "VectorSearch"
    else:
        cat = classify_node(node)

    next_force = "VectorSearch" if cat == "VectorSearch" and force_cat is None else force_cat

    wall = node_wall_time(node, gather_loops)

    # Walk children. Main children contribute to accum (scaled if needed) and
    # to children_wall. Subplan children (InitPlan/SubPlan) do not contribute
    # to children_wall — their time is already inside this node's exclusive —
    # but we walk them into subplan_attrib so the bucketized share can be
    # redistributed into the referring node's self_time below.
    #
    # Rule 7 (parent-cap under parallel execution): a parent's reported
    # `Actual Total Time` is inclusive of its children's elapsed wall-clock
    # from the leader's perspective. Under a Gather, each worker's
    # `Actual Total Time` is the per-worker wall (avg across workers) — but
    # that per-worker wall can include worker-side startup/teardown that
    # extends past the leader's Gather window. PG reports the Gather itself
    # using the leader's timeline, so children's walls can legitimately sum
    # to more than the Gather's reported wall. The right fix is to trust the
    # parent: when children_wall > wall, scale each child's attribution by
    # (wall / children_wall) so the operator slices inside the parallel
    # subtree sum to the parent's reported elapsed wall. Nothing to do for
    # non-parallel subtrees where parent is always >= sum(children) by PG
    # semantics; the branch is a no-op there.
    child_locals: list = []  # list of (wall, local_accum) per non-subplan child
    children_wall = 0.0
    subplan_attrib: dict = defaultdict(float)
    for c in node.get("Plans", ()) or ():
        if c.get("Parent Relationship") in SUBPLAN_REL:
            _find_subplan_attribution(c, child_gl, None, False, subplan_attrib)
        else:
            local: dict = defaultdict(float)
            w = _walk(c, local, child_gl, next_force, in_vec)
            children_wall += w
            child_locals.append((w, local))

    if children_wall > wall and wall > 0.0:
        scale = wall / children_wall
        for _, local in child_locals:
            for k, v in local.items():
                accum[k] += v * scale
        children_wall = wall  # subtree now fits exactly; self_time = 0 below
    else:
        for _, local in child_locals:
            for k, v in local.items():
                accum[k] += v

    self_time = wall - children_wall

    # Redistribute categorized subplan time into the referring node's exclusive
    # share, capped at self_time so sum(accum) == plan_root_wall stays true.
    # Skip when the referring node is itself VS (its own classification already
    # covers the subplan work) or when self_time is non-positive.
    #
    # Priority rule (preserves legacy VS behavior): pay VS first up to self_time
    # (VS is the most expensive to misattribute), then proportionally distribute
    # any remaining budget across the other categories found inside the subplan.
    if self_time > 0.0 and cat != "VectorSearch":
        vs_share = subplan_attrib.pop("VectorSearch", 0.0)
        if vs_share > 0.0:
            vs_move = min(vs_share, self_time)
            accum["VectorSearch"] += vs_move
            self_time -= vs_move
        other_total = sum(subplan_attrib.values())
        if other_total > 0.0 and self_time > 0.0:
            move_total = min(other_total, self_time)
            scale = move_total / other_total
            for k, v in subplan_attrib.items():
                accum[k] += v * scale
            self_time -= move_total

    accum[cat] += self_time
    return wall


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def _unwrap(plan_json: Any) -> tuple[dict, Optional[float]]:
    """
    Normalize `EXPLAIN (... FORMAT JSON)` shapes to (root_plan_node, exec_time_ms).

    Accepts the top-level list, the inner dict, or a bare Plan node. Returns
    the Plan node plus Execution Time if present at the top level (else None).
    """
    if isinstance(plan_json, list) and plan_json:
        plan_json = plan_json[0]
    if isinstance(plan_json, dict) and "Plan" in plan_json and "Node Type" not in plan_json:
        exec_ms = plan_json.get("Execution Time")
        exec_ms = float(exec_ms) if isinstance(exec_ms, (int, float)) else None
        return plan_json["Plan"], exec_ms
    return plan_json, None


def parse_plan_with_residual(plan_json: Any) -> tuple[dict[str, float], float]:
    """
    Walk the plan tree and return `(operators, residual_ms)`.

    `residual_ms = Execution Time − sum(operators from walk)` when a top-level
    JSON with `Execution Time` is provided. If residual > 0, it's added to
    `Other` so the returned operators sum to `Execution Time`. This captures
    PG-level wall clock that lives above the plan root — spill I/O,
    trigger time, executor startup/shutdown — in an honest bucket rather
    than silently dropping it.

    When the input is a bare Plan node (no `Execution Time`), residual is 0.
    """
    root, exec_ms = _unwrap(plan_json)
    accum: dict[str, float] = defaultdict(float)
    plan_wall = _walk(root, accum, None, None, False)
    residual = 0.0
    if exec_ms is not None:
        residual = exec_ms - plan_wall
        if residual > 0.0:
            accum["Other"] += residual
    return dict(accum), residual


def parse_plan(plan_json: Any) -> dict[str, float]:
    """
    Parse a full EXPLAIN ANALYZE JSON into {category: ms}.

    When given a top-level JSON, the result sums to `Execution Time` exactly
    (any residual between `plan_root_wall` and Execution Time is added to
    `Other`). When given a bare Plan node, the result sums to `plan_root_wall`.
    """
    ops, _ = parse_plan_with_residual(plan_json)
    return ops


def walk_plan(node: dict, *_args, **_kwargs) -> tuple[dict, float]:
    """Legacy shim: old call sites expected `(accum, wall)`. Uses the residual-padded form."""
    ops, _ = parse_plan_with_residual(node)
    # `wall` used to be the plan root's own wall-clock; preserve that for
    # callers that branch on it, by recomputing from the raw walk.
    raw_root, _ = _unwrap(node)
    return ops, node_wall_time(raw_root, None)


# ---------------------------------------------------------------------------
# Detection: did the planner fall back to brute-force ANN?
# ---------------------------------------------------------------------------

def detect_ann_fallback(plan_json: Any) -> Optional[str]:
    """
    Detect when an ANN-labelled run (HNSW / IVFFlat) has a plan that isn't
    actually using the vector index. Two independent signals:

      A) **Wrong index**: any Index Scan classify_node tags as `VectorSearch`
         whose `Index Name` does not contain `hnsw` or `ivfflat` — the vec
         distance operator slipped into a btree/gist/other index somehow.

      B) **Brute-force pattern on reviews/images**: a vec-classified Sort
         sits above a Seq Scan / Bitmap Heap Scan of `reviews` or `images`
         AND the plan contains zero `hnsw`/`ivfflat` Index Scans. Gated on
         "no vec-aware index anywhere" so self-joins like q11_end (which
         has a small Seq Scan on `query_img` and a real HNSW Index Scan on
         `data_img` in the same plan) don't false-positive.

    The hardcoded `reviews` / `images` check is schema-specific to the VECH
    benchmark — `VEC_RELATIONS` above. Update there if the schema changes.

    Returns a warning string, or None if the plan is clean.
    """
    root, _ = _unwrap(plan_json)

    vec_idx_scan_count = [0]        # Index Scan with hnsw/ivfflat index
    wrong_idx_scan = []             # VS Index Scan with NON-vec index
    brute_force_rels: list[str] = []  # Seq/Bitmap of reviews/images under a vec Sort

    def _walk(n: dict, under_vec_sort: bool) -> None:
        raw = n.get("Node Type", "")
        clean = raw[len("Parallel "):] if raw.startswith("Parallel ") else raw

        if classify_node(n) == "VectorSearch" and "Index Scan" in clean:
            idx_name = (n.get("Index Name") or "").lower()
            if "hnsw" in idx_name or "ivfflat" in idx_name:
                vec_idx_scan_count[0] += 1
            else:
                wrong_idx_scan.append(
                    (idx_name or "(none)", n.get("Relation Name", "?"))
                )

        sort_key = " ".join(n.get("Sort Key") or [])
        is_vec_sort = "Sort" in clean and any(op in sort_key for op in VEC_OPS)
        next_under = under_vec_sort or is_vec_sort

        if under_vec_sort and clean in ("Seq Scan", "Bitmap Heap Scan"):
            rel = n.get("Relation Name", "")
            if rel in VEC_RELATIONS:
                brute_force_rels.append(rel)

        for c in n.get("Plans", ()) or ():
            _walk(c, next_under)

    _walk(root, False)

    if wrong_idx_scan:
        names = ", ".join(sorted(set(f"{i}({r})" for i, r in wrong_idx_scan)))
        return (f"VS-classified Index Scan is NOT using a vector-aware index "
                f"(hnsw/ivfflat). Found: {names}")

    if brute_force_rels and vec_idx_scan_count[0] == 0:
        rels = ", ".join(sorted(set(brute_force_rels)))
        return (f"planner fell back to brute force on {rels} — Sort(<#>) over "
                f"Seq/Bitmap Scan with no hnsw/ivfflat Index Scan anywhere in "
                f"the plan")

    return None
