#!/bin/bash
# _vsds_transforms.sh
#
# Shared helper for applying index-alias transforms to (INDEX, PARAMS, EXTRA_TAG)
# triples. Sourced by both run_vsds.sh (inside run_benchmark) and quick_run.sh
# (before the maxbench command is built), so CagraCH / IVFH aliases +
# auto use_cuvs + auto index_data_on_gpu behavior is defined in one place.
#
# Contract:
#   apply_index_transforms RAW_INDEX RAW_PARAMS EXISTING_TAG
#
# Sets globals:
#   ACTUAL_INDEX        — faiss-acceptable index string (IVFH→IVF, CagraCH→Cagra, else passthrough)
#   TRANSFORMED_PARAMS  — PARAMS with alias auto-params merged in
#   TRANSFORM_TAG       — suffix to append to EXTRA_TAG ("ivf_h", "cagra_ch", or "")
#
# Semantics for merging:
#   - Alias (CagraCH, IVFH): STRICT preset. Force-sets the preset's key=values
#     even if user pre-set a conflicting value. The alias implies a recipe; if
#     you want to override the recipe, use the raw index name (Cagra, IVF…).
#   - Plain name (Cagra, IVF…): missing-only fill. If user already set a key,
#     their value wins.
#   - use_cuvs: always force-set (Cagra→1, IVF→0). cuVS forces interleaved layout
#     for IVF which is incompatible with how Maximus uses it; Cagra requires
#     cuVS. Mis-setting use_cuvs either crashes or silently changes layout.

# Bash 4.3+ required for `local -n`. cscs-gh200, dgx-spark-02, sgs-gpu06 all have it.

# Set a key=value in a params string (comma-separated), replacing any existing
# value for that key. If the key is absent, appends.
params_force_set() {
    local var_name="$1"
    local kv="$2"
    local key="${kv%%=*}"
    local -n params_ref="$var_name"
    if [[ ",$params_ref," == *",${key}="* ]]; then
        # Replace existing occurrence (preserves leading-comma or start-of-string)
        params_ref=$(printf '%s' "$params_ref" | sed -E "s/(^|,)${key}=[^,]*/\1${kv}/")
    else
        params_ref="${params_ref},${kv}"
    fi
}

# Append key=value to params only if the key isn't already set. User's value wins.
params_add_if_missing() {
    local var_name="$1"
    local kv="$2"
    local key="${kv%%=*}"
    local -n params_ref="$var_name"
    if [[ ",$params_ref," != *",${key}="* ]]; then
        params_ref="${params_ref},${kv}"
    fi
}

apply_index_transforms() {
    local raw_index="$1"
    local raw_params="$2"
    # $3 = existing EXTRA_TAG — consumed by caller for filename composition

    ACTUAL_INDEX="$raw_index"
    TRANSFORMED_PARAMS="$raw_params"
    TRANSFORM_TAG=""

    # --- CagraCH (graph cache + host view) ---
    # Strict preset: always host view, always cache graph.
    if [[ "$raw_index" == *"CagraCH"* ]]; then
        ACTUAL_INDEX="${raw_index/CagraCH/Cagra}"
        params_force_set TRANSFORMED_PARAMS "cagra_cache_graph=1"
        params_force_set TRANSFORMED_PARAMS "index_data_on_gpu=0"
        TRANSFORM_TAG="cagra_ch"
    elif [[ "$raw_index" == *"CagraC"* ]]; then
        # CagraC: graph cache + data on GPU (mirrors CagraCH but GPU-resident data).
        ACTUAL_INDEX="${raw_index/CagraC/Cagra}"
        params_force_set TRANSFORMED_PARAMS "cagra_cache_graph=1"
        params_force_set TRANSFORMED_PARAMS "index_data_on_gpu=1"
        TRANSFORM_TAG="cagra_c"
    elif [[ "$raw_index" == *"Cagra"* ]]; then
        # Plain Cagra: default is data on GPU. User may override.
        params_add_if_missing TRANSFORMED_PARAMS "index_data_on_gpu=1"
    fi

    # --- IVFH (inverted lists on host, referenced via ATS) ---
    # Strict preset: always host view.
    if [[ "$raw_index" == *"IVFH"* ]]; then
        ACTUAL_INDEX="${ACTUAL_INDEX//IVFH/IVF}"
        params_force_set TRANSFORMED_PARAMS "index_data_on_gpu=0"
        TRANSFORM_TAG="ivf_h"
    elif [[ "$raw_index" == *"IVF"* ]] && [[ "$raw_index" != *"IVFH"* ]]; then
        # Plain IVF: default is data on GPU (full H2D copy). User may override.
        params_add_if_missing TRANSFORMED_PARAMS "index_data_on_gpu=1"
    fi

    # --- use_cuvs auto-correct (always force; user mis-set would crash or corrupt) ---
    if [[ "$raw_index" == *"Cagra"* ]]; then
        params_force_set TRANSFORMED_PARAMS "use_cuvs=1"
    elif [[ "$raw_index" == *"IVF"* ]]; then
        params_force_set TRANSFORMED_PARAMS "use_cuvs=0"
    fi
}

# Join an existing EXTRA_TAG with a TRANSFORM_TAG, handling empties correctly.
#   existing ""           + transform ""         -> ""
#   existing ""           + transform "ivf_h"    -> "ivf_h"
#   existing "k_100"      + transform ""         -> "k_100"
#   existing "k_100"      + transform "ivf_h"    -> "k_100-ivf_h"
join_extra_tag() {
    local existing="$1"
    local transform="$2"
    if [ -z "$existing" ]; then
        printf '%s' "$transform"
    elif [ -z "$transform" ]; then
        printf '%s' "$existing"
    else
        printf '%s-%s' "$existing" "$transform"
    fi
}
