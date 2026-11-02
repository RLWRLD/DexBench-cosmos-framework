# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Sourced by the DexBench launchers (or by hand: `source examples/_dexbench_env.sh`): exports the settings in examples/dexbench.env (or $DEXBENCH_ENV_FILE), the
# machine's and the user's (see examples/dexbench.env.example). A variable already set in the shell wins, and an empty
# value is skipped, so an unfilled line leaves the tool's own default (an empty HF_HOME would not). Values are never
# printed.
_dexbench_env_file="${DEXBENCH_ENV_FILE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/dexbench.env}"
if [[ -f "$_dexbench_env_file" ]]; then
    while IFS= read -r _line || [[ -n "$_line" ]]; do
        _line="${_line%%#*}"
        [[ "$_line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=[[:space:]]*(.*[^[:space:]])?[[:space:]]*$ ]] || continue
        _key="${BASH_REMATCH[1]}" _value="${BASH_REMATCH[2]:-}"
        [[ -n "$_value" && -z "${!_key+set}" ]] && export "$_key=$_value"
    done < "$_dexbench_env_file"
elif [[ -n "${DEXBENCH_ENV_FILE:-}" ]]; then
    echo "ERROR: DEXBENCH_ENV_FILE not found: $DEXBENCH_ENV_FILE" >&2
    unset _dexbench_env_file
    return 1  # sourced: a shell that sources this by hand stays open; the launchers stop on the failure
fi
unset _dexbench_env_file _line _key _value
