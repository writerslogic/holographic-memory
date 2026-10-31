#!/bin/zsh
set -eu
setopt pipe_fail
script_dir=${0:A:h}
exec uv run --no-project --offline python "$script_dir/rename_warm.py" "$@"
