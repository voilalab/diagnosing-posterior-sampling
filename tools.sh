#!/usr/bin/env bash
# tools.sh — run all linters on dirty (or all) Python files

excluded=(.venv)

with_docs=false
all_files=false
for arg in "$@"; do
    [[ "$arg" == "--with-docs" ]] && with_docs=true
    [[ "$arg" == "--all" ]]     && all_files=true
done

if [ "$all_files" = true ]; then
    prune_args=()
    for subtree in "${excluded[@]}"; do
        prune_args+=(-path "./${subtree%/}" -prune -o)
    done
    files=$(find . "${prune_args[@]}" -type f -name "*.py" -print)
else
    files=$(git diff --name-only --diff-filter=ACMR HEAD | grep '\.py$')
fi

[ -z "$files" ] && echo "No Python files found." && exit 0

echo "$files" | xargs uv run ruff check
echo "$files" | xargs uv run ruff check --fix
echo "$files" | xargs uv run ty check

if [ "$with_docs" = true ]; then
    echo "$files" | xargs uv run pydoclint --style="google"
    echo "$files" | xargs uv run pydocstyle --convention="google"
fi