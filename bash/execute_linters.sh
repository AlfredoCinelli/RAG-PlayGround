#!/bin/bash

path="$1"
echo "Run on this path: $path" 
read -p "Do you want to run linters in dynamic mode?: y/n " type_run
if [[ "$type_run" == "n" ]]; then
    ruff check $path
    ruff format $path
    mypy $path --pretty
else
    # Ruff section

    read -p "Do you want to run ruff?: y/n " run_ruff
    if [[ "$run_ruff" == "y" ]]; then
        read -p "Do you want ruff to auto-fix when fixable errors?: y/n " ruff_fix
        if [[ "$ruff_fix" == "y" ]]; then
            ruff check $path --fix
        else
            ruff check $path
        fi
    fi

    # Black section (via Ruff)

    read -p "Do you want to run black (via Ruff)?: y/n " run_black
    if [[ "$run_black" == "y" ]]; then 
        ruff format $path
    fi

    # Mypy section

    read -p "Do you want to run mypy?: y/n " run_mypy
    if [[ "$run_mypy" == "y" ]]; then 
        mypy $path --pretty
    fi
fi