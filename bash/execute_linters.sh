#!/bin/bash

path="$1"
echo "Run on this path: $path" 
read -p "Do you want to run linters in dynamic mode?: y/n " type_run
if [[ "$type_run" == "n" ]]; then
    ruff $path
    isort $path
    black $path
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

    # Isort section

    read -p "Do you want to run isort?: y/n " run_isort
    if [[ "$run_isort" == "y" ]]; then 
        isort $path
    fi

    # Black section

    read -p "Do you want to run black?: y/n " run_black
    if [[ "$run_isort" == "y" ]]; then 
        black $path
    fi

    # Mypy section

    #cd .. || exit
    read -p "Do you want to run mypy?: y/n " run_mypy
    if [[ "$run_mypy" == "y" ]]; then 
        mypy $path --pretty
    fi
fi