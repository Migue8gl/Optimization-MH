#!/bin/bash

# Function to show help information
show_help() {
    echo "Usage: $0 [-p] -s SEED [-d DATASET] [-t TRUE/FALSE]"
    echo "  -p           Parallel execution (optional)"
    echo "  -s SEED      Seed for random number generation (required)"
    echo "  -d DATASET   Dataset to process (1, 2, 3) (optional, only if needed)"
    echo "  -t TEST      Activate test to dump debug info into txt files (optional)"
    exit 1
}

# Check if no arguments were provided
if [ $# -eq 0 ]; then
    echo "Error: No arguments provided."
    show_help
fi

# Variables
Parallel=false
Seed=""
Dataset=""
Test=false  # New variable for -t option

# Define the directory for results
ResultsDir="./files/results"

# Process command line arguments
while getopts ":ps:d:t:" opt; do
    case $opt in
        p)
            Parallel=true
            ;;
        s)
            if [[ -n "$OPTARG" ]]; then
                Seed="$OPTARG"
            else
                echo "Error: The -s option requires a seed value."
                show_help
            fi
            ;;
        d)
            if [[ -n "$OPTARG" ]]; then
                Dataset="$OPTARG"
            else
                echo "Error: The -d option requires a dataset value."
                show_help
            fi
            ;;
        t)
            if [[ "$OPTARG" == "TRUE" || "$OPTARG" == "true" || "$OPTARG" == "1" ]]; then
                Test=true
            elif [[ "$OPTARG" == "FALSE" || "$OPTARG" == "false" || "$OPTARG" == "0" ]]; then
                Test=false
            else
                echo "Error: The -t option requires TRUE or FALSE as the argument."
                show_help
            fi
            ;;
    esac
done

# If -help or --help is provided, show help and exit without error
if [[ "$1" == "-help" || "$1" == "-h" ]]; then
    show_help
fi

# Check if Seed is provided (it's always required)
if [[ -z "$Seed" ]]; then
    echo "Error: The -s (Seed) option is required."
    show_help
fi

# Verificar si el directorio de resultados existe, si no, crearlo
if [ ! -d "$ResultsDir" ]; then
    mkdir -p "$ResultsDir"
fi

# Define datasets
Datasets=("heart" "parkinsons" "ionosphere")

# Function to execute the program for a single dataset
run_single_dataset() {
    dataset_index=$((Dataset - 1))
    if [ $dataset_index -ge 0 ] && [ $dataset_index -lt ${#Datasets[@]} ]; then
        dataset_name="${Datasets[$dataset_index]}"
        if [[ "$Test" == true ]]; then
            ./bin/ga "$Seed" "$Dataset" "$Test" > "$ResultsDir/${dataset_name}_results.txt"
        else
            ./bin/ga "$Seed" "$Dataset" > "$ResultsDir/${dataset_name}_results.txt"
        fi
    else
        echo "Error: Invalid dataset index. Available datasets are 1, 2, and 3."
        show_help
    fi
}

# If -p is provided, execute all datasets in parallel
if [[ "$Parallel" == true ]]; then
    for ((i = 1; i <= ${#Datasets[@]}; i++)); do
        Dataset="$i"
        run_single_dataset &
    done
else
    # If a specific dataset is provided, execute only that dataset sequentially
    if [ -n "$Dataset" ]; then
        run_single_dataset
    else
        # Otherwise, execute all datasets sequentially
        for ((i = 1; i <= ${#Datasets[@]}; i++)); do
            Dataset="$i"
            run_single_dataset
        done
    fi
fi

# Wait for all background processes to finish (if running in parallel)
wait

