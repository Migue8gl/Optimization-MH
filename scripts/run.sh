#!/bin/bash

#!/bin/bash

# Function to show help information
show_help() {
    echo "Usage: $0 [-p] [-t] [-h] [-s SEED] [-d DATASET]"
    echo "  -p           Parallel execution (optional)"
    echo "  -s SEED      Seed for random number generation (optional)"
    echo "  -d DATASET   Dataset to process (1, 2, 3) (optional)"
    echo "  -t TEST      Activate test to dump debug info into txt files (optional)"
    echo "  -h HELP      Show this help message and exit"
    exit 1
}

# Variables with default values
Parallel=false
Seed=""
Dataset=""
Test=false

# Define the directory for results
ResultsDir="./files/results"

# Process command line arguments
while getopts "ps:d:t:h" opt; do
    case $opt in
        p)
            Parallel=true
            ;;
        s)
            Seed="$OPTARG"
            ;;
        d)
            Dataset="$OPTARG"
            ;;
        t)
            Test=true
            ;;
        h)
            show_help
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            show_help
            ;;
    esac
done

# Create results directory if it doesn't exist
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
        
        # Build the command with options based on provided values
        cmd="./bin/ga"

        if [ -n "$Seed" ]; then
            cmd="$cmd -s $Seed"
        fi

        if [ -n "$Dataset" ]; then
            cmd="$cmd -d $Dataset"
        fi

        if [ "$Test" == true ]; then
            cmd="$cmd -t"
        fi

        # Execute the command
        $cmd > "$ResultsDir/${dataset_name}_results.txt"
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

