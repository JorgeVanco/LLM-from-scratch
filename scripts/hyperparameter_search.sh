#!/bin/bash

# Usage: ./hyperparameter_search.sh [options]

set -e

# Default configuration
CONFIG_FILE="configs/baseline.yaml"
RESULTS_CSV="hyperparameter_search_results.csv"
NUM_GPUS=8
BASE_LOG_DIR="logs/hyperparameter_search"
BASE_CHECKPOINT_DIR="checkpoints/hyperparameter_search"
RESUME_SEARCH=false
SEARCH_TYPE="grid"
PARAMETER_NAME="learning_rate"
PARAMETER_PATH="scheduler.max_learning_rate,optimizer.lr"

# Global array to store PIDs of active background experiments
declare -a ACTIVE_PIDS=()

# PID of the progress monitor, to be killed on exit
progress_pid=""

# Function to clean up background processes and exit gracefully
cleanup_and_exit() {
    echo -e "\nScript interrupted. Cleaning up background processes..."
    # Iterate through active PIDs and attempt to kill them
    for pid in "${ACTIVE_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then # Check if process is still running
            echo "Killing experiment process $pid..."
            kill "$pid" 2>/dev/null || true # Kill and ignore errors if already dead
        fi
    done
    # Kill the progress monitor if it's running
    if [ -n "$progress_pid" ] && kill -0 "$progress_pid" 2>/dev/null; then
        echo "Killing progress monitor $progress_pid..."
        kill "$progress_pid" 2>/dev/null || true
    fi
    # Optionally, remove the lock file if it exists
    rm -f "${RESULTS_CSV}.lock"
    exit 1 # Exit with a non-zero status to indicate interruption
}

# Trap signals to ensure cleanup on script interruption
trap 'cleanup_and_exit' INT TERM

# Function to display help
show_help() {
    cat << EOF
Usage: $0 [options]

Options:
  --config PATH           Path to config file (default: configs/baseline.yaml)
  --output PATH          Output CSV file (default: hyperparameter_search_results.csv)
  --gpus NUM             Number of GPUs to use (default: 8)
  --resume               Resume previous search
  --search-type TYPE     Search type: grid, random, or custom (default: grid)
  --parameter NAME       Parameter name for display (default: learning_rate)
  --parameter-path PATH  Config path(s) to set, comma-separated (default: scheduler.max_learning_rate,optimizer.lr)
  --values VALUES        Space-separated values to search (overrides search-type presets)
  -h, --help             Show this help message

Examples:
  # Learning rate search (default)
  $0 --search-type grid

  # Batch size search
  $0 --parameter batch_size --parameter-path training.batch_size --values "16 32 64 128 256"

  # AdamW beta1 search
  $0 --parameter beta1 --parameter-path optimizer.beta1 --values "0.9 0.95 0.99"

  # AdamW beta2 search
  $0 --parameter beta2 --parameter-path optimizer.beta2 --values "0.999 0.9999"

  # Multiple parameter paths (e.g., setting both scheduler and optimizer LR)
  $0 --parameter learning_rate --parameter-path scheduler.max_learning_rate,optimizer.lr --values "1e-4 2e-4 3e-4"

  # Weight decay search
  $0 --parameter weight_decay --parameter-path optimizer.weight_decay --values "0.0 0.01 0.1"

  # Dropout search
  $0 --parameter dropout --parameter-path model.dropout --values "0.0 0.1 0.2 0.3"

EOF
}

# Parse command line arguments
CUSTOM_VALUES=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --output)
            RESULTS_CSV="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --resume)
            RESUME_SEARCH=true
            shift
            ;;
        --search-type)
            SEARCH_TYPE="$2"
            shift 2
            ;;
        --parameter)
            PARAMETER_NAME="$2"
            shift 2
            ;;
        --parameter-path)
            PARAMETER_PATH="$2"
            shift 2
            ;;
        --values)
            CUSTOM_VALUES="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Hyperparameter value configurations
declare -A SEARCH_CONFIGS

# Learning rate configurations
SEARCH_CONFIGS[learning_rate_grid]="1e-5 4e-5 7e-5 1e-4 4e-4 7e-4 1e-3 4e-3 7e-3 1e-2"
SEARCH_CONFIGS[learning_rate_random]="2.3e-5 7.8e-5 1.2e-4 4.7e-4 8.9e-4 2.1e-3 5.6e-3 1.3e-2"
SEARCH_CONFIGS[learning_rate_custom]="1e-4 2e-4 3e-4 4e-4 5e-4"

# Batch size configurations
SEARCH_CONFIGS[batch_size_grid]="8 16 32 64 128 256"
SEARCH_CONFIGS[batch_size_random]="12 24 48 96 192"
SEARCH_CONFIGS[batch_size_custom]="16 32 64 128"

# AdamW beta1 configurations
SEARCH_CONFIGS[beta1_grid]="0.8 0.85 0.9 0.95 0.99"
SEARCH_CONFIGS[beta1_random]="0.82 0.88 0.92 0.96"
SEARCH_CONFIGS[beta1_custom]="0.9 0.95 0.99"

# AdamW beta2 configurations
SEARCH_CONFIGS[beta2_grid]="0.99 0.995 0.999 0.9999"
SEARCH_CONFIGS[beta2_random]="0.992 0.997 0.9995"
SEARCH_CONFIGS[beta2_custom]="0.999 0.9999"

# Weight decay configurations
SEARCH_CONFIGS[weight_decay_grid]="0.0 0.01 0.05 0.1 0.2"
SEARCH_CONFIGS[weight_decay_random]="0.005 0.03 0.07 0.15"
SEARCH_CONFIGS[weight_decay_custom]="0.0 0.01 0.1"

# Dropout configurations
SEARCH_CONFIGS[dropout_grid]="0.0 0.1 0.2 0.3 0.4 0.5"
SEARCH_CONFIGS[dropout_random]="0.05 0.15 0.25 0.35 0.45"
SEARCH_CONFIGS[dropout_custom]="0.0 0.1 0.2 0.3"

# Get parameter values
if [ -n "$CUSTOM_VALUES" ]; then
    IFS=' ' read -ra PARAMETER_VALUES <<< "$CUSTOM_VALUES"
else
    # Try to find predefined configuration
    config_key="${PARAMETER_NAME}_${SEARCH_TYPE}"
    if [[ -n "${SEARCH_CONFIGS[$config_key]}" ]]; then
        IFS=' ' read -ra PARAMETER_VALUES <<< "${SEARCH_CONFIGS[$config_key]}"
    else
        echo "Error: No predefined configuration found for parameter '$PARAMETER_NAME' with search type '$SEARCH_TYPE'"
        echo "Available configurations:"
        for key in "${!SEARCH_CONFIGS[@]}"; do
            echo "  $key: ${SEARCH_CONFIGS[$key]}"
        done
        echo ""
        echo "Use --values to specify custom values, or choose from available parameter/search-type combinations."
        exit 1
    fi
fi

# Validation
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    exit 1
fi

if [ ${#PARAMETER_VALUES[@]} -eq 0 ]; then
    echo "Error: No parameter values defined"
    exit 1
fi

# Create results directory
mkdir -p "${BASE_LOG_DIR}"
mkdir -p "${BASE_CHECKPOINT_DIR}"

# Initialize or resume CSV file
if [ "$RESUME_SEARCH" = true ] && [ -f "$RESULTS_CSV" ]; then
    echo "Resuming previous search. Existing results:"
    cat "$RESULTS_CSV"
    echo ""
else
    echo "${PARAMETER_NAME},final_loss,best_val_loss,experiment_name,gpu_id,status,start_time,end_time,duration_minutes,epochs,best_epoch" > "${RESULTS_CSV}"
fi

# Function to check if experiment already completed
is_experiment_completed() {
    local param_value=$1

    (
        flock -s 200
        grep -q "^${param_value}," "$RESULTS_CSV" && return 0
    ) 200>>"${RESULTS_CSV}.lock"
    return 1
}

# Function to parse JSON results from train.py output
parse_results() {
    local output="$1"
    
    # Extract JSON from output (looking for RESULTS_JSON: prefix)
    local json_line=$(echo "$output" | grep "RESULTS_JSON:" | head -1)
    
    if [ -n "$json_line" ]; then
        # Remove the RESULTS_JSON: prefix
        local json_data="${json_line#*RESULTS_JSON:}"
        
        python3 -c "
import json
import sys
try:
    data = json.loads('$json_data')
    print(f\"{data.get('final_loss', 'N/A')},{data.get('best_val_loss', 'N/A')},{data.get('status', 'unknown')},{data.get('epochs', 'N/A')},{data.get('best_epoch', 'N/A')}\")
except:
    print('N/A,N/A,failed,N/A,N/A')
"
    else
        echo "N/A,N/A,failed,N/A,N/A"
    fi
}

# Function to build parameter override arguments
build_parameter_overrides() {
    local param_value=$1
    local overrides=""
    
    # Split parameter paths by comma and create override for each
    IFS=',' read -ra PATHS <<< "$PARAMETER_PATH"
    for path in "${PATHS[@]}"; do
        overrides+=" ${path}=${param_value}"
    done
    
    echo "$overrides"
}

run_experiment() {
    local param_value=$1
    local gpu_id=$2
    local exp_name="${PARAMETER_NAME}_${param_value}_gpu_${gpu_id}"
    local log_file="${BASE_LOG_DIR}/${exp_name}.log"

    local start_time=$(date +%s)
    local start_time_str=$(date '+%Y-%m-%d %H:%M:%S')

    echo "[$start_time_str] Starting experiment: ${exp_name} on GPU ${gpu_id}"
    
    # Build parameter overrides
    local param_overrides=$(build_parameter_overrides "$param_value")
    
    # Execute the training command
    # Output is redirected to a log file
    uv run -m src.train \
        --config="${CONFIG_FILE}" \
        ${param_overrides} \
        experiment_name="${exp_name}" \
        logging.log_dir="${BASE_LOG_DIR}" \
        logging.checkpoint_dir="${BASE_CHECKPOINT_DIR}" \
        logging.run_name="${exp_name}" \
        training.device="cuda:${gpu_id}" \
        > "${log_file}" 2>&1

    local exit_code=$?
    local end_time=$(date +%s)
    local end_time_str=$(date '+%Y-%m-%d %H:%M:%S')
    local duration=$((($end_time - $start_time) / 60))

    local output=$(cat "${log_file}")
    local parsed_results=$(parse_results "$output")
    IFS=',' read -r final_loss best_val_loss status epochs best_epoch <<< "$parsed_results"

    if [ $exit_code -ne 0 ]; then
        status="failed"
    fi

    # Use flock for safe concurrent writing to the CSV file
    (
        flock -x 200 # Exclusive lock
        echo "${param_value},${final_loss},${best_val_loss},${exp_name},${gpu_id},${status},${start_time_str},${end_time_str},${duration},${epochs},${best_epoch}" >> "${RESULTS_CSV}"
    ) 200>>"${RESULTS_CSV}.lock" # Use a consistent lock file

    if [ "$status" = "completed" ]; then
        echo "[$end_time_str] ✓ Completed experiment: ${exp_name} on GPU ${gpu_id} (${duration} min, Loss: ${final_loss}, Val: ${best_val_loss})"
    else
        echo "[$end_time_str] ✗ Failed experiment: ${exp_name} on GPU ${gpu_id} (${duration} min)"
    fi
}

# Function to wait for available GPU slot by managing ACTIVE_PIDS
wait_for_gpu() {
    # Loop until the number of active PIDs is less than NUM_GPUS
    while [ ${#ACTIVE_PIDS[@]} -ge $NUM_GPUS ]; do
        wait -n
    
        # Rebuild list of active PIDs
        new_pids=()
        for pid in "${ACTIVE_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
            fi
        done
        ACTIVE_PIDS=("${new_pids[@]}")
    done
}

# Function to display real-time progress
show_progress() {
    local completed=0
    local total=${#PARAMETER_VALUES[@]}
    
    while [ $completed -lt $total ]; do
        if [ -f "$RESULTS_CSV" ]; then
            completed=$(flock -s "${RESULTS_CSV}.lock" -c "tail -n +2 \"${RESULTS_CSV}\" 2>/dev/null | wc -l || echo 0")
        fi
        
        # Show progress with color
        local progress_pct=$((completed * 100 / total))
        echo -ne "\r\033[K[\033[32m$completed\033[0m/$total] Progress: $progress_pct% | "
        
        # Show current best result
        if [ -f "$RESULTS_CSV" ] && [ $completed -gt 0 ]; then
            local best_result
            best_result=$(flock -s "${RESULTS_CSV}.lock" -c "tail -n +2 \"${RESULTS_CSV}\" | grep \"completed\" | sort -t',' -k3 -n | head -1")
            
            if [ -n "$best_result" ]; then
                local best_param=$(echo "$best_result" | cut -d',' -f1)
                local best_val_loss=$(echo "$best_result" | cut -d',' -f3)
                echo -ne "Best: ${PARAMETER_NAME}=$best_param, Val=$best_val_loss"
            fi
        fi
        
        sleep 10
    done
    echo ""
}

# Main execution
echo "========================================"
echo "HYPERPARAMETER SEARCH"
echo "========================================"
echo "Config file: $CONFIG_FILE"
echo "Parameter: $PARAMETER_NAME"
echo "Parameter path(s): $PARAMETER_PATH"
echo "Search type: $SEARCH_TYPE"
echo "Parameter values: ${PARAMETER_VALUES[*]}"
echo "Number of GPUs: $NUM_GPUS"
echo "Results file: $RESULTS_CSV"
echo "Resume search: $RESUME_SEARCH"
echo "========================================"
echo ""

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
    echo ""
else
    echo "Warning: nvidia-smi not found. Cannot check GPU status."
fi

# Launch experiments
gpu_counter=0
experiments_launched=0
experiments_skipped=0

for param_value in "${PARAMETER_VALUES[@]}"; do
    # Skip if already completed and resuming
    if [ "$RESUME_SEARCH" = true ] && is_experiment_completed "$param_value"; then
        echo "Skipping already completed experiment for ${PARAMETER_NAME}: $param_value"
        ((experiments_skipped++))
        continue
    fi
    
    # Wait for available GPU if all are busy
    wait_for_gpu
    
    # Run experiment in background and store PID
    run_experiment "$param_value" "$gpu_counter" &
    ACTIVE_PIDS+=($!) # Add the PID of the last background command to the array
    
    experiments_launched=$((experiments_launched + 1))
    
    # Move to next GPU
    gpu_counter=$(( (gpu_counter + 1) % NUM_GPUS ))
    
    # Small delay to prevent race conditions
    sleep 2
done

echo "Launched $experiments_launched experiments, skipped $experiments_skipped"

# Show progress in background
show_progress &
progress_pid=$! # Store the PID of the progress monitor

# Wait for all background jobs launched by this script to complete
echo "Waiting for all experiments to complete..."
for pid in "${ACTIVE_PIDS[@]}"; do
    wait "$pid" || true # Use || true to prevent script from exiting if a wait fails (e.g., process already reaped)
done

# Kill progress monitor
# Check if progress_pid is still active before trying to kill
if [ -n "$progress_pid" ] && kill -0 "$progress_pid" 2>/dev/null; then
    kill "$progress_pid" 2>/dev/null || true
fi

echo ""
echo "========================================"
echo "HYPERPARAMETER SEARCH COMPLETED!"
echo "========================================"
echo "Results saved to: ${RESULTS_CSV}"

# Display enhanced results summary
echo ""
echo "=== ENHANCED RESULTS SUMMARY ==="
printf "%-15s | %-10s | %-13s | %-8s | %-8s | %-10s | %s\n" "$PARAMETER_NAME" "Final Loss" "Best Val Loss" "Duration" "Epochs" "Best Epoch" "Status"
echo "----------------|------------|---------------|----------|----------|------------|--------"
# Use flock directly on the file for shared lock
flock -s "${RESULTS_CSV}.lock" -c "tail -n +2 \"${RESULTS_CSV}\" | while IFS=',' read -r param_value final_loss best_val_loss exp_name gpu_id status start_time end_time duration epochs best_epoch; do
    printf \"%-15s | %-10s | %-13s | %-8s | %-8s | %-10s | %s\n\" \"\$param_value\" \"\$final_loss\" \"\$best_val_loss\" \"\${duration}m\" \"\$epochs\" \"\$best_epoch\" \"\$status\"
done"

# Enhanced best parameter analysis
echo ""
echo "=== BEST ${PARAMETER_NAME^^} ANALYSIS ==="
# Use flock directly on the file for shared lock
best_result=$(flock -s "${RESULTS_CSV}.lock" -c "tail -n +2 \"${RESULTS_CSV}\" | grep \"completed\" | sort -t',' -k3 -n | head -1")

if [ -n "$best_result" ]; then
    IFS=',' read -r best_param best_final_loss best_val_loss exp_name gpu_id status start_time end_time duration epochs best_epoch <<< "$best_result"
    echo "🏆 Best ${PARAMETER_NAME}: ${best_param}"
    echo "   Validation loss: ${best_val_loss}"
    echo "   Final training loss: ${best_final_loss}"
    echo "   Training duration: ${duration} minutes"
    echo "   Total epochs: ${epochs}"
    echo "   Best epoch: ${best_epoch}"
    echo "   Best model checkpoint: ${BASE_CHECKPOINT_DIR}/${exp_name}/best.pt"
    echo ""
    echo "🚀 To use this configuration:"
    local param_overrides=$(build_parameter_overrides "$best_param")
    echo "   uv run -m src.train --config ${CONFIG_FILE}${param_overrides}"
else
    echo "No successful experiments found"
fi

# Summary statistics
echo ""
echo "=== SUMMARY STATISTICS ==="
total_experiments=$(flock -s "${RESULTS_CSV}.lock" -c "tail -n +2 \"${RESULTS_CSV}\" | wc -l")
completed_experiments=$(flock -s "${RESULTS_CSV}.lock" -c "tail -n +2 \"${RESULTS_CSV}\" | grep -c \"completed\"")
failed_experiments=$(flock -s "${RESULTS_CSV}.lock" -c "tail -n +2 \"${RESULTS_CSV}\" | grep -c \"failed\"")
total_duration=$(flock -s "${RESULTS_CSV}.lock" -c "tail -n +2 \"${RESULTS_CSV}\" | cut -d',' -f9 | awk '{sum+=\$1} END {print sum}'")

echo "Total experiments: $total_experiments"
echo "Completed: $completed_experiments"
echo "Failed: $failed_experiments"
echo "Total training time: ${total_duration} minutes"

if [ "$completed_experiments" -gt 0 ]; then
    avg_duration=$(echo "scale=1; $total_duration / $completed_experiments" | bc -l 2>/dev/null || echo "$total_duration")
    echo "Average training time: ${avg_duration} minutes per experiment"
fi

echo ""
echo "=== NEXT STEPS ==="
echo "1. Analyze results: python analyze_results.py ${RESULTS_CSV}"
if [ -n "$best_result" ]; then
    local param_overrides=$(build_parameter_overrides "$best_param")
    echo "2. Train final model with best ${PARAMETER_NAME}: uv run -m src.train --config ${CONFIG_FILE}${param_overrides}"
fi
echo "3. Clean up logs: rm -rf ${BASE_LOG_DIR}"