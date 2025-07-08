#!/bin/bash

# Usage: ./hyperparameter_search.sh [options]

set -e

# Default configuration
CONFIG_FILE="configs/baseline.yaml"
RESULTS_CSV="lr_search_results.csv"
NUM_GPUS=8
BASE_LOG_DIR="logs/lr_search"
BASE_CHECKPOINT_DIR="checkpoints/lr_search"
RESUME_SEARCH=false
SEARCH_TYPE="grid"

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

# Parse command line arguments
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
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --config PATH      Path to config file (default: configs/baseline.yaml)"
            echo "  --output PATH      Output CSV file (default: lr_search_results.csv)"
            echo "  --gpus NUM         Number of GPUs to use (default: 8)"
            echo "  --resume           Resume previous search"
            echo "  --search-type TYPE Search type: grid, random, or custom (default: grid)"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Learning rate configurations
declare -A SEARCH_CONFIGS
SEARCH_CONFIGS[grid]="1e-5 4e-5 7e5 1e-4 4e-4 7e-4 1e-3 4e-3 7e-3 1e-2"
SEARCH_CONFIGS[random]="2.3e-5 7.8e-5 1.2e-4 4.7e-4 8.9e-4 2.1e-3 5.6e-3 1.3e-2"
SEARCH_CONFIGS[custom]="1e-4 2e-4 3e-4 4e-4 5e-4"

# Get learning rates based on search type
IFS=' ' read -ra LEARNING_RATES <<< "${SEARCH_CONFIGS[$SEARCH_TYPE]}"

# Validation
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    exit 1
fi

if [ ${#LEARNING_RATES[@]} -eq 0 ]; then
    echo "Error: No learning rates defined for search type '$SEARCH_TYPE'"
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
    echo "learning_rate,final_loss,best_val_loss,experiment_name,gpu_id,status,start_time,end_time,duration_minutes,epochs,best_epoch" > "${RESULTS_CSV}"
fi

# Function to check if experiment already completed
is_experiment_completed() {
    local lr=$1

    (
        flock -s 200
        grep -q "^${lr}," "$RESULTS_CSV" && return 0
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

run_experiment() {
    local lr=$1
    local gpu_id=$2
    local exp_name="lr_${lr}_gpu_${gpu_id}"
    local log_file="${BASE_LOG_DIR}/${exp_name}.log"

    local start_time=$(date +%s)
    local start_time_str=$(date '+%Y-%m-%d %H:%M:%S')

    echo "[$start_time_str] Starting experiment: ${exp_name} on GPU ${gpu_id}"
    
    # Execute the training command
    # Output is redirected to a log file
    uv run -m src.train \
        --config="${CONFIG_FILE}" \
        scheduler.max_learning_rate=${lr} \
        optimizer.lr=${lr} \
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
        echo "${lr},${final_loss},${best_val_loss},${exp_name},${gpu_id},${status},${start_time_str},${end_time_str},${duration},${epochs},${best_epoch}" >> "${RESULTS_CSV}"
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
    local total=${#LEARNING_RATES[@]}
    
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
                local best_lr=$(echo "$best_result" | cut -d',' -f1)
                local best_val_loss=$(echo "$best_result" | cut -d',' -f3)
                echo -ne "Best: LR=$best_lr, Val=$best_val_loss"
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
echo "Search type: $SEARCH_TYPE"
echo "Learning rates: ${LEARNING_RATES[*]}"
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

for lr in "${LEARNING_RATES[@]}"; do
    # Skip if already completed and resuming
    if [ "$RESUME_SEARCH" = true ] && is_experiment_completed "$lr"; then
        echo "Skipping already completed experiment for LR: $lr"
        ((experiments_skipped++))
        continue
    fi
    
    # Wait for available GPU if all are busy
    wait_for_gpu
    
    # Run experiment in background and store PID
    run_experiment "$lr" "$gpu_counter" &
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
printf "%-12s | %-10s | %-13s | %-8s | %-8s | %-10s | %s\n" "Learning Rate" "Final Loss" "Best Val Loss" "Duration" "Epochs" "Best Epoch" "Status"
echo "-------------|------------|---------------|----------|----------|------------|--------"
# Corrected: Use flock directly on the file for shared lock
flock -s "${RESULTS_CSV}.lock" -c "tail -n +2 \"${RESULTS_CSV}\" | while IFS=',' read -r lr final_loss best_val_loss exp_name gpu_id status start_time end_time duration epochs best_epoch; do
    printf \"%-12s | %-10s | %-13s | %-8s | %-8s | %-10s | %s\n\" \"\$lr\" \"\$final_loss\" \"\$best_val_loss\" \"\${duration}m\" \"\$epochs\" \"\$best_epoch\" \"\$status\"
done"

# Enhanced best learning rate analysis
echo ""
echo "=== BEST LEARNING RATE ANALYSIS ==="
# Corrected: Use flock directly on the file for shared lock
best_result=$(flock -s "${RESULTS_CSV}.lock" -c "tail -n +2 \"${RESULTS_CSV}\" | grep \"completed\" | sort -t',' -k3 -n | head -1")

if [ -n "$best_result" ]; then
    IFS=',' read -r best_lr best_final_loss best_val_loss exp_name gpu_id status start_time end_time duration epochs best_epoch <<< "$best_result"
    echo "🏆 Best learning rate: ${best_lr}"
    echo "   Validation loss: ${best_val_loss}"
    echo "   Final training loss: ${best_final_loss}"
    echo "   Training duration: ${duration} minutes"
    echo "   Total epochs: ${epochs}"
    echo "   Best epoch: ${best_epoch}"
    echo "   Best model checkpoint: ${BASE_CHECKPOINT_DIR}/${exp_name}/best.pt"
    echo ""
    echo "🚀 To use this configuration:"
    echo "   python src/train.py --config ${CONFIG_FILE} scheduler.max_learning_rate=${best_lr} optimizer.lr=${best_lr}"
else
    echo "No successful experiments found"
fi

# Summary statistics
echo ""
echo "=== SUMMARY STATISTICS ==="

echo "Total experiments: $(flock -s "${RESULTS_CSV}.lock" -c "tail -n +2 \"${RESULTS_CSV}\" | wc -l")"
echo "Completed: $(flock -s "${RESULTS_CSV}.lock" -c "tail -n +2 \"${RESULTS_CSV}\" | grep -c \"completed\"")"
echo "Failed: $(flock -s "${RESULTS_CSV}.lock" -c "tail -n +2 \"${RESULTS_CSV}\" | grep -c \"failed\"")"
echo "Total training time: $(flock -s "${RESULTS_CSV}.lock" -c "tail -n +2 \"${RESULTS_CSV}\" | cut -d',' -f9 | awk '{sum+=\$1} END {print sum}'")"

if [ "$completed_experiments" -gt 0 ]; then
    avg_duration=$(echo "scale=1; $total_duration / $completed_experiments" | bc -l 2>/dev/null || echo "$total_duration")
    echo "Average training time: ${avg_duration} minutes per experiment"
fi

echo ""
echo "=== NEXT STEPS ==="
echo "1. Analyze results: python analyze_results.py ${RESULTS_CSV}"
echo "2. Train final model with best LR: python src/train.py --config ${CONFIG_FILE} scheduler.max_learning_rate=${best_lr} optimizer.lr=${best_lr}"
echo "3. Clean up logs: rm -rf ${BASE_LOG_DIR}"
echo "4. Clean up checkpoints: rm -rf ${BASE_CHECKPOINT_DIR}"
