#!/bin/bash

# Real-time monitor for hyperparameter search
# Usage: ./monitor_search.sh [results_csv]

RESULTS_CSV=${1:-"lr_search_results.csv"}
LOG_DIR=${2:-"logs/lr_search"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to display header
show_header() {
    clear
    echo -e "${BLUE}=================================${NC}"
    echo -e "${BLUE}   HYPERPARAMETER SEARCH MONITOR${NC}"
    echo -e "${BLUE}=================================${NC}"
    echo -e "Results file: ${RESULTS_CSV}"
    echo -e "Log directory: ${LOG_DIR}"
    echo -e "Last updated: $(date)"
    echo ""
}

# Function to show GPU status
show_gpu_status() {
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}GPU Status:${NC}"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
        while IFS=',' read -r idx name util mem_used mem_total temp; do
            printf "  GPU %s: %s | Util: %s%% | Mem: %s/%s MB | Temp: %s°C\n" \
                   "$idx" "$name" "$util" "$mem_used" "$mem_total" "$temp"
        done
        echo ""
    fi
}

# Function to show experiment status
show_experiment_status() {
    if [ ! -f "$RESULTS_CSV" ]; then
        echo -e "${RED}Results file not found: $RESULTS_CSV${NC}"
        return
    fi

    echo -e "${YELLOW}Experiment Status:${NC}"
    
    # Count experiments by status, ensuring default to 0 and trimming whitespace
    # Using 'tr -d "\n"' to remove potential newlines from wc -l output
    local total=$(tail -n +2 "$RESULTS_CSV" 2>/dev/null | wc -l | tr -d '\n' || echo 0)
    local completed=$(tail -n +2 "$RESULTS_CSV" 2>/dev/null | grep -c "completed" | tr -d '\n' || echo 0)
    local failed=$(tail -n +2 "$RESULTS_CSV" 2>/dev/null | grep -c "failed" | tr -d '\n' || echo 0)
    
    # Debugging prints for the raw values before arithmetic
    echo "DEBUG (monitor): total='$total', completed='$completed', failed='$failed'"

    local running=$((total - completed - failed))
    
    echo -e "  Total: $total | ${GREEN}Completed: $completed${NC} | ${RED}Failed: $failed${NC} | ${BLUE}Running: $running${NC}"
    echo ""
    
    # Show recent results
    if [ "$total" -gt 0 ]; then # Use quotes for numerical comparison
        echo -e "${YELLOW}Recent Results:${NC}"
        printf "  %-12s | %-10s | %-13s | %-8s | %s\n" "Learning Rate" "Final Loss" "Best Val Loss" "Duration" "Status"
        echo "  -------------|------------|---------------|----------|--------"
        
        tail -n +2 "$RESULTS_CSV" | tail -5 | while IFS=',' read -r lr final_loss best_val_loss exp_name gpu_id status start_time end_time duration; do
            case $status in
                "completed") color=$GREEN ;;
                "failed") color=$RED ;;
                *) color=$NC ;;
            esac
            printf "  %-12s | %-10s | %-13s | %-8s | ${color}%s${NC}\n" \
                   "$lr" "$final_loss" "$best_val_loss" "${duration}m" "$status"
        done
        echo ""
    fi
}

# Function to show best result so far
show_best_result() {
    if [ ! -f "$RESULTS_CSV" ]; then
        return
    fi

    local best_result=$(tail -n +2 "$RESULTS_CSV" 2>/dev/null | grep "completed" | sort -t',' -k3 -n | head -1)
    if [ -n "$best_result" ]; then
        echo -e "${GREEN}Best Result So Far:${NC}"
        local best_lr=$(echo "$best_result" | cut -d',' -f1)
        local best_val_loss=$(echo "$best_result" | cut -d',' -f3)
        local best_final_loss=$(echo "$best_result" | cut -d',' -f2)
        echo -e "  Learning Rate: $best_lr"
        echo -e "  Validation Loss: $best_val_loss"
        echo -e "  Final Training Loss: $best_final_loss"
        echo ""
    fi
}

# Function to show active processes
show_active_processes() {
    echo -e "${YELLOW}Active Training Processes:${NC}"
    
    # Find Python processes running train.py
    local processes=$(ps aux | grep -E "python.*train\.py" | grep -v grep)
    if [ -n "$processes" ]; then
        echo "$processes" | while read -r line; do
            local pid=$(echo "$line" | awk '{print $2}')
            # Extract GPU ID and LR more robustly, considering they might not always be present
            local gpu_id=$(echo "$line" | grep -o "training\.device=cuda:[0-9]" | cut -d':' -f2 || echo "N/A")
            local lr=$(echo "$line" | grep -o "scheduler\.max_learning_rate=[0-9e.-]*" | cut -d'=' -f2 || echo "N/A")
            echo -e "  PID: $pid | GPU: $gpu_id | LR: $lr"
        done
    else
        echo -e "  No active training processes found"
    fi
    echo ""
}

# Function to show log tail for active experiments
show_recent_logs() {
    if [ ! -d "$LOG_DIR" ]; then
        return
    fi

    echo -e "${YELLOW}Recent Log Activity:${NC}"
    
    # Find the most recently modified log file
    local recent_log=$(find "$LOG_DIR" -name "*.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2)
    
    if [ -n "$recent_log" ]; then
        local exp_name=$(basename "$recent_log" .log)
        echo -e "  Latest experiment: $exp_name"
        echo -e "  ${BLUE}Last 3 lines:${NC}"
        tail -3 "$recent_log" 2>/dev/null | sed 's/^/    /'
    else
        echo -e "  No log files found"
    fi
    echo ""
}

# Main monitoring loop
main() {
    echo -e "${GREEN}Starting hyperparameter search monitor...${NC}"
    echo -e "${YELLOW}Press Ctrl+C to exit${NC}"
    sleep 2

    while true; do
        show_header
        show_gpu_status
        show_experiment_status
        show_best_result
        show_active_processes
        show_recent_logs
        
        echo -e "${BLUE}Refreshing in 30 seconds...${NC}"
        sleep 30
    done
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n${GREEN}Monitor stopped${NC}"; exit 0' INT

# Run main function
main
