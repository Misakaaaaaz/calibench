#!/bin/bash
# Script to monitor experiment progress

echo "=========================================="
echo "SMART Experiments Status Monitor"
echo "Time: $(date)"
echo "=========================================="

echo ""
echo "=== RUNNING PROCESSES ==="
ps aux | grep -E "run_sensitivity|run_combination|run_imbalance" | grep -v grep | wc -l
echo "processes are running"

echo ""
echo "=== EXPERIMENT 1: Hyperparameter Sensitivity ==="
if [ -f "/hdd/haolan/SMART/LogitsGap/experiment1_full.log" ]; then
    echo "Last 3 lines:"
    tail -3 /hdd/haolan/SMART/LogitsGap/experiment1_full.log
    echo ""
    grep "Experiment [0-9]*/20" /hdd/haolan/SMART/LogitsGap/experiment1_full.log | tail -1
fi

echo ""
echo "=== EXPERIMENT 2: Method Combinations ==="
if [ -f "/hdd/haolan/SMART/LogitsGap/experiment2_full.log" ]; then
    echo "Last 3 lines:"
    tail -3 /hdd/haolan/SMART/LogitsGap/experiment2_full.log
fi

echo ""
echo "=== EXPERIMENT 3: Class Imbalance ==="
if [ -f "/hdd/haolan/SMART/LogitsGap/experiment3_full.log" ]; then
    echo "Last 3 lines:"
    tail -3 /hdd/haolan/SMART/LogitsGap/experiment3_full.log
fi

echo ""
echo "=========================================="

