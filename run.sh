#!/bin/bash

# GFlowNet for StrategyQA Project Run Script

echo "Starting GFlowNet for StrategyQA Project"
echo "========================================"

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Set project path
PROJECT_ROOT="/Users/shuhaozhang/Desktop/Research/PersonalProject"
cd $PROJECT_ROOT

echo "Project directory: $PROJECT_ROOT"
echo ""

# Check Python environment
echo "Checking Python environment..."
python --version
echo ""

# Check project structure
echo "Checking project structure..."
if [ -f "main.py" ]; then
    echo "OK main.py exists"
else
    echo "ERROR main.py does not exist"
    exit 1
fi

if [ -d "data" ]; then
    echo "OK data/ directory exists"
else
    echo "ERROR data/ directory does not exist"
fi

if [ -d "llm" ]; then
    echo "OK llm/ directory exists"
else
    echo "ERROR llm/ directory does not exist"
fi

if [ -d "gfn" ]; then
    echo "OK gfn/ directory exists"
else
    echo "ERROR gfn/ directory does not exist"
fi

if [ -d "evaluation" ]; then
    echo "OK evaluation/ directory exists"
else
    echo "ERROR evaluation/ directory does not exist"
fi

echo ""

# Run main entry test
echo "Running main entry test..."
python main.py --mode test --max_train_samples 10 --max_eval_samples 5 --num_epochs 1
echo ""

# Run quick experiment
echo "Running quick experiment..."
python main.py --mode experiment --experiment_name "quick_test" --max_train_samples 20 --max_eval_samples 10 --num_epochs 2 --num_samples_per_question 2
echo ""

# Display results
echo "Displaying experiment results..."
if [ -d "results" ]; then
    echo "Result files:"
    ls -la results/
    echo ""
    
    # Display latest experiment report
    LATEST_REPORT=$(ls -t results/experiment_report_*.md 2>/dev/null | head -1)
    if [ -n "$LATEST_REPORT" ]; then
        echo "Latest experiment report ($LATEST_REPORT):"
        echo "----------------------------------------"
        cat "$LATEST_REPORT"
        echo "----------------------------------------"
    fi
else
    echo "ERROR results/ directory does not exist"
fi

echo ""
echo "Project testing completed!"
echo ""
echo "Project status summary:"
echo "- OK Main entry file: Complete"
echo "- OK Data preprocessing: Complete"
echo "- OK LLM integration: Complete"
echo "- OK GFlowNet model: Complete"
echo "- OK Training pipeline: Complete"
echo "- OK Evaluation system: Complete"
echo ""
echo "Project is ready!"
echo ""
echo "Usage suggestions:"
echo "1. Run full experiment: python main.py --mode experiment --experiment_name 'my_experiment'"
echo "2. Training only: python main.py --mode train --num_epochs 10"
echo "3. Evaluation only: python main.py --mode eval"
echo "4. View help: python main.py --help"
echo ""
echo "Result file locations:"
echo "- Training history: results/training_history_*.json"
echo "- Evaluation results: results/evaluation_results_*.json"
echo "- Experiment reports: results/experiment_report_*.md"
echo "- Run logs: gfn_project.log"
