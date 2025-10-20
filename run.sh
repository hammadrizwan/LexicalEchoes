#!/bin/bash
set -e  # stop if any script fails

echo "🚀 Starting sequential run..."

echo "▶ Running file1.py..."
python counterfact.py --data_type=penme --model_type=gemma-3-1b-it
echo "✅ Finished file1.py"

echo "▶ Running file2.py..."
python counterfact.py --data_type=penme --model_type=gemma-3-4b-it
echo "✅ Finished file2.py"

echo "▶ Running file3.py..."
python counterfact.py --data_type=penme --model_type=gemma-3-12b-it
echo "✅ Finished file3.py"

echo "🎉 All scripts completed successfully!"
