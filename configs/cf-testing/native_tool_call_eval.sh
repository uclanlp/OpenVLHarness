export LMUData=/local/<user>/data/LMUData
python run_benchmark.py \
    --config configs/cf-testing/native_tool_call_agent.yaml \
    --judge-config configs/cf-testing/judge.yaml \
    --data RealWorldQA \
    --num-proc 10 \
    --save-html 20 \
    --incremental-json