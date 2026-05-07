# export LMUData=/data/<user>/lmu_data
# python run_experience_learning.py \
#     --config configs/cf-testing/base_agent_with_tools_3vl.yaml \
#     --reflection-config configs/cf-testing/reflection.yaml \
#     --judge-config configs/cf-testing/judge.yaml \
#     --data FSC147_val \
#     --num-proc 10 \
#     --no-reflection \

# python run_experience_learning.py \
#     --config configs/cf-testing/base_agent_with_tools_3vl.yaml \
#     --reflection-config configs/cf-testing/reflection_v1_ins.yaml \
#     --judge-config configs/cf-testing/judge.yaml \
#     --data FSC147_val \
#     --num-proc 10 \

# python run_experience_learning.py \
#     --config configs/cf-testing/base_agent_with_tools_3vl.yaml \
#     --reflection-config configs/cf-testing/reflection_v2_ins.yaml \
#     --judge-config configs/cf-testing/judge.yaml \
#     --data RealWorldQA MMStar \
#     --num-proc 10 \

# python run_experience_learning.py \
#     --config configs/cf-testing/base_agent_with_tools_3vl.yaml \
#     --reflection-config configs/cf-testing/reflection_v1_sum.yaml \
#     --judge-config configs/cf-testing/judge.yaml \
#     --data FSC147_val \
#     --num-proc 10 \

# python run_experience_learning.py \
#     --config configs/cf-testing/base_agent_with_tools_3vl.yaml \
#     --reflection-config configs/cf-testing/reflection_v2_sum.yaml \
#     --judge-config configs/cf-testing/judge.yaml \
#     --data RealWorldQA MMStar \
#     --num-proc 10 \

# python run_benchmark.py --config configs/cf-testing/base_agent_with_tools_depth.yaml --judge-config configs/cf-testing/judge.yaml --data RealWorldQA --num-proc 10 --num-examples 20
# python run_benchmark.py --config configs/cf-testing/base_agent_with_tools_and_code.yaml --judge-config configs/cf-testing/judge.yaml --data RealWorldQA --num-proc 10
# python run_benchmark.py --config configs/cf-testing/base_agent_with_tools_and_code_with_exp.yaml --judge-config configs/cf-testing/judge.yaml --data RealWorldQA --num-proc 10
# python run_benchmark.py --config configs/cf-testing/base_agent_with_tools_and_code_2x.yaml --judge-config configs/cf-testing/judge.yaml --data RealWorldQA --num-proc 10

python run_benchmark.py --config configs/cf-testing/base_agent_with_new_memory.yaml --judge-config configs/cf-testing/judge.yaml \
        --data MMBench_DEV_EN_V11 RealWorldQA MMStar SimpleVQA VStarBench FREAK_MCQ FREAK_VQA HallusionBench MMSIBench_wo_circular EmbSpatialBench ERQA MindCubeBench_tiny_raw_qa OmniSpatialBench MMSearch RealXBench\
        --num-proc 100 --num-examples 10


export LMUData=/data/<user>/lmu_data
# python run_benchmark.py --config configs/cf-testing/base_agent_with_tools_and_code.yaml --judge-config configs/cf-testing/judge.yaml --data FSC147_val --num-proc 10
# python run_benchmark.py --config configs/cf-testing/base_agent_with_tools_and_code_with_exp.yaml --judge-config configs/cf-testing/judge.yaml --data FSC147_val --num-proc 10
python run_benchmark.py --config configs/cf-testing/base_agent_with_new_memory.yaml --judge-config configs/cf-testing/judge.yaml --data FSC147_val CountQA PixmoCount_val --num-proc 100 --num-examples 10

