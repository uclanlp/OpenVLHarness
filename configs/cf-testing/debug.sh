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


# --data RealWorldQA MMStar SimpleVQA VStarBench FREAK_MCQ FREAK_VQA HallusionBench MMSIBench_wo_circular EmbSpatialBench ERQA MindCubeBench_tiny_raw_qa OmniSpatialBench MMBench_DEV_EN_V11 RealXBench MMSearch \
# python run_benchmark.py --config configs/cf-testing/v7_instruct.yaml --judge-config configs/cf-testing/judge.yaml \
#         --data RealWorldQA MMStar SimpleVQA VStarBench FREAK_MCQ FREAK_VQA HallusionBench MMSIBench_wo_circular EmbSpatialBench ERQA MindCubeBench_tiny_raw_qa OmniSpatialBench MMBench_DEV_EN_V11 RealXBench MMSearch \
#         --num-proc 100

python run_benchmark.py --config configs/cf-testing/v8_instruct.yaml --judge-config configs/cf-testing/judge.yaml \
        --data MMSIBench_wo_circular ERQA MindCubeBench_tiny_raw_qa OmniSpatialBench SparBench_tiny EmbSpatialBench \
        --num-proc 100



# python run_benchmark.py --config configs/cf-testing/gpt_agent_with_tools.yaml --judge-config configs/cf-testing/judge.yaml \
#         --data RealXBench VStarBench FREAK_MCQ FREAK_VQA ERQA MMSIBench_wo_circular FSC147_val MindCubeBench_tiny_raw_qa RealWorldQA MMStar SimpleVQA HallusionBench EmbSpatialBench OmniSpatialBench MMBench_DEV_EN_V11 \
#         --num-proc 10

# python run_benchmark.py --config configs/cf-testing/gpt_agent.yaml --judge-config configs/cf-testing/judge.yaml \
#         --data MMSIBench_wo_circular MindCubeBench_tiny_raw_qa OmniSpatialBench SparBench_tiny\
#         --num-proc 10 --resume


# python run_benchmark.py --config configs/cf-testing/base_agent.yaml --judge-config configs/cf-testing/judge.yaml \
#         --data RealWorldQA MMStar SimpleVQA VStarBench FREAK_MCQ FREAK_VQA HallusionBench MMSIBench_wo_circular EmbSpatialBench ERQA MindCubeBench_tiny_raw_qa OmniSpatialBench MMBench_DEV_EN_V11 \
#         --num-proc 100 --resume


# export LMUData=/data/<user>/lmu_data
# python run_benchmark.py --config configs/cf-testing/base_agent.yaml --judge-config configs/cf-testing/judge.yaml \
#         --data CountQA PixmoCount_val \
#         --num-proc 100

# export LMUData=/data/<user>/lmu_data
# python run_benchmark.py --config configs/cf-testing/base_agent_with_tools_and_code.yaml --judge-config configs/cf-testing/judge.yaml --data FSC147_val --num-proc 10
# python run_benchmark.py --config configs/cf-testing/base_agent_with_tools_and_code_with_exp.yaml --judge-config configs/cf-testing/judge.yaml --data FSC147_val --num-proc 10
# python run_benchmark.py --config configs/cf-testing/base_agent_with_new_memory.yaml --judge-config configs/cf-testing/judge.yaml --data FSC147_val CountQA PixmoCount_val --num-proc 100 --num-examples 10

