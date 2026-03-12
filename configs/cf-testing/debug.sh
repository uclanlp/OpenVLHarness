# python run_experience_learning.py \
#     --config configs/cf-testing/base_agent_with_tools.yaml \
#     --reflection-config configs/cf-testing/reflection.yaml \
#     --judge-config configs/cf-testing/judge.yaml \
#     --data RealWorldQA MMStar \
#     --num-proc 10 \
#     --no-reflection \

# python run_experience_learning.py \
#     --config configs/cf-testing/base_agent_with_tools_3vl.yaml \
#     --reflection-config configs/cf-testing/reflection_v1_ins.yaml \
#     --judge-config configs/cf-testing/judge.yaml \
#     --data RealWorldQA MMStar \
#     --num-proc 10 \

# python run_experience_learning.py \
#     --config configs/cf-testing/base_agent_with_tools_3vl.yaml \
#     --reflection-config configs/cf-testing/reflection_v2_ins.yaml \
#     --judge-config configs/cf-testing/judge.yaml \
#     --data RealWorldQA MMStar \
#     --num-proc 10 \

python run_experience_learning.py \
    --config configs/cf-testing/base_agent_with_tools_3vl.yaml \
    --reflection-config configs/cf-testing/reflection_v1_sum.yaml \
    --judge-config configs/cf-testing/judge.yaml \
    --data RealWorldQA MMStar \
    --num-proc 10 \

python run_experience_learning.py \
    --config configs/cf-testing/base_agent_with_tools_3vl.yaml \
    --reflection-config configs/cf-testing/reflection_v2_sum.yaml \
    --judge-config configs/cf-testing/judge.yaml \
    --data RealWorldQA MMStar \
    --num-proc 10 \
