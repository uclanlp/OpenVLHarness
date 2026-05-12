conda create -n da3 python=3.12 -y
conda activate da3
pip install --index-url https://download.pytorch.org/whl/cu126 "torch>=2" torchvision
pip install --index-url https://download.pytorch.org/whl/cu126 xformers
# For h100 servers uncomment this line
# conda install -c nvidia "cuda-nvcc=12.6" "cuda-cudart-dev=12.6" -y
export CUDA_HOME=$CONDA_PREFIX
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
pip install "depth-anything-3[app] @ git+https://github.com/ByteDance-Seed/Depth-Anything-3.git"
