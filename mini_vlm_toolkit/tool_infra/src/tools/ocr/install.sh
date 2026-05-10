conda create -n ocr python=3.12 -y
conda activate ocr
python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -m pip install -U "paddleocr[doc-parser]"
paddleocr install_genai_server_deps vllm
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu126torch2.8-cp312-cp312-linux_x86_64.whl