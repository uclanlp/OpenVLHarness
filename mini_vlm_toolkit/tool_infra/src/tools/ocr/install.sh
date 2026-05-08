conda create -n ocr python=3.12 -y
conda activate ocr
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126/
python -m pip install "transformers>=5.0.0"
pip install flash-attn --no-build-isolation
python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -m pip install -U "paddleocr[doc-parser]"
pip install python-multipart
