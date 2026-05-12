conda create -n sa3 python=3.12 -y
conda activate sa3
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "sam3 @ git+https://github.com/facebookresearch/sam3.git"
pip install "setuptools<81" einops psutil fastapi uvicorn python-multipart pycocotools "opencv-python<4.12" matplotlib
