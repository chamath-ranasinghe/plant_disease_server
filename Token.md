mkdir -p ~/.kaggle
cp ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json


pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121