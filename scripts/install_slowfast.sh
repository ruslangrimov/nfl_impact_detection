pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
git clone https://github.com/facebookresearch/detectron2 detectron2_repo
pip install -e detectron2_repo

git clone https://github.com/facebookresearch/slowfast
export PYTHONPATH=$(pwd)/slowfast:$PYTHONPATH
cd slowfast
python setup.py build develop
cd ..

