@echo off
git init
echo venv/ > .gitignore
echo .env >> .gitignore
py -3.13 -m venv venv
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python scripts\verify_cuda.py
