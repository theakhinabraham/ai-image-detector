# AI-Generated Image Detector

MCA project: a classifier trained in this repo that estimates whether an image is real or AI-generated.
Full documentation comes in Phase 7.

## Setup (macOS)
```bash
brew install python@3.11
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
# Kaggle token: kaggle.com -> Settings -> API -> Create New Token, then
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images -p data/raw/cifake --unzip
python -I scripts/inspect_data.py --config configs/smallcnn.yaml
python -m pytest -q
```
