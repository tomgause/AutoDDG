We recommend using UV for package management.
```
git clone https://github.com/tomgause/AutoDDG.git
cd AutoDDG
uv init
uv sync
source .venv/bin/activate
```
Download Aerial Seabirds West Africa dataset.
```
python scripts/prepare_datasets.py --dataset "~/Work/amnh/ddg/datasets/aerial_seabirds_west_africa"
```
Once you have prepared your data, you can test DAVE few-shot. First, be sure to download the **DAVE_3_shot.pth** and **verification.pth** files from the DAVE repo and put them under `weights`.
```
python scripts/eval.py scripts/configs/DAVE_birds_few_shot_eval.json
```