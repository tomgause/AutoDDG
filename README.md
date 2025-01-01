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
Once you have prepared your data, you can test DAVE few-shot. First, be sure to download the **DAVE_3_shot.pth** and **verification.pth** files from the DAVE repo.
```
python scripts/DAVE_few_shot_eval.py --skip_train --model_name DAVE_3_shot --backbone resnet50 --swav_backbone --reduction 8 --num_enc_layers 3 --num_dec_layers 3 --kernel_dim 3 --emb_dim 256 --num_objects 3 --num_workers 8 --use_query_pos_emb --use_objectness --use_appearance --batch_size 1 --pre_norm
```