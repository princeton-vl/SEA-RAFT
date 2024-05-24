# The training scripts are tested on 8 NVIDIA L40 GPUs

# Stage0: Stereo pretraining on TartanAir
python -u train.py --cfg config/train/Tartan480x640-M.json
# Stage1: FlyingChairs, please update your ckpts path (from Stage0 results) in the config file (restore_ckpt)
python -u train.py --cfg config/train/Tartan-C368x496-M.json
# Stage2: FlyingThings3D, please update your ckpts path (from Stage1 results) in the config file (restore_ckpt)
python -u train.py --cfg config/train/Tartan-C-T432x960-M.json
# Stage3: FlyingThings + Sintel + KITTI + HD1K, please update your ckpts path (from Stage2 results) in the config file (restore_ckpt)
# The ckpts from this stage are used for sintel submission 
python -u train.py --cfg config/train/Tartan-C-T-TSKH432x960-M.json

# Stage4 (finetune for KITTI submission): KITTI, please update your ckpts path (from Stage3 results) in the config file (restore_ckpt)
python -u train.py --cfg config/train/Tartan-C-T-TSKH-kitti432x960-M.json

# Stage5 (finetune for Spring submission): Spring, please update your ckpts path (from Stage3 results) in the config file (restore_ckpt
python -u train.py --cfg config/train/Tartan-C-T-TSKH-spring540x960-M.json
