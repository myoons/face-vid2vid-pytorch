export CUDA_VISIBLE_DEVICES=0

python -m torch.distributed.run run.py \
    --config config/vox-256.yaml \
    --mode train \
    --device_ids 0