export CUDA_VISIBLE_DEVICES=1,2,3

python -m torch.distributed.launch run.py \
    --config config/vox-256.yaml \
    --mode train \
    --device_ids 0,1,2