export CUDA_VISIBLE_DEVICES=1,2,3

python -m torch.distributed.launch run.py \
    --config config/vox_256_0.25_occlusion.yaml \
    --mode train \
    --device_ids 0,1,2