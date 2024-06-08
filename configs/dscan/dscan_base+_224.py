_base_ = [
    '../_base_/models/dscan/base+_224.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_dscan.py',
    '../_base_/default_runtime.py'
]

train_dataloader = dict(batch_size=64)
val_dataloader = dict(batch_size=64)