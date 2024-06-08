optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        # for batch in each gpu is 128, 8 gpu
        # lr = 5e-4 * 64 * 1 / 512 = 6.25e-05
        lr=4e-4,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    # paramwise_cfg=dict(
    #     norm_decay_mult=0.0,
    #     bias_decay_mult=0.0,
    #     custom_keys={
    #         '.cls_token': dict(decay_mult=0.0),
    #     }),
)

# learning policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=4e-4,
        by_epoch=True,
        begin=0,
        end=6,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=94,
        eta_min=1e-5,
        by_epoch=True,
        begin=6,
        end=125)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=125, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=64)