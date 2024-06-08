# model settings
# Only for evaluation
img_size=224,
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='DVAN',
        attn_module="DCNv3_SW_KA",
        kernel_size=[5, [1, 11]],
        pad=[2, [0, 5]],
        embed_dims=[32, 64, 180, 256],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.1,
        depths=[4, 4, 18, 4],
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='BN', requires_grad=True),
        channel_attention=True),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=256,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))
