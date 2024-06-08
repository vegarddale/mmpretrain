# model settings
# Only for evaluation
img_size=224,
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='DVAN',
        attn_module="DCNv3_SW_KA",
        kernel_size=[5, [1, 7]],
        pad=[2, [0, 3]],
        embed_dims=[32, 64, 160, 256],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.1,
        depths=[6, 6, 24, 6],
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(eps=1e-06, requires_grad=True, type='LN'),
        channel_attention=True),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=256,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))
