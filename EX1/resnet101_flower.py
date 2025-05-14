_base_ = [
    '../_base_/models/resnet101.py',         # 模型架构
    '../_base_/datasets/imagenet_bs64.py',  # 数据加载
    '../_base_/schedules/imagenet_bs256.py', # 训练计划
    '../_base_/default_runtime.py'          # 默认运行时设置
]

model = dict(
    backbone=dict(
        frozen_stages=2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_8xb32_in1k_20210831-539c63f8.pth',
            prefix='backbone',
        )
    ),
    head=dict(num_classes=5,
              topk=(1,),
    )
)

data_preprocessor = dict(
    num_classes=5,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)
# bgr_mean =[123.675, 116.28, 103.53]
# bgr_std = [58.395, 57.12, 57.375]

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224, backend='pillow', interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    # 随机增强
    # dict(
    #     type='RandAugment',
    #     policies='timm_standard',  # 使用timm库中的标准策略
    #     num_policies=2,
    #     total_level=10,
    #     magnitude_level=5,
    #     magnitude_std=0.5,
    #     hparams=dict(
    #         pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(
        type='AutoAugment',
        policies='imagenet',
        hparams=dict(pad_val=[round(x) for x in bgr_mean])),
    dict(
        type='RandomErasing',
        erase_prob=0.15,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=bgr_mean,
        fill_std=bgr_std),
    dict(type='PackClsInputs'),
]
test_pipeline = [
    # 加载图像
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=224),
    dict(type='PackClsInputs'),
]
train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    dataset=dict(
        type='ImageNet',
        data_root='data/flower_dataset',
        ann_file='train.txt',
        data_prefix='train',
        classes='data/flower_dataset/classes.txt',
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=64,
    num_workers=4,
    dataset=dict(
        type='ImageNet',
        data_root='data/flower_dataset',
        ann_file='val.txt',
        data_prefix='val',
        classes='data/flower_dataset/classes.txt',
        pipeline=test_pipeline,
    )
)
test_dataloader = val_dataloader

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, weight_decay=0.0001),
    # norm_type: 使用的范数类型，此处使用范数2。
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    # 预热
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    # 主要学习率策略
    dict(
        type='CosineAnnealingLR',
        T_max=295,
        eta_min=1.0e-6,
        by_epoch=True,
        begin=5,
        end=100),
    # 配置动量调整策略 使损失函数收敛更快
    dict(
        type='CosineAnnealingParamScheduler',
        param_name='weight_decay',
        eta_min=0.00001,
        by_epoch=True,
        begin=0,
        end=100)
]

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)

val_evaluator = dict(type='Accuracy', topk=(1,))
test_evaluator = val_evaluator

load_from = 'checkpoints/resnet101-5d3b4d8f.pth'

# load_from = 'D:/model_cache/hub/checkpoints/resnet101-5d3b4d8f.pth'
#load_from = 'resnet152-394f9c45.pth'
