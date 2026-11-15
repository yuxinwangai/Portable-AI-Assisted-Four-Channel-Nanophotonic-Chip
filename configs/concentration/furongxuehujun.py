auto_scale_lr = dict(base_batch_size=16)
data_preprocessor = dict(
    mean=[
        48.36210632324219,
        71.38200759887695,
        94.03285598754883,
    ],
    num_classes=7,
    std=[
        28.05500713267208,
        36.83280731212999,
        28.262580704192363,
    ],
    to_rgb=True)
dataset_type = 'CustomDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=1, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='VisualizationHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
mean_B_list = 94.03285598754883
mean_G_list = 71.38200759887695
mean_R_list = 48.36210632324219
model = dict(
    backbone=dict(
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        type='ResNet'),
    head=dict(
        dropout_rate=0.5,
        in_channels=2048,
        loss=dict(loss_weight=1.0, type='CrossEntropyLoss'),
        mid_channels=[1024],
        num_classes=7,
        topk=(
            1,
            5,
        ),
        type='StackedLinearClsHead'),
    neck=dict(type='GlobalAveragePooling'),
    type='ImageClassifier')
optim_wrapper = dict(
    optimizer=dict(lr=0.15, momentum=0.9, type='SGD', weight_decay=0.001))
param_scheduler = dict(T_max=300, by_epoch=True, type='CosineAnnealingLR')
randomness = dict(deterministic=False, seed=None)
resume = False
std_B_list = 28.262580704192363
std_G_list = 36.83280731212999
std_R_list = 28.05500713267208
test_cfg = dict()
test_dataloader = dict(
    batch_size=8,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='test.txt',
        classes=[
            '10^1 CFU/mL',
            '10^2 CFU/mL',
            '10^3 CFU/mL',
            '10^4 CFU/mL',
            '10^5 CFU/mL',
            '10^6 CFU/mL',
            '10^7 CFU/mL',
        ],
        data_prefix='test',
        data_root='../phone_portable_260212/furongxuehujun_dataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset'),
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(topk=(1, ), type='Accuracy'),
    dict(type='SingleLabelMetric'),
    dict(type='ConfusionMatrix'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=1)
train_dataloader = dict(
    batch_size=16,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='train.txt',
        classes=[
            '10^1',
            '10^2',
            '10^3',
            '10^4',
            '10^5',
            '10^6',
            '10^7',
        ],
        data_prefix='train',
        data_root='../phone_portable_260212/furongxuehujun_dataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(direction='vertical', prob=0.5, type='RandomFlip'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(direction='vertical', prob=0.5, type='RandomFlip'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=8,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='val.txt',
        classes=[
            '10^1',
            '10^2',
            '10^3',
            '10^4',
            '10^5',
            '10^6',
            '10^7',
        ],
        data_prefix='val',
        data_root='../phone_portable_260212/furongxuehujun_dataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset'),
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(topk=(1, ), type='Accuracy'),
    dict(type='SingleLabelMetric'),
    dict(type='ConfusionMatrix'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'work_fold'

