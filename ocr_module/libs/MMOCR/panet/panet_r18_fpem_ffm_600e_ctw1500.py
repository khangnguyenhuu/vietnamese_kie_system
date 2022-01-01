optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='poly', power=0.9)
total_epochs = 5000
checkpoint_config = dict(interval=10)
log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
model = dict(
    type='PANet',
    backbone=dict(
        type='mmdet.ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
        norm_eval=True,
        style='caffe'),
    neck=dict(type='FPEM_FFM', in_channels=[64, 128, 256, 512]),
    bbox_head=dict(
        type='PANHead',
        text_repr_type='poly',
        in_channels=[128, 128, 128, 128],
        out_channels=6,
        loss=dict(type='PANLoss')),
    train_cfg=None,
    test_cfg=None)
dataset_type = 'IcdarDataset'
data_root = 'data'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='ColorJitter', brightness=0.12549019607843137, saturation=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='ScaleAspectJitter',
        img_scale=[(3000, 640)],
        ratio_range=(0.7, 1.3),
        aspect_ratio_range=(0.9, 1.1),
        multiscale_mode='value',
        keep_ratio=False),
    dict(type='PANetTargets', shrink_ratio=(1.0, 0.7)),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='RandomRotateTextDet'),
    dict(
        type='RandomCropInstances',
        target_size=(640, 640),
        instance_key='gt_kernels'),
    dict(type='Pad', size_divisor=32),
    dict(
        type='CustomFormatBundle',
        keys=['gt_kernels', 'gt_mask'],
        visualize=dict(flag=False, boundary_key='gt_kernels')),
    dict(type='Collect', keys=['img', 'gt_kernels', 'gt_mask'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(3000, 640),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(3000, 640), keep_ratio=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='IcdarDataset',
        ann_file='data/instances_training.json',
        img_prefix='data/imgs',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                color_type='color_ignore_orientation'),
            dict(
                type='LoadTextAnnotations',
                with_bbox=True,
                with_mask=True,
                poly2mask=False),
            dict(
                type='ColorJitter',
                brightness=0.12549019607843137,
                saturation=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='ScaleAspectJitter',
                img_scale=[(3000, 640)],
                ratio_range=(0.7, 1.3),
                aspect_ratio_range=(0.9, 1.1),
                multiscale_mode='value',
                keep_ratio=False),
            dict(type='PANetTargets', shrink_ratio=(1.0, 0.7)),
            dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
            dict(type='RandomRotateTextDet'),
            dict(
                type='RandomCropInstances',
                target_size=(640, 640),
                instance_key='gt_kernels'),
            dict(type='Pad', size_divisor=32),
            dict(
                type='CustomFormatBundle',
                keys=['gt_kernels', 'gt_mask'],
                visualize=dict(flag=False, boundary_key='gt_kernels')),
            dict(type='Collect', keys=['img', 'gt_kernels', 'gt_mask'])
        ]),
    val=dict(
        type='IcdarDataset',
        ann_file='data/instances_test.json',
        img_prefix='data/imgs',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                color_type='color_ignore_orientation'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(3000, 640),
                flip=False,
                transforms=[
                    dict(
                        type='Resize', img_scale=(3000, 640), keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='IcdarDataset',
        ann_file='data/instances_test.json',
        img_prefix='data/imgs',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                color_type='color_ignore_orientation'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(3000, 640),
                flip=False,
                transforms=[
                    dict(
                        type='Resize', img_scale=(3000, 640), keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=10, metric='hmean-iou')
work_dir = 'panet'
gpu_ids = range(0, 1)
