patch_size = (128, 128, 128)
intermedia_size = 25
win_level = 150
win_width = 800
constant_shift = 3

trainner = dict(type='Trainner', runner_config=dict(type='EpochBasedRunner'))
model = dict(
    type='CorSkSegNetwork',
    backbone=dict(type='ResUnet', in_ch=2, channels=16, blocks=3),
    apply_sync_batchnorm=True,
    head=dict(
        type='CoronarySkSegHead',
        in_channels=16,
        scale_factor=(2.0, 2.0, 2.0),
    ),

    pipeline=[dict(
            type='Augmentation3d',
            aug_parameters={
                'rot_range_x': (-10.0, 10.0),
                'rot_range_y': (-10.0, 10.0),
                'rot_range_z': (-10.0, 10.0),
                'scale_range_x': (1.0, 1.0),
                'scale_range_y': (1.0, 1.0),
                'scale_range_z': (1.0, 1.0),
                'shift_range_x': (-0.0, 0.0),
                'shift_range_y': (-0.0, 0.0),
                'shift_range_z': (-0.0, 0.0),
                'elastic_alpha': [3.0, 3.0, 3.0],  # x,y,z
                'smooth_num': 4,
                'field_size': [17, 17, 17],  # x,y,z
                'size_o': patch_size,
                'itp_mode_dict': {
                    'mask': 'bilinear'
                },
                'out_style': 'resize'
            }
        )]

)

train_cfg = None
test_cfg = None

data = dict(
    imgs_per_gpu=10,
    workers_per_gpu=3,
    shuffle=True,
    drop_last=False,
    dataloader=dict(
        type='SampleDataLoader',
        source_batch_size=4,
        source_thread_count=1,
        source_prefetch_count=1,
    ),
    train=dict(
        type='CorSkSeg_Sample_Dataset',
        dst_list_file='/media/tx-eva-20/ssd/data/cor/train/20210211_all/train_all.lst',
        patch_size=patch_size,
        win_level=win_level,
        win_width=win_width,
        constant_shift=constant_shift,
        shift_range=10,
        sk_noise_prob=0.9,
        joint_noise_ratio=0.2,
        joint_noise_range=2,
        joint_noise_count=3,
        other_noise_ratio=0.1,
        other_noise_range=1,
        sample_frequent=20,
    ),
)

optimizer = dict(type='Adam', lr=5e-4, weight_decay=5e-4)
optimizer_config = {}

lr_config = dict(policy='step', warmup='linear', warmup_iters=90, warmup_ratio=1.0 / 3, step=[25, 75], gamma=0.2)

checkpoint_config = dict(interval=5)

log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])

cudnn_benchmark = False
work_dir = './checkpoints/skseg_1.0/2021_02_23'
gpus = 3
find_unused_parameters = True
total_epochs = 100
autoscale_lr = None
validate = False
launcher = 'pytorch'  # ['none', 'pytorch', 'slurm', 'mpi']
dist_params = dict(backend='nccl')
log_level = 'INFO'
seed = None
deterministic = False
resume_from = None
load_from = None  # '/media/tx-eva-20/data2/zh/workspace/pytorch_exp/cor_recon/cor_dl_centerline/example/other_pth/dir_04_1405.pth'
workflow = [('train', 1)]
