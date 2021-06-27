patch_size = 19
intermedia_size = 25
win_level = 150
win_width = 800
isotropy_spacing = 0.3
data_pyramid_level = 3
data_pyramid_step = 1
direction_sphere_radius = 1.5
direction_sphere_sample_count = 1000

trainner = dict(type='Trainner', runner_config=dict(type='EpochBasedRunner'))
model = dict(
    type='CoronaryDirectionNetwork',
    #backbone=dict(type='SKNet', d_count=direction_sphere_sample_count),
    backbone=dict(type='ResNext3Da', d_count=direction_sphere_sample_count),
    # backbone=dict(type='ShuffleNet3D', width_mult=1.0, d_count=direction_sphere_sample_count),
    apply_sync_batchnorm=True,
    head=dict(
        type='CoronaryDirectionHead',
        # loss_dict=dict(type='Binary_Focal_Loss', gamma=1.0, alpha=0.5, reduction='elementwise_mean'),
    ),

    pipeline=[]

)

train_cfg = None
test_cfg = None

data = dict(
    imgs_per_gpu=32,
    workers_per_gpu=1,
    shuffle=True,
    drop_last=False,
    dataloader=dict(
        type='SampleDataLoader',
        source_batch_size=4,
        source_thread_count=1,
        source_prefetch_count=1,
    ),
    train=dict(
        type='CoronarySeg_direction_Sample_Dataset',
        dst_list_file='/media/tx-deepocean/Data/heart/358/train.lst',
        patch_size=patch_size,
        isotropy_spacing=isotropy_spacing,
        win_level=win_level,
        win_width=win_width,
        data_pyramid_level=data_pyramid_level,
        data_pyramid_step=data_pyramid_step,
        joint_sample_prob=0.4,
        translation_prob=0.3,
        pseudo_ratio=0.2,
        rotation_prob=1.0,
        rot_range=[360, 360, 360],
        mixup_prob=0.5,
        gaussian_noise_prob=0.0,
        direction_sphere_radius=direction_sphere_radius,
        direction_sphere_sample_count=direction_sphere_sample_count,
        sample_frequent=120,
    ),
)

optimizer = dict(type='Adam', lr=5e-4, weight_decay=5e-4)
optimizer_config = {}

lr_config = dict(policy='step', warmup='linear', warmup_iters=90, warmup_ratio=1.0 / 3, step=[300, 600], gamma=0.2)

checkpoint_config = dict(interval=30)

log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])

cudnn_benchmark = False
work_dir = './checkpoints/direction_pretrain/2021_05_08'
gpus = 3
find_unused_parameters = True
total_epochs = 900
autoscale_lr = None
validate = False
launcher = 'pytorch'  # ['none', 'pytorch', 'slurm', 'mpi']
dist_params = dict(backend='nccl')
log_level = 'INFO'
seed = None
deterministic = False
resume_from = None # '/media/tx-deepocean/Data/code/cor_dl_centerline/train/checkpoints/direction_1.x/2021_04_09/epoch_30.pth'
load_from = '/media/tx-deepocean/Data/code/direction_sdl/2021_03_19/epoch_540.pth'  # '/media/tx-eva-20/data2/zh/workspace/pytorch_exp/cor_recon/cor_dl_centerline/example/other_pth/dir_04_1405.pth'
workflow = [('train', 1)]
