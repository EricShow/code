patch_size = (128, 128, 128)
intermedia_size = 25
win_level = 150
win_width = 800
constant_shift = 10

trainner = dict(type="Trainner", runner_config=dict(type="EpochBasedRunner"))
model = dict(
    type="CorPlaqueNetwork",
    backbone=dict(type="ResUnet", in_ch=2, channels=32, blocks=3),
    apply_sync_batchnorm=False,
    head=dict(type="CoronaryPlaqueHead", in_channels=32, scale_factor=(2.0, 2.0, 2.0),),
    pipeline=[
        dict(
            type="Augmentation3d",
            aug_parameters={
                "rot_range_x": (-20.0, 20.0),
                "rot_range_y": (-20.0, 20.0),
                "rot_range_z": (-20.0, 20.0),
                "scale_range_x": (1.0, 1.0),
                "scale_range_y": (1.0, 1.0),
                "scale_range_z": (1.0, 1.0),
                "shift_range_x": (-0.0, 0.0),
                "shift_range_y": (-0.0, 0.0),
                "shift_range_z": (-0.0, 0.0),
                "elastic_alpha": [3.0, 3.0, 3.0],  # x,y,z
                "smooth_num": 4,
                "field_size": [17, 17, 17],  # x,y,z
                "size_o": patch_size,
                "itp_mode_dict": {"mask": "bilinear"},
                "out_style": "resize",
            },
        )
    ],
)

train_cfg = None
test_cfg = None

data = dict(
    imgs_per_gpu=5,
    workers_per_gpu=1,
    shuffle=True,
    drop_last=False,
    dataloader=dict(type="SampleDataLoader", source_batch_size=3, source_thread_count=1, source_prefetch_count=1,),
    train=dict(
        type="CorPlaque_Sample_Dataset",
        dst_list_file="/media/tx-eva-20/ssd/data/cor/train/plaque/train_experti.lst",
        patch_size=patch_size,
        win_level=win_level,
        win_width=win_width,
        constant_shift=constant_shift,
        shift_range=10,
        sample_frequent=10,
    ),
)

optimizer = dict(type="Adam", lr=5e-4, weight_decay=5e-4)
optimizer_config = {}

lr_config = dict(policy="step", warmup="linear", warmup_iters=30, warmup_ratio=1.0 / 3, step=[25, 75], gamma=0.2)

checkpoint_config = dict(interval=1)

log_config = dict(interval=5, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")])

cudnn_benchmark = False
work_dir = "./checkpoints/plaque_1.0/2021_03_03"
gpus = 1
find_unused_parameters = True
total_epochs = 100
autoscale_lr = None
validate = False
launcher = "pytorch"  # ['none', 'pytorch', 'slurm', 'mpi']
dist_params = dict(backend="nccl")
log_level = "INFO"
seed = None
deterministic = False
resume_from = None
load_from = None  # '/media/tx-eva-20/data2/zh/workspace/pytorch_exp/cor_recon/cor_dl_centerline/example/other_pth/dir_04_1405.pth'
workflow = [("train", 1)]
