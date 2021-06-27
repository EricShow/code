trainner = dict(type="Trainner", runner_config=dict(type="EpochBasedRunner"))

win_level = 150
win_width = 800
patch_size = (128, 128, 128)
core_size = (64, 64, 64)
resize_size = (96, 96, 96)
padding = 10
seg_thresh = 0.3

model = dict(
    type="HeatmapNetwork",
    backbone=dict(type="ResUnet", in_ch=1, channels=32, blocks=3),
    apply_sync_batchnorm=True,  # 默认为False, True表示使用sync_batchnorm，只有分布式训练才可以使用
    head=dict(type="CoronaryHeatmapHead", in_channels=32, scale_factor=(2.0, 2.0, 2.0)),
    pipeline=[
        dict(
            type="Augmentation3d",
            aug_parameters={
                "rot_range_x": [-10, 10],
                "rot_range_y": [-10, 10],
                "rot_range_z": [-10, 10],
                "scale_range_x": (0.9, 1.1),
                "scale_range_y": (0.9, 1.1),
                "scale_range_z": (0.9, 1.1),
                "shift_range_x": (-0.1, 0.1),
                "shift_range_y": (-0.1, 0.1),
                "shift_range_z": (-0.1, 0.1),
                "elastic_alpha": [3.0, 3.0, 3.0],  # x,y,z
                "smooth_num": 4,
                "field_size": [17, 17, 17],  # x,y,z
                "out_style": "resize",
                "size_o": resize_size,
                "itp_mode_dict": {"img": "bilinear"},
            },
        )
    ],
)

train_cfg = None
test_cfg = None


data = dict(
    # imgs_per_gpu=64,
    imgs_per_gpu=2,
    workers_per_gpu=2,
    shuffle=True,
    drop_last=False,
    dataloader=dict(type="SampleDataLoader", source_batch_size=2, source_thread_count=1, source_prefetch_count=1),
    train=dict(
        type="Cor_Heatmap_ysy_Sample_Dataset",
        dst_list_file="/media/e/heart/coronary_seg_train_data/orgnized_data/fp-heatmap-data/train.lst",
        crop_size=patch_size,
        win_level=win_level,
        win_width=win_width,
        random_chose_center_prob=0.3,
        sample_frequent=30,
    ),
)


optimizer = dict(type="Adam", lr=1e-3, weight_decay=5e-4)
optimizer_config = {}

lr_config = dict(policy="step", warmup="linear", warmup_iters=1, warmup_ratio=1.0 / 3, step=[20, 50, 80, 90], gamma=0.2)

checkpoint_config = dict(interval=5)

log_config = dict(interval=5, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")])

cudnn_benchmark = False
work_dir = "./checkpoints/heatmap_ysy_0313"
gpus = 2
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
load_from = None
workflow = [("train", 1)]
