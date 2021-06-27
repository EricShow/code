# Getting Started
Follow the instructions to run this example

## Prerequisites
- python3.7
- virtualenv

## Installation

In this folder:

    $ virtualenv venv -p python3.7 && source venv/bin/activate && python3.7 -m pip install -r requirements.txt


## 运行demo

    $ USER_NAME=${USER} make

## 网页化查看训练过程

    $ make tensorboard

## 格式化代码

    $ make format

## 生成输入模型的数据

    $ USER_NAME=${USER} make generate_data


## 文件结构说明

```
├── train
│    └── config
│        ├── coronary_seg_config.py
│    └── custom
│        ├── datastet
│            ├── dataset.py
│            └── __init__.py
│        ├── model
│            ├── coronaryseg_head.py
│            └── coronaryseg_network.py
│            └── __init__.py
│        ├── utils
│            ├── generate_dataset.py
│            ├── save_torchscript.py
│            └── __init__.py
│        ├── __init__.py
│    └── train.py
│    └── run_dist.sh
│    └── requirements.txt
│    └── Makefile
│    └── README.md
├── README.md
```


- train.py: 训练代码入口，需要注意的是，在train.py里import custom，训练相关需要注册的模块可以直接放入到custom文件夹下面，会自动进行注册; 一般来说，训练相关的代码务必放入到custom文件夹下面!!!<br>

- ./custom/dataset/dataset.py: dataset类，需要@DATASETS.register_module进行注册方可被识别<br>

- ./custom/model/coronaryseg_network.py: 模型head文件，需要@HEADS.register_module进行注册方可被识别<br>

- ./custom/model/coronaryseg_head.py: 整个网络部分，训练阶段构建模型，forward方法输出loss的dict, 通过@NETWORKS.register_module进行注册

- ./custom/utils/generate_dataset.py: 从原始数据生成输入到模型的数据，供custom/dataset/dataset.py使用

- ./custom/utils/save_torchscript.py: 生成模型对应的静态图(根据CoronarySeg_Network中的forward_test函数)

- ./config/coronary_seg_config.py: 训练的配置文件

- run_dist.sh: 分布式训练的运行脚本
