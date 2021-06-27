DATA_URL=
MODEL_URL=

TRAIN_ROOT=train_data
MODEL_ROOT=checkpoints

ORI_DATA_ROOT=$(TRAIN_ROOT)/origin
PROCESSED_DATA_ROOT=$(TRAIN_ROOT)/processed_data

ORI_DATA_URL=192.168.108.54:/volume1/数据使用/模型训练数据集/冠脉产品线/冠脉分割/cor_seg4.x/4.0/train_data/origin/
PROCESSED_DATA_URL=192.168.108.54:/volume1/数据使用/模型训练数据集/冠脉产品线/冠脉分割/cor_seg4.x/4.0/train_data/processed_data/

# make USER_NAME=..
run: 
	bash run_dist.sh -d 1 -g 1 -c ./config/coronary_reduceFP_config.py

format:
	pre-commit run --all-files --show-diff-on-failure

tensorboard:
	tensorboard serve --logdir checkpoints/ --host '0.0.0.0'

download: download_ori
	test $(USER_NAME)
	if [ ! -d "$(PROCESSED_DATA_ROOT)" ]; then \
		echo "Downloading processed data..."; \
		rsync -avP $(USER_NAME)@$(PROCESSED_DATA_URL) $(PROCESSED_DATA_ROOT); \
	fi

download_ori:
	test $(USER_NAME)
	if [ ! -d "$(ORI_DATA_ROOT)" ]; then \
		mkdir -p $(ORI_DATA_ROOT); \
		echo "Downloading original data..."; \
		rsync -avP $(USER_NAME)@$(ORI_DATA_URL) $(ORI_DATA_ROOT); \
	fi

generate_data: download_ori
	python3.7 custom/utils/generate_dataset.py --src_path $(ORI_DATA_ROOT)/data --tgt_path $(PROCESSED_DATA_ROOT)/data --save_lst $(PROCESSED_DATA_ROOT)/train.lst
	python3.7 custom/utils/generate_plaque.py --size_range 1.0 3.0 --tgt_path $(PROCESSED_DATA_ROOT)/plaque/small
	python3.7 custom/utils/generate_plaque.py --size_range 3.0 6.0 --tgt_path $(PROCESSED_DATA_ROOT)/plaque/large



clean:
	rm -rf $(MODEL_ROOT)
	rm -rf $(PROCESSED_DATA_ROOT)
