CUDA_VISIBLE_DEVICES=2,3,4,5 python -u -m torch.distributed.launch --nproc_per_node=4 --master_port 20002\
	codred-blend.py \
	--model bert-base-cased \
	--seq_len 512 \
	--dev \
	--test \
	--per_gpu_train_batch_size 1 \
	--per_gpu_eval_batch_size 1 \
	--learning_rate 3e-5 \
	--epochs 1 \
	--num_workers 8 \
	--logging_step 10 \
    --index_file ../data/retrieval_index/path_mining_3_hop_random_open_shared_entities.json
	#--train \
	#--dev_file ../data/open_setting_data/dev_data_shared_entities_ranked.json \
	#--test_file ../data/open_setting_data/test_data_shared_entities_ranked.json \
