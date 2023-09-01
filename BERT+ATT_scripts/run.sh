CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u -m torch.distributed.launch --nproc_per_node=4 --master_port 20003 \
	codred-blend.py \
	--model bert-base-cased \
	--seq_len 512 \
	--train \
	--dev \
	--test \
	--per_gpu_train_batch_size 1 \
	--per_gpu_eval_batch_size 1 \
	--learning_rate 3e-5 \
	--epochs 8 \
	--num_workers 8 \
	--logging_step 10 \
	--bridge_only \
	--add_special \
    --index_file ../data/retrieval_index/path_mining_3_hop_finetune_ext_inference_open_shared_entities.json  >> run_path_mining_3_hop_finetune_ext_inference_add_special_bridge &
    #--index_file ../data/retrieval_index/path_mining_3_hop_bm25_open_shared_entities.json  >> run_path_mining_3_hop_bm25_add_special_bridge &
	#--dev_file ../data/open_setting_data/dev_data_shared_entities_ranked.json \
	#--test_file ../data/open_setting_data/test_data_shared_entities_ranked.json \
