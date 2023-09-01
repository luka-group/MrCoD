CUDA_VISIBLE_DEVICES=0,2,4,7 python -u -m torch.distributed.launch --nproc_per_node=4 --master_port 20000 \
	codred-blend.py \
	--model bert-base-cased \
	--load_model_path r/4/output/checkpoint-176491/pytorch_model.bin \
	--seq_len 512 \
	--dev \
	--test \
	--per_gpu_train_batch_size 1 \
	--per_gpu_eval_batch_size 1 \
	--learning_rate 3e-5 \
	--epochs 1 \
	--num_workers 8 \
	--logging_step 10 \
	--dev_file ../data/open_setting_data/dev_data_shared_entities_ranked.json \
	--test_file ../data/open_setting_data/test_data_shared_entities_ranked.json \
    --index_file ../data/retrieval_index/path_mining_3_hop_finetune_ext_inference_open_shared_entities.json
	#--train \
