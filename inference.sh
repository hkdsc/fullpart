python inference.py \
	--stage1.transformer-ckpt pretrained_models/s1/pytorch_model.ckpt \
	--stage2.transformer-ckpt pretrained_models/s2/pytorch_model.ckpt \
	--raw-path assets/demo_examples/toy_gun \
	--raw-sample-id toy_gun_aaa \
	--output-dir ./debug/infer_debug