python inference.py \
	--stage1.transformer-ckpt pretrained_models/s1/step_384000.ckpt \
	--stage2.transformer-ckpt pretrained_models/s2/step_96000.ckpt \
	--raw-image assets/demo_examples/gundam.png \
	--raw-box sample_results/gundam1_part/eval_box.npy \
	--raw-sample_id gundam1_part_aaa \
	--output-dir ./debug/infer_debug