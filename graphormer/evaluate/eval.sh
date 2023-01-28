CUDA_VISIBLE_DEVICES=0 python train.py --num_epoch=20 --pre_layernorm --encoder_normalize_before --inference_cnn --batch_size=256 --csv_file=pred_comp_mouse_heart1 > test_heart_all.log
CUDA_VISIBLE_DEVICES=0 python train.py --num_epoch=20 --pre_layernorm --encoder_normalize_before --inference_cnn --batch_size=256 --csv_file=comp_denovo_overlap_mouse_kidney1 > test_kidney_overlap.log

