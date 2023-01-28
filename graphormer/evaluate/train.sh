CUDA_VISIBLE_DEVICES=1 python train.py --num_epoch=20 --pre_layernorm --encoder_normalize_before --inference_cnn --batch_size=256 --csv_file=comp_denovo_overlap_mouse_brain1 > test_brain_overlap.log
CUDA_VISIBLE_DEVICES=1 python train.py --num_epoch=20 --pre_layernorm --encoder_normalize_before --inference_cnn --batch_size=256 --csv_file=pred_comp_mouse_brain1 > test_brain_all.log
