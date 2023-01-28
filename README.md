# GlycoNovo: glycan-denovo-sequencing

This is a database-independent method for determining the structure of glycans, composed of two stages. In the first stage, GlycoNovo derives the compositions of glycans from mass spectra, using a dynamic programming method. In the second stage, it constructs the topology of the glycan based on its composition. At each iteration, the model attempts to extract information about the next monosaccharide, starting from the root(i.e. asparagine (Asn) residue of the peptide). We believe that the structure of glycans follows a certain pattern. GlycoNovo uses a novel deep learning based graph encoding structure called Graphormer to learn such patterns and extract features from the differences between observed peaks and theoretical m/z values obtained from the learned distribution.
If you want to train the model
```
python evaluate/train.py --num_epoch=20 --pre_layernorm --encoder_normalize_before --inference_cnn --batch_size=256 --csv_file=pred_comp_mouse_heart1 > test_heart_all.log
```

composition denovo can be done by
```
python composition_denovo.py --mgf_file=MouseKidney-Z-T-1.refined.mgf --csv_file=glycanfinder.glycopsms.MouseKidney-Z-T-1.csv --output_file=mouse_kidney_pred_comp1.csv

```
Inference
```
python evaluate/train.py --num_epoch=20 --pre_layernorm --encoder_normalize_before --inference_cnn --batch_size=256 --csv_file=pred_comp_mouse_heart1 > test_heart_all.log
```

To perform evaluation, please use the dataset `Demo_IgG_Orbitrap` and the pretrained model, and run the following sections in the notebook:
- `I/O FUNCTIONS and DATA PRE-PROCESSING`
- `MODEL PREPARATION`
- `MODEL DENOVO SEQUENCING`

To perform training, please use the dataset `mouse_brain` and run the following sections in the notebook:
- `I/O FUNCTIONS and DATA PRE-PROCESSING`
- `MODEL PREPARATION`
- `MODEL TRAINING`

More details and explanations are provided in the notebook. This is an on-going work, more data and materials will be further added. Please feel free to contact us if you have any questions.
