# GlycoNovo: glycan-denovo-sequencing

This is a database-independent method for determining the structure of glycans, composed of two stages. In the first stage, GlycoNovo derives the compositions of glycans from mass spectra, using a dynamic programming method. In the second stage, it constructs the topology of the glycan based on its composition. At each iteration, the model attempts to extract information about the next monosaccharide, starting from the root(i.e. asparagine (Asn) residue of the peptide). We believe that the structure of glycans follows a certain pattern. GlycoNovo uses a novel deep learning based graph encoding structure called Graphormer to learn such patterns and extract features from the differences between observed peaks and theoretical m/z values obtained from the learned distribution.

If you want to train the model with respect to glcan structure
```
python evaluate/train.py --num_epoch=20 --pre_layernorm --encoder_normalize_before --train --batch_size=256 --graph_model=../../examples/property_prediction/ckpts/model_pos_node_stop.pt
```


If you want to train the model with respect to spectrum
```
python evaluate/train.py --num_epoch=20 --pre_layernorm --encoder_normalize_before --train_cnn --batch_size=256 --csv_file=../../../Graphormer/data/mouse_tissues.csv --graph_model=../../examples/property_prediction/ckpts/model_pos_node_stop.pt --cnn_model=../../examples/property_prediction/ckpts/mouse_tissue_all.pt
```


If you want to evaluate the framework
Please note that if you don't want to generate composition through composition denovo, you have to include your csv with a column named "Glycan" that includes composition information before building structure through neural network
composition denovo can be done by
```
python composition_denovo.py --mgf_file=MouseKidney-Z-T-1.refined.mgf --csv_file=glycanfinder.glycopsms.MouseKidney-Z-T-1.csv --output_file=mouse_kidney_pred_comp1.csv

```
build structure
```
python evaluate/train.py --num_epoch=20 --pre_layernorm --encoder_normalize_before --inference_cnn --batch_size=256 --csv_file=comp_denovo_strucgp_mouse_kidney1
```
prediction
```
python evaluate/train.py --num_epoch=20 --pre_layernorm --encoder_normalize_before --prediction --batch_size=256 --csv_file=comp_denovo_strucgp_mouse_kidney1 --graph_model=../../examples/property_prediction/ckpts/model_pos_node_stop.pt --cnn_model=../../examples/property_prediction/ckpts/mouse_tissue_all.pt
```
More details and explanations are provided in the notebook. This is an on-going work, more data and materials will be further added. Please feel free to contact us if you have any questions.
