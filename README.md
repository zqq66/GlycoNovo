# GlycoNovo: glycan-denovo-sequencing

This is a database-independent method for determining the structure of glycans, composed of two stages. In the first stage, GlycoNovo derives the compositions of glycans from mass spectra, using a dynamic programming method. In the second stage, it constructs the topology of the glycan based on its composition. At each iteration, the model attempts to extract information about the next monosaccharide, starting from the root(i.e. asparagine (Asn) residue of the peptide). We believe that the structure of glycans follows a certain pattern. GlycoNovo uses a novel deep learning based graph encoding structure called Graphormer to learn such patterns and extract features from the differences between observed peaks and theoretical m/z values obtained from the learned distribution.

All Data and checkpoints can be found here https://drive.google.com/drive/folders/1XbAtWeh4HoHD1l3bMeyMOxM_vB15IFIi?usp=sharing

As mentioned in the paper, candidates will be automatically generated during training process.
If you want to inference the framework without training, please make sure to include "all_entries.pkl" in criterions directory

After cloning the repository
1. redirect to the entry of the whole program

```
cd graphormer/evaluate
```

2. Install dependency (make sure python>=3.9 installed)
```
bash install.sh
```

Train the model with respect to glycan structure
```
python train.py --num_epoch=20 --pre_layernorm --encoder_normalize_before --train --batch_size=256 --graph_model=../../examples/property_prediction/ckpts/model_pos_node_stop.pt
```


Train the model with respect to spectrum
```
python train.py --num_epoch=10 --pre_layernorm --encoder_normalize_before --train_cnn --batch_size=256 --csv_file=../../../Graphormer/data/mouse_tissues.csv --graph_model=../../examples/property_prediction/ckpts/model_pos_node_stop.pt --cnn_model=../../examples/property_prediction/ckpts/mouse_tissue_all.pt --mgf_file=../../../Graphormer/data/mouse_tissues_spectrum.mgf
```


Evaluate the framework

Please note that if you don't want to generate composition through composition denovo, you have to include your csv with a column named "Glycan" that includes composition information before building structure through neural network
composition denovo can be done by
```
cd ..
python composition_denovo.py --mgf_file=MouseKidney-Z-T-1.refined.mgf --csv_file=../Graphormer/data/glycanfinder.glycopsms.MouseKidney-Z-T-1.csv --output_file=mouse_kidney_pred_comp1.csv

```
build structure

If you want to test with your own data, please remember to replace glycan_db with path to the glycan database used for your glycan database search.

```
python train.py --pre_layernorm --encoder_normalize_before --inference_cnn --batch_size=256 --csv_file=../../../Graphormer/data/IgGz.csv --mgf_file=../../../Graphormer/data/mouse_tissues_spectrum.mgf --graph_model=../../examples/property_prediction/ckpts/model_pos_node_stop.pt --cnn_model=../../examples/property_prediction/ckpts/mouse_tissue_all.pt --glycan_db=../../../Graphormer/data/glycan_database/glycans-v2.pkl
```
prediction
```
python train.py --pre_layernorm --encoder_normalize_before --prediction --batch_size=256 --csv_file=../../../Graphormer/data/comp_denovo_strucgp_mouse_kidney1.csv --mgf_file=../../../Graphormer/data/mouse_tissues_spectrum.mgf --graph_model=../../examples/property_prediction/ckpts/model_pos_node_stop.pt --cnn_model=../../examples/property_prediction/ckpts/mouse_tissue_all.pt
```

To compare with strucgp results, where input.csv should be replaced by strucgp results, where output.csv will include an extra column of GlycoCT format
```
python3 convert_glycan_format.py input.csv output.csv
```

More details and explanations are provided in the scripts. This is an on-going work, more data and materials will be further added. Please feel free to contact us if you have any questions.
