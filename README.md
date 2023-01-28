# GlycoNovo: glycan-denovo-sequencing

This is a deep learning-based model for glycan de novo sequencing from tandem mass spectrometry data. Two neural networks are designed to construct the glycan tree of an intact N-linked glycopeptide from root to leaves, starting from the peptide (root) and iteratively adding monosaccharides (leaves) to the tree. The first neural network captures the matched glycopeptide Y and B ions between the glycan tree and the spectrum. The second one, a graph neural network, captures the structure of the glycan tree.

The training, testing data and pretrained model are publicly available here: 
https://drive.google.com/drive/folders/1o84GYwIlAg06rJJTcgR1LXSRf6KNnvF1?usp=sharing

The training, prediction, and evaluation can be done by following the Python notebook `aa.glycan_manuscript.ipynb`.

To perform evaluation, please use the dataset `Demo_IgG_Orbitrap` and the pretrained model, and run the following sections in the notebook:
- `I/O FUNCTIONS and DATA PRE-PROCESSING`
- `MODEL PREPARATION`
- `MODEL DENOVO SEQUENCING`

To perform training, please use the dataset `mouse_brain` and run the following sections in the notebook:
- `I/O FUNCTIONS and DATA PRE-PROCESSING`
- `MODEL PREPARATION`
- `MODEL TRAINING`

More details and explanations are provided in the notebook. This is an on-going work, more data and materials will be further added. Please feel free to contact us if you have any questions.
