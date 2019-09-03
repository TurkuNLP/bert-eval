# Diagnostic Classifier Experiments


## Requirements

Download treebanks: `sh dl_data.sh`

The languages used in the experiments are: Danish, English, Finnish, German, Norwegian (Bokmaal and Nynorsk), and Swedish.


## Main/Non-Main Auxiliary Classification

Prepare train, dev and test sets for all language and BERT version (m-BERT, BERT-en, BERT-de) pairs: 
`sh prepare_all.sh`

Train and evaluate classifier for all pairs: `sh classify_all.sh`

Experiment results are written to `exp_classification.log`.
