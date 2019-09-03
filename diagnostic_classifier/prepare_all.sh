MODEL_DE='bert-base-german-cased'
MODEL_EN_U='bert-base-uncased'
MODEL_EN_C='bert-base-cased'
MODEL_ALL='bert-base-multilingual-cased'


python prepare_classification.py $MODEL_ALL data/UD_English-EWT/en_ewt-ud-train.conllu data/en_mbert_train
python prepare_classification.py $MODEL_ALL data/UD_English-EWT/en_ewt-ud-dev.conllu data/en_mbert_dev
python prepare_classification.py $MODEL_ALL data/UD_English-EWT/en_ewt-ud-test.conllu data/en_mbert_test

python prepare_classification.py $MODEL_EN_U data/UD_English-EWT/en_ewt-ud-train.conllu data/en_uncased_train
python prepare_classification.py $MODEL_EN_U data/UD_English-EWT/en_ewt-ud-dev.conllu data/en_uncased_dev
python prepare_classification.py $MODEL_EN_U data/UD_English-EWT/en_ewt-ud-test.conllu data/en_uncased_test

#python prepare_classification.py $MODEL_EN_C data/UD_English-EWT/en_ewt-ud-train.conllu data/en_cased_train
#python prepare_classification.py $MODEL_EN_C data/UD_English-EWT/en_ewt-ud-dev.conllu data/en_cased_dev
#python prepare_classification.py $MODEL_EN_C data/UD_English-EWT/en_ewt-ud-test.conllu data/en_cased_test

python prepare_classification.py $MODEL_ALL data/UD_German-HDT/de_hdt-ud-train-a.conllu data/de_mbert_train
python prepare_classification.py $MODEL_ALL data/UD_German-HDT/de_hdt-ud-dev.conllu data/de_mbert_dev
python prepare_classification.py $MODEL_ALL data/UD_German-HDT/de_hdt-ud-test.conllu data/de_mbert_test

python prepare_classification.py $MODEL_DE data/UD_German-HDT/de_hdt-ud-train-a.conllu data/de_train
python prepare_classification.py $MODEL_DE data/UD_German-HDT/de_hdt-ud-dev.conllu data/de_dev
python prepare_classification.py $MODEL_DE data/UD_German-HDT/de_hdt-ud-test.conllu data/de_test

python prepare_classification.py $MODEL_ALL data/UD_Danish-DDT/da_ddt-ud-train.conllu data/da_train
python prepare_classification.py $MODEL_ALL data/UD_Danish-DDT/da_ddt-ud-dev.conllu data/da_dev
python prepare_classification.py $MODEL_ALL data/UD_Danish-DDT/da_ddt-ud-test.conllu data/da_test

python prepare_classification.py $MODEL_ALL data/UD_Finnish-TDT/fi_tdt-ud-train.conllu data/fi_train
python prepare_classification.py $MODEL_ALL data/UD_Finnish-TDT/fi_tdt-ud-dev.conllu data/fi_dev
python prepare_classification.py $MODEL_ALL data/UD_Finnish-TDT/fi_tdt-ud-test.conllu data/fi_test

python prepare_classification.py $MODEL_ALL data/UD_Norwegian-Bokmaal/no_bokmaal-ud-train.conllu data/nb_train
python prepare_classification.py $MODEL_ALL data/UD_Norwegian-Bokmaal/no_bokmaal-ud-dev.conllu data/nb_dev
python prepare_classification.py $MODEL_ALL data/UD_Norwegian-Bokmaal/no_bokmaal-ud-test.conllu data/nb_test

python prepare_classification.py $MODEL_ALL data/UD_Norwegian-Nynorsk/no_nynorsk-ud-train.conllu data/nn_train
python prepare_classification.py $MODEL_ALL data/UD_Norwegian-Nynorsk/no_nynorsk-ud-dev.conllu data/nn_dev
python prepare_classification.py $MODEL_ALL data/UD_Norwegian-Nynorsk/no_nynorsk-ud-test.conllu data/nn_test

python prepare_classification.py $MODEL_ALL data/UD_Swedish-Talbanken/sv_talbanken-ud-train.conllu data/sv_train
python prepare_classification.py $MODEL_ALL data/UD_Swedish-Talbanken/sv_talbanken-ud-dev.conllu data/sv_dev
python prepare_classification.py $MODEL_ALL data/UD_Swedish-Talbanken/sv_talbanken-ud-test.conllu data/sv_test
