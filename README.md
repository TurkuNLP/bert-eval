# bert-eval


## Requirements

`pip3 install pytorch-transformers`


## Generation (toy example)

Generate text from a given seed.

Usage: `python3 generate.py --model /path/to/mybert-model --vocab /path/to/mybert-model/mybert-model.vocab.wp`

Where `mybert-model` is a directory containing `pytorch_model.bin` and `config.json`. Use `--mask_len` to define how many subwords to generate (default is 30).


## Cloze test

We mask a random 15\% of words in each sentence and try to predict them back. In case a word is composed of several subwords, all subwords are masked. Accuracy is measured on subword level, i.e.\ how many times the model gives the highest confidence score for the original subword.

Usage: `python3 cloze.py --model /path/to/mybert-model --vocab /path/to/mybert-model/mybert-model.vocab.wp --test_sentences sentences.txt --max_sent 1000`

Run `python3 cloze.py -h` for more options.

#### Finnish data

Finnish evaluation data can be downloaded with `./get_finnish_data.sh`, this greps UD_Finnish-TDT training sentences, and saves them under a file name `finnish_sentences.txt`.
