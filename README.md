# Code for evaluating BERT

This repository contains the code used for the experiments presented in the paper ["Is Multilingual BERT Fluent in Language Generation?"](https://www.aclweb.org/anthology/W19-6204/), and cosists of three tasks of increasing complexity and designed to test the capabilities of BERT with a focus on text generation. The tasks are:

* [Diagonistic classifier: main/non-main auxiliary prediction](https://github.com/TurkuNLP/bert-eval/tree/master/diagnostic_classifier)
* Cloze test word prediction
* Sentence generation given left and right-hand context

If you find this useful, please cite our paper as:
```
@inproceedings{ronnqvist-2019-bert-eval,
    title = "Is Multilingual BERT Fluent in Language Generation?",
    author = "R\"onnqvist, Samuel and Kanerva, Jenna and Salakoski, Tapio and Ginter, Filip",
    booktitle = "Proceedings of the First NLPL Workshop on Deep Learning for Natural Language Processing",
    month = oct,
    year = "2019",
    address = "Turku, Finland",
    publisher = "Association for Computational Linguistics"
}
```

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
