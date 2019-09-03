import sys
import numpy as np

MODEL = sys.argv[1]
PATH = sys.argv[2] #'data/UD_Finnish-TDT/fi_tdt-ud-train.conllu'
OUT_PREFIX = sys.argv[3]
if len(sys.argv) >= 5:
    LIMIT = int(sys.argv[4])
else:
    LIMIT = 1000

#MODEL = 'bert-base-german-cased'
#MODEL = 'bert-base-cased'
#MODEL = 'bert-base-multilingual-cased'

### Load m-BERT

from pytorch_transformers import BertModel, BertTokenizer
import torch


bert = BertModel.from_pretrained(MODEL)
tokenizer = BertTokenizer.from_pretrained(MODEL)


### Read CoNLL-U
print("---", file=sys.stderr)
print("MODEL:", MODEL, file=sys.stderr)
print("FILE:", PATH, file=sys.stderr)

ID,FORM,LEMMA,UPOS,_,FEATS,HEAD,DEPREL,DEPS,MISC=range(10)
def get_sentences(f, limit=None):
    """conllu reader"""
    sent=[]
    comment=[]
    sent_cnt = 0
    for line in f:
        line=line.strip()
        if not line: # new sentence
            if sent:
                yield comment,sent
                sent_cnt += 1
                if limit and sent_cnt >= limit:
                    return
            comment=[]
            sent=[]
        elif line.startswith("#"):
            comment.append(line)
        else: #normal line
            sent.append(line.split("\t"))
    else:
        if sent:
            yield comment, sent


cntr0, cntr1 = 0,0

examples = []
for comment, sent in get_sentences(open(PATH), limit=None):
    tokens = {}
    root = None
    for token in sent:
        tokens[token[ID]] = token
        if token[DEPREL] == 'root':
            root = token[ID]
    surface = ' '.join([t[FORM] for t in tokens.values()])#.replace(':n', ' : n')#.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?').replace(' ;', ';')
    for token in tokens.values():
        if token[DEPREL] in ['cop', 'aux']:
            if token[HEAD] == root:
                examples.append({'class': 1, 'sent': surface, 'token_idx': int(token[ID]), 'token': token[FORM]})
                cntr1 += 1
            else:
                examples.append({'class': 0, 'sent': surface, 'token_idx': int(token[ID]), 'token': token[FORM]})
                cntr0 += 1
    if len(examples) >= LIMIT:
        break

print("Class counts:", file=sys.stderr)
print("0:", cntr0, file=sys.stderr)
print("1:", cntr1, file=sys.stderr)
print("0+1:", cntr1+cntr0, file=sys.stderr)
print()

data = []
labels = []

skips = 0
for ex in examples:
    input_ids = torch.tensor([tokenizer.encode(ex['sent'])])
    with torch.no_grad():
        try:
            last_hidden_states = bert(input_ids[:511])[0]  # Models outputs are now tuples
        except RuntimeError:
            continue
    pieces = [tokenizer.ids_to_tokens[int(x)] for x in input_ids[0]]
    ex['pieces'] = pieces
    word_nr = 0
    matches = []
    # Look for ex['token_idx'] in piece sequence
    for piece_idx, piece in enumerate(pieces):
        if not piece.startswith('##'):
            word_nr += 1
        if word_nr == ex['token_idx']:
            break

    print("Searching:", pieces[piece_idx:])
    in_token = False
    for pi, p in enumerate(pieces[piece_idx:]):
        if in_token:
            matches.append({'piece_idx': piece_idx+pi, 'piece': p})
            if ex['token'].endswith(p.strip('#')) or pi > 6:
                break
        elif ex['token'].startswith(p):
            matches.append({'piece_idx': piece_idx+pi, 'piece': p})
            if ex['token'] == p:
                break
            else:
                in_token = True

    for k,v in ex.items():
        if k=='pieces':
            v = ' '.join(v)
        print("%s:" % k,v)

    for match in matches:
        print(match)
        data.append(last_hidden_states[0][match['piece_idx']])
        labels.append(ex['class'])

    if not matches:
        print("No match.")
        skips += 1
    print()


print("Skips:", skips, file=sys.stderr)
print("Input-output pairs:", len(data), file=sys.stderr)
print("Class 1:", sum(labels), '/', len(labels), "(%.2f%%)" % (sum(labels)/len(labels)*100), file=sys.stderr)

from keras.utils import np_utils
data = np.array([np.array(x) for x in data])
np.save(OUT_PREFIX+"_data.npy", data)
np.save(OUT_PREFIX+"_labels.npy", np_utils.to_categorical(labels))
