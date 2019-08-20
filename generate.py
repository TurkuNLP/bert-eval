

# Modified version of https://github.com/nyu-dl/bert-gen (https://arxiv.org/abs/1902.04094)


#!pip3 install pytorch_pretrained_bert
#!pip3 install pytorch_transformers

import numpy as np
import torch

from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM

import random
import math
import time



class BertGeneration(object):


    def __init__(self, model_directory, vocab_file, lower=False):
        

        # Load pre-trained model (weights)

        self.model = BertForMaskedLM.from_pretrained(model_directory)
        self.model.eval()
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.model = self.model.cuda()

        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=lower)

        self.CLS = '[CLS]'
        self.SEP = '[SEP]'
        self.MASK = '[MASK]'
        self.mask_id = self.tokenizer.convert_tokens_to_ids([self.MASK])[0]
        self.sep_id = self.tokenizer.convert_tokens_to_ids([self.SEP])[0]
        self.cls_id = self.tokenizer.convert_tokens_to_ids([self.CLS])[0]



    def tokenize_batch(self, batch):
        return [self.tokenizer.convert_tokens_to_ids(sent) for sent in batch]

    def untokenize_batch(self, batch):
        return [self.tokenizer.convert_ids_to_tokens(sent) for sent in batch]

    def detokenize(self, sent):
        """ Roughly detokenizes (mainly undoes wordpiece) """
        new_sent = []
        for i, tok in enumerate(sent):
            if tok.startswith("##"):
                new_sent[len(new_sent) - 1] = new_sent[len(new_sent) - 1] + tok[2:]
            else:
                new_sent.append(tok)
        return new_sent



    def generate_step(self, out, gen_idx, temperature=None, top_k=0, sample=False, return_list=True):
        """ Generate a word from from out[gen_idx]
        
        args:
            - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
            - gen_idx (int): location for which to generate for
            - top_k (int): if >0, only sample from the top k most probable words
            - sample (Bool): if True, sample from full distribution. Overridden by top_k 
        """
        logits = out[:, gen_idx]
        if temperature is not None:
            logits = logits / temperature
        if top_k > 0:
            kth_vals, kth_idx = logits.topk(top_k, dim=-1)
            dist = torch.distributions.categorical.Categorical(logits=kth_vals)
            idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
        elif sample:
            dist = torch.distributions.categorical.Categorical(logits=logits)
            idx = dist.sample().squeeze(-1)
        else:
            idx = torch.argmax(logits, dim=-1)
        return idx.tolist() if return_list else idx
      
  
    def get_init_text(self, seed_text, max_len, batch_size = 1, rand_init=False):
        """ Get initial sentence by padding seed_text with either masks or random words to max_len """
        batch = [seed_text + [self.MASK] * max_len + [self.SEP] for _ in range(batch_size)]
        #if rand_init:
        #    for ii in range(max_len):
        #        init_idx[seed_len+ii] = np.random.randint(0, len(tokenizer.vocab))
        
        return self.tokenize_batch(batch)

    def printer(self, sent, should_detokenize=True):
        if should_detokenize:
            sent = self.detokenize(sent)[1:-1]
        print(" ".join(sent))


    # This is the meat of the algorithm. The general idea is
    # 1. start from all masks
    # 2. repeatedly pick a location, mask the token at that location, and generate from the probability distribution given by BERT
    # 3. stop when converged or tired of waiting

    # We consider three "modes" of generating:
    # - generate a single token for a position chosen uniformly at random for a chosen number of time steps
    # - generate in sequential order (L->R), one token at a time
    # - generate for all positions at once for a chosen number of time steps

    # The `generate` function wraps and batches these three generation modes. In practice, we find that the first leads to the most fluent samples.

    # Generation modes as functions


    def parallel_sequential_generation(self, seed_text, batch_size=10, max_len=15, top_k=0, temperature=None, max_iter=300, burnin=200,
                                       cuda=False, print_every=10, verbose=True):
        """ Generate for one random position at a timestep
        
        args:
            - burnin: during burn-in period, sample from full distribution; afterwards take argmax
        """
        seed_len = len(seed_text)
        batch = self.get_init_text(seed_text, max_len, batch_size)
        
        for ii in range(max_iter):
            kk = np.random.randint(0, max_len)
            for jj in range(batch_size):
                batch[jj][seed_len+kk] = self.mask_id
            inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
            out = self.model(inp)[0]
            topk = top_k if (ii >= burnin) else 0
            idxs = self.generate_step(out, gen_idx=seed_len+kk, top_k=topk, temperature=temperature, sample=(ii < burnin))
            for jj in range(batch_size):
                batch[jj][seed_len+kk] = idxs[jj]
                
            if verbose and np.mod(ii+1, print_every) == 0:
                for_print = self.tokenizer.convert_ids_to_tokens(batch[0])
                for_print = for_print[:seed_len+kk+1] + ['(*)'] + for_print[seed_len+kk+1:]
                print("iter", ii+1, " ".join(for_print))
            
        return self.untokenize_batch(batch)

    def parallel_generation(self, seed_text, batch_size=10, max_len=15, top_k=0, temperature=None, max_iter=300, sample=True, 
                            cuda=False, print_every=10, verbose=True):
        """ Generate for all positions at each time step """
        seed_len = len(seed_text)
        batch = self.get_init_text(seed_text, max_len, batch_size)
        
        for ii in range(max_iter):
            inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
            out = self.model(inp)[0]
            for kk in range(max_len):
                idxs = self.generate_step(out, gen_idx=seed_len+kk, top_k=top_k, temperature=temperature, sample=sample)
                for jj in range(batch_size):
                    batch[jj][seed_len+kk] = idxs[jj]
                
            if verbose and np.mod(ii, print_every) == 0:
                print("iter", ii+1, " ".join(self.tokenizer.convert_ids_to_tokens(batch[0])))
        
        return self.untokenize_batch(batch)
            
    def sequential_generation(self, seed_text, batch_size=10, max_len=15, leed_out_len=15, 
                              top_k=0, temperature=None, sample=True, cuda=False):
        """ Generate one word at a time, in L->R order """
        seed_len = len(seed_text)
        batch = self.get_init_text(seed_text, max_len, batch_size)
        
        for ii in range(max_len):
            inp = [sent[:seed_len+ii+leed_out_len]+[self.sep_id] for sent in batch]
            inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
            out = self.model(inp)[0]
            idxs = self.generate_step(out, gen_idx=seed_len+ii, top_k=top_k, temperature=temperature, sample=sample)
            for jj in range(batch_size):
                batch[jj][seed_len+ii] = idxs[jj]
            
        return self.untokenize_batch(batch)


    def generate(self, n_samples, seed_text="[CLS]", batch_size=10, max_len=25, 
                 generation_mode="parallel-sequential",
                 sample=True, top_k=100, temperature=1.0, burnin=200, max_iter=500,
                 cuda=False, print_every=1, leed_out_len=15):
        # main generation function to call
        sentences = []
        n_batches = math.ceil(n_samples / batch_size)
        start_time = time.time()
        for batch_n in range(n_batches):
            if generation_mode == "parallel-sequential":
                batch = self.parallel_sequential_generation(seed_text, batch_size=batch_size, max_len=max_len, top_k=top_k,
                                                       temperature=temperature, burnin=burnin, max_iter=max_iter, 
                                                       cuda=cuda, verbose=False)
            elif generation_mode == "sequential":
                batch = self.sequential_generation(seed_text, batch_size=batch_size, max_len=max_len, top_k=top_k, 
                                              temperature=temperature, leed_out_len=leed_out_len, sample=sample,
                                              cuda=cuda)
            elif generation_mode == "parallel":
                batch = self.parallel_generation(seed_text, batch_size=batch_size,
                                            max_len=max_len, top_k=top_k, temperature=temperature, 
                                            sample=sample, max_iter=max_iter, 
                                            cuda=cuda, verbose=False)
            
            if (batch_n + 1) % print_every == 0:
                print("Finished batch %d in %.3fs" % (batch_n + 1, time.time() - start_time))
                start_time = time.time()
            
            sentences += batch
        return sentences


def main(args):
    #Let's call the actual generation function! We'll use the following settings
    #- max_len (40): length of sequence to generate
    #- top_k (100): at each step, sample from the top_k most likely words
    #- temperature (1.0): smoothing parameter for the next word distribution. Higher means more like uniform; lower means more peaky
    #- burnin (250): for non-sequential generation, for the first burnin steps, sample from the entire next word distribution, instead of top_k
    #- max_iter (500): number of iterations to run for
    #- seed_text (["CLS"]): prefix to generate for. We found it crucial to start with the CLS token; you can try adding to it 

    n_samples = 5
    batch_size = 5
    max_len = 30
    top_k = 100
    temperature = 1.0
    generation_mode = args.mode
    leed_out_len = 5 # max_len
    burnin = 250
    sample = True
    max_iter = 500

    model = BertGeneration(args.model_directory, args.vocab_file, args.lowercase)

    while True:

        user_seed = input("Seed for text generation: ")

        # Choose the prefix context 
        seed_text = ['[CLS]'] + model.tokenizer.tokenize(user_seed.strip())
        
        print(seed_text)
        bert_sents = model.generate(n_samples, seed_text=seed_text, batch_size=batch_size, max_len=max_len,
                              generation_mode=generation_mode,
                              sample=sample, top_k=top_k, temperature=temperature, burnin=burnin, max_iter=max_iter,
                              cuda=model.cuda, leed_out_len=leed_out_len)

        for sent in bert_sents:
          model.printer(sent, should_detokenize=True)


if __name__=="__main__":
    import argparse
    argparser = argparse.ArgumentParser(description='')
    argparser.add_argument('--model_directory', required=True, type=str, help='Directory with pytorch_model.bin and config.yaml')
    argparser.add_argument('--vocab_file', required=True, type=str, help='Name of the vocabulary file.')
    argparser.add_argument('--lowercase', default=False, action="store_true", help='Lowercase text (Default: False)')
    argparser.add_argument('--mode', default="parallel-sequential", choices=["parallel-sequential", "sequential", "parallel"], help='Generation mode')
    args = argparser.parse_args()

    main(args)


