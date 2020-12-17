# -*- coding: utf-8 -*-
# Python version: 3
#
# SFU CMPT413/825 Fall 2019, HW4
# default solution
# Simon Fraser University
# Jetic Gū
#
#
from queue import PriorityQueue
import os
import re
import sys
import optparse
from tqdm import tqdm

import torch
from torch import nn

import pandas as pd
from torchtext import data
import sacrebleu
import random

#import support.hyperparams as hp
#import support.datasets as ds

# hyperparameters


class hp:
    # vocab
    pad_idx = 1
    sos_idx = 2

    # architecture
    hidden_dim = 256
    embed_dim = 256
    n_layers = 2
    dropout = 0.2
    batch_size = 32
    num_epochs = 10
    lexicon_cap = 25000

    # training
    max_lr = 1e-4
    cycle_length = 3000

    # generation
    max_len = 50

    # system
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---YOUR ASSIGNMENT---
# -- Step 1: Baseline ---
# The attention module is completely broken now. Fix it using the definition
# given in the HW description.


class AttentionModule(nn.Module):
    def __init__(self, attention_dim):
        """
        You shouldn't deleted/change any of the following defs, they are
        essential for successfully loading the saved model.
        """
        super(AttentionModule, self).__init__()
        self.W_enc = nn.Linear(attention_dim, attention_dim,
                               bias=False)  # .to(hp.device)
        self.W_dec = nn.Linear(attention_dim, attention_dim,
                               bias=False)  # .to(hp.device)
        self.V_att = nn.Linear(attention_dim, 1, bias=False)  # .to(hp.device)
        return

    # Start working from here, both 'calcAlpha' and 'forward' need to be fixed
    def calcAlpha(self, decoder_hidden, encoder_out):
        """
        param encoder_out: (seq, batch, dim),
        param decoder_hidden: (seq, batch, dim)
        """
        seq, batch, dim = encoder_out.shape
        scores = torch.tanh(self.W_dec(decoder_hidden)+self.W_enc(encoder_out)).permute(1, 0, 2).to(hp.device)

        alpha = torch.nn.functional.softmax(self.V_att(scores), dim=1)  # .to(hp.device)

        return alpha

    def forward(self, decoder_hidden, encoder_out):
        """
        encoder_out: (seq, batch, dim),
        decoder_hidden: (seq, batch, dim)
        """
        alpha = self.calcAlpha(decoder_hidden,encoder_out).to(hp.device)

        seq, _, dim = encoder_out.shape  # 7,1,256
        context = (torch.sum(alpha*encoder_out.permute(1, 0, 2),dim=1)).reshape(1, 1, dim)

        return context, alpha.permute(2, 0, 1)


# -- Step 2: Improvements ---
# Implement UNK replacement, BeamSearch, translation termination criteria here,
# you can change 'greedyDecoder' and 'translate'.
def greedyDecoder(decoder, encoder_out, encoder_hidden, maxLen,
                  eos_index):
    seq1_len, batch_size, _ = encoder_out.size()
    target_vocab_size = decoder.target_vocab_size

    outputs = torch.autograd.Variable(
        encoder_out.data.new(maxLen, batch_size, target_vocab_size))
    alphas = torch.zeros(maxLen, batch_size, seq1_len)
    # take what we need from encoder
    decoder_hidden = encoder_hidden[-decoder.n_layers:]
    # start token (ugly hack)
    output = torch.autograd.Variable(
        outputs.data.new(1, batch_size).fill_(eos_index).long())
    for t in range(maxLen):
        output, decoder_hidden, alpha = decoder(
            output, encoder_out, decoder_hidden)
        outputs[t] = output
        alphas[t] = alpha.data
        output = torch.autograd.Variable(output.data.max(dim=2)[1])
        if int(output.data) == eos_index:
            break

    return outputs, alphas.permute(1, 2, 0)

def translate(model, test_iter):
    results = []
    for i, batch in tqdm(enumerate(test_iter)):
        output, attention = model(batch.src)
        output = output.topk(1)[1]
        output = model.tgt2txt(output[:, 0].data).strip().split('<EOS>')[0]

        '''remove repeated consecutive translated texts'''
        output = remove_repeated_word(output)
        ''' replace <unk> with the most frequent word "the" '''
        output = output.replace('<unk>','the')
        results.append(output)
    return results

def remove_repeated_word(text):
    '''function to remove consecutive repeated words'''
    # if len(text) less than 2, we do nothing
    if len(text) <  2:
        return text

    sentence = []
    list_text = text.split()
    for idx in range(len(list_text)):
        # first word, we move to next word and set the prev as the first word
        if idx ==0:
            prev_word = list_text[idx]
            sentence.append(prev_word)
            continue

        current_word = list_text[idx]
        # if previous word is the same as current word, we don't to sentence
        if prev_word== current_word:
            continue
        else:
            sentence.append(current_word)
            prev_word = current_word
    return ' '.join(sentence)

def argmax(iterable):
    '''get index of the max value (argmax) in the list'''
    return max(enumerate(iterable), key=lambda x: x[1])[0]

def bleu(ref_t, pred_t):
    '''calculate BLEU scores'''
    return sacrebleu.corpus_bleu(pred_t, [ref_t], force=True, lowercase=True, tokenize='none')


# ---Model Definition etc.---
# DO NOT MODIFY ANYTHING BELOW HERE
class Encoder(nn.Module):
    """
    Encoder class
    """

    def __init__(self, source_vocab_size, embed_dim, hidden_dim,
                 n_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(source_vocab_size, embed_dim,
                                  padding_idx=hp.pad_idx)
        self.rnn = nn.GRU(embed_dim,
                          hidden_dim,
                          n_layers,
                          dropout=dropout,
                          bidirectional=True)

    def forward(self, source, hidden=None):
        """
        param source: batched input indices
        param hidden: initial hidden value of self.rnn
        output (encoder_out, encoder_hidden):
            encoder_hidden: the encoder RNN states of length len(source)
            encoder_out: the final encoder states, both direction summed up
                together h^{forward} + h^{backward}
        """
        embedded = self.embed(source)  # (batch_size, seq_len, embed_dim)
        # get encoded states (encoder_hidden)
        encoder_out, encoder_hidden = self.rnn(embedded, hidden)

        # sum bidirectional outputs
        encoder_final = (encoder_out[:, :, :self.hidden_dim] +  # forward
                         encoder_out[:, :, self.hidden_dim:])   # backward

        # encoder_final:  (seq_len, batch_size, hidden_dim)
        # encoder_hidden: (n_layers * num_directions, batch_size, hidden_dim)
        return encoder_final, encoder_hidden


class Decoder(nn.Module):
    def __init__(self, target_vocab_size,
                 embed_dim, hidden_dim,
                 n_layers,
                 dropout):
        super(Decoder, self).__init__()
        self.target_vocab_size = target_vocab_size
        self.n_layers = n_layers
        self.embed = nn.Embedding(target_vocab_size,
                                  embed_dim,
                                  padding_idx=hp.pad_idx)
        self.attention = AttentionModule(hidden_dim)

        self.rnn = nn.GRU(embed_dim + hidden_dim,
                          hidden_dim,
                          n_layers,
                          dropout=dropout)

        self.out = nn.Linear(hidden_dim * 2, target_vocab_size)

    def forward(self, output, encoder_out, decoder_hidden):
        """
        decodes one output frame
        """
        embedded = self.embed(output)  # (1, batch, embed_dim)
        context, alpha = self.attention(decoder_hidden[-1:], encoder_out)
        # 1, 1, 50 (seq, batch, hidden_dim)
        rnn_output, decoder_hidden =\
            self.rnn(torch.cat([embedded, context], dim=2), decoder_hidden)
        output = self.out(torch.cat([rnn_output, context], 2))
        return output, decoder_hidden, alpha


class Seq2Seq(nn.Module):
    def __init__(self, fields=None, srcLex=None, tgtLex=None, build=True):
        super(Seq2Seq, self).__init__()
        # If we are loading the model, we don't build it here
        if build is True:
            self.params = {
                'srcLexSize': len(srcLex.vocab),
                'tgtLexSize': len(tgtLex.vocab),
                'embed_dim': hp.embed_dim,
                'hidden_dim': hp.hidden_dim,
                'n_layers': hp.n_layers,
                'dropout': hp.dropout,
                'fields': fields,
                'maxLen': hp.max_len,
            }
            self.build()

    def build(self):
        # self.params are loaded, start building the model accordingly
        self.encoder = Encoder(
            source_vocab_size=self.params['srcLexSize'],
            embed_dim=self.params['embed_dim'],
            hidden_dim=self.params['hidden_dim'],
            n_layers=self.params['n_layers'],
            dropout=self.params['dropout'])
        self.decoder = Decoder(
            target_vocab_size=self.params['tgtLexSize'],
            embed_dim=self.params['embed_dim'],
            hidden_dim=self.params['hidden_dim'],
            n_layers=self.params['n_layers'],
            dropout=self.params['dropout'])
        self.fields = self.params['fields']
        self.maxLen = self.params['maxLen']

    def forward(self, source, maxLen=None, eos_index=2):
        """
        This method implements greedy decoding
        param source: batched input indices
        param maxLen: maximum length of generated output
        param eos_index: <EOS> token's index
        """
        if maxLen is None:
            maxLen = self.maxLen
        encoder_out, encoder_hidden = self.encoder(source)

        return greedyDecoder(self.decoder, encoder_out, encoder_hidden, maxLen, eos_index)

    def tgt2txt(self, tgt):
        return " ".join([self.fields['tgt'].vocab.itos[int(i)] for i in tgt])

    def save(self, file):
        torch.save((self.params, self.state_dict()), file)

    def load(self, file):
        self.params, state_dict = torch.load(file, map_location='cpu')
        self.build()
        self.load_state_dict(state_dict)


class DataFrameDataset(data.Dataset):
    """Class for using pandas DataFrames as a datasource"""

    def __init__(self, examples, fields, filter_pred=None):
        """
        Create a dataset from a pandas dataframe of examples and Fields
        Arguments:
            examples pd.DataFrame: DataFrame of examples
            fields {str: Field}: The Fields to use in this tuple. The
                string is a field name, and the Field is the associated field.
            filter_pred (callable or None): use only exanples for which
                filter_pred(example) is true, or use all examples if None.
                Default is None
        """
        fields = dict(fields)
        self.examples = examples.apply(
            SeriesExample.fromSeries, args=(fields,), axis=1).tolist()
        if filter_pred is not None:
            self.examples = filter(filter_pred, self.examples)
        self.fields = dict(fields)
        # Unpack field tuples
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]


class SeriesExample(data.Example):
    """Class to convert a pandas Series to an Example"""

    @classmethod
    def fromSeries(cls, data, fields):
        return cls.fromdict(data.to_dict(), fields)

    @classmethod
    def fromdict(cls, data, fields):
        ex = cls()

        for key, field in fields.items():
            if key not in data:
                raise ValueError(
                    f"Specified key {key} was not found in the input data")
            if field is not None:
                setattr(ex, key, field.preprocess(data[key]))
            else:
                setattr(ex, key, data[key])
        return ex


def biload(src_file, tgt_file, linesToLoad=50000, verbose=False):
    src = open(src_file,encoding='utf-8').read().lower().strip().split("\n")
    tgt = open(tgt_file,encoding='utf-8').read().lower().strip().split("\n")
    return list(map(lambda x: (x[0].strip().split(), x[1].strip().split()), zip(src, tgt)))[:linesToLoad]


def bitext2Dataset(src, tgt, srcLex, tgtLex,
                   linesToLoad=50000, maxLen=hp.max_len):
    data = biload(src, tgt, linesToLoad=linesToLoad, verbose=False)
    data = [(f, e) for f, e in data if len(f) <= maxLen and len(e) <= maxLen]
    data = {'src': [f for f, e in data],
            'tgt': [e for f, e in data]}

    df = pd.DataFrame(data, columns=["src", "tgt"])
    dataset = DataFrameDataset(df, [('src', srcLex), ('tgt', tgtLex)])
    return dataset


def loadData(batch_size, device=0,
             trainNum=sys.maxsize, testNum=sys.maxsize):
    def tokenize(x):
        return x.split()

    srcLex = data.Field()
    tgtLex = data.Field(init_token="<SOS>", eos_token="<EOS>")

    train = bitext2Dataset('./data/train.tok.de',
                           './data/train.tok.en', srcLex, tgtLex,
                           linesToLoad=trainNum)
    val = bitext2Dataset('./data/val.tok.de',
                         './data/val.tok.en', srcLex, tgtLex)
    test = bitext2Dataset('./data/input/dev.txt',
                          './data/reference/dev.out', srcLex, tgtLex,
                          linesToLoad=testNum,
                          maxLen=sys.maxsize)

    srcLex.build_vocab(train.src, max_size=hp.lexicon_cap)
    tgtLex.build_vocab(train.tgt, max_size=hp.lexicon_cap)

    train_iter, = data.BucketIterator.splits(
        (train,),
        batch_size=batch_size,
        sort_key=lambda x: len(x.src),
        device=device,
        repeat=False)

    val_iter, = data.BucketIterator.splits(
        (val,),
        batch_size=batch_size,
        device=device,
        repeat=False)

    test_iter = data.Iterator(
        test,
        batch_size=1,
        device=device,
        sort=False,
        sort_within_batch=False,
        shuffle=False,
        repeat=False)

    return train_iter, val_iter, test_iter, srcLex, tgtLex


def loadTestData(srcFile, srcLex, device=0, linesToLoad=sys.maxsize):
    def tokenize(x):
        return x.split()
    test = bitext2Dataset(srcFile,
                          srcFile, srcLex, srcLex, linesToLoad,
                          maxLen=sys.maxsize)
    test_iter = data.Iterator(
        test,
        batch_size=1,
        device=device,
        sort=False,
        sort_within_batch=False,
        shuffle=False,
        repeat=False)
    return test_iter


if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option(
        "-m", "--model", dest="model", default=os.path.join('data', 'seq2seq_E049.pt'),
        help="model file")
    optparser.add_option(
        "-i", "--input", dest="input", default=os.path.join('data', 'input', 'dev.txt'),
        help="input file")
    optparser.add_option(
        "-n", "--num", dest="num", default=sys.maxsize, type='int',
        # "-n", "--num", dest="num", default=5, type='int',
        help="num of lines to load")
    optparser.add_option(
        "-o", "--outputfile", dest="outputfile",
                         default='output.txt', help="print result to output file")


    optparser.add_option(
        "-r", "--refcases", dest="ref", default=os.path.join('data', 'reference', 'dev.out'), help="references [default: data/reference/dev.out]")
    optparser.add_option(
        "-a", "--model2", dest="model2", default=os.path.join('data', 'seq2seq_E048.pt'),
        help="model file")
    optparser.add_option(
        "-b", "--model3", dest="model3", default=os.path.join('data', 'seq2seq_E047.pt'),
        help="model file")
    optparser.add_option(
        "-c", "--model4", dest="model4", default=os.path.join('data', 'seq2seq_E046.pt'),
        help="model file")
    optparser.add_option(
        "-t", "--train", dest="train", default='True',help="train to learn ensemble weights")
    optparser.add_option(
        # "-e", "--ensemble", dest="ensemble", default='True',help="use ensemble mode or not")
        "-e", "--ensemble", dest="ensemble", default='False',help="use ensemble mode or not")

    (opts, _) = optparser.parse_args()

    if opts.ensemble == 'False':
        model = Seq2Seq(build=False)
        model.load(opts.model)
        model.to(hp.device)
        model.eval()
        # loading test dataset
        test_iter = loadTestData(opts.input, model.fields['src'],
                                    device=hp.device, linesToLoad=opts.num)
        final_result = translate(model, test_iter)
        print("\n".join(final_result))

    else:
        '''Use Ensemble mode (load up various models and decode), took long time !'''
        '''load all models '''
        ''' each model took around 6 mins for decoding (GPU)'''
        list_model=[opts.model,opts.model2,opts.model3,opts.model4]
        test_iter = None
        list_results = []
        for opts_model in list_model:
            model = Seq2Seq(build=False)
            model.load(opts_model)
            model.to(hp.device)
            model.eval()

            if test_iter is None:
                '''load data once using for all models'''
                test_iter = loadTestData(opts.input, model.fields['src'],
                                         device=hp.device, linesToLoad=opts.num)
            results = translate(model, test_iter)
            list_results.append(results)

        '''Get reference file '''
        with open(opts.ref, 'r') as refh:
            ref_data = [str(x).strip() for x in refh.read().splitlines()]

        final_result=[]
        ''' initialize dict to learn weight of ensemble models '''
        dict_count = {}
        for idx_model in range(len(list_model)):
            dict_count["model_"+str(idx_model)] =0

        '''iterate row by row of the translated text'''
        for idx_row in range(len(list_results[0])):
            scores = []
            '''use the below function if we decode and find ensemble weights during training phase (dev.txt)'''
            if opts.train == 'True' and (len(ref_data) == len(test_iter)):
                for result in list_results:
                    '''compare result with reference row by row '''
                    bleu_score =bleu([ref_data[idx_row]], [result[idx_row]]).score
                    scores.append(bleu_score)

                # find index of max BLEU scores
                idx_max = argmax(scores)
                final_result.append(list_results[idx_max][idx_row])
                dict_count["model_"+str(idx_max)] += 1

            else:
                '''else, we decode during test phase (test.txt)'''
                '''Default: get weights from trained ensemble model(E049,...,E046) =[0.42, 0.24, 0.18,0.16] '''
                default_weight = [0.42, 0.24, 0.18,0.16]

                '''randomly draw according to weight of trained ensemble models'''
                idx_draw = random.choices(population=[i for i in range(len(list_results))], weights=default_weight ,k=1)[0]
                final_result.append(list_results[idx_draw][idx_row])
                dict_count["model_"+str(idx_draw)] += 1

        if opts.train == 'True' and (len(ref_data) == len(test_iter)):
            ''' get weight of trained ensemble models'''
            for model,val in dict_count.items():
                dict_count[model] = val/len(list_results[0])

        print("\n".join(final_result))

    ''' Print out to file instead of using python ... > ... '''
    original_stdout = sys.stdout
    with open(opts.outputfile, 'w',encoding='utf-8') as f:

        sys.stdout = f  # Change the standard output to the file we created.
        print("\n".join(final_result))

        sys.stdout = original_stdout
