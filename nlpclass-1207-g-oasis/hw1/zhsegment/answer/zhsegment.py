import re, string, random, glob, operator, heapq, codecs, sys, optparse, os, logging, math
from functools import reduce
from collections import defaultdict
from math import log10

# additional library
import operator

'''these indexes are for iterative_segmentation function'''
INDEX_WORD = 0
INDEX_STARTPOS = 1
INDEX_PROBABILITY = 2
INDEX_BACKPOINTER = 3

'''these indexes are for default segmentation function'''
IDX_WORD = 0
IDX_PROBABILITY = 1
MAX_WORD_LENGTH =15

class Segment:

    def __init__(self, Pw):
        self.Pw = Pw

    def Pwords(self, words):
        "The Probability of words."
        return self.Pw(words)

    def segment_old(self, text):
        "Return a list of words that is the best segmentation of text."
        '''Dev score = 0.86 and took long time to run, implementaion using Heap'''
        if not text: return []

        # call iterative_segmentation function
        segmentation = iterative_segmentation(text,self.Pw,self.Pwords)

        return segmentation

    def segment(self,text):
        "Return a list of words that is the best segmentation of text."
        '''Dev score = 0.93 and took long time to run, implementaion using normal Dict'''

        ''' dictionary as dynamic programming table'''
        chart = {}

        '''iterate through line of text'''
        for idx_text in range(len(text)):

            '''iterate and decide whether to add words to chart '''
            for idx_word in range(1, MAX_WORD_LENGTH + 1):

                '''continue if word length goes out of text length'''
                if (idx_text - idx_word + 1) < 0:
                    continue

                '''get word from text'''
                word = text[idx_text-idx_word+1:idx_text+1]

                '''get probability of current word'''
                prob = math.log(self.Pwords(word))

                ''' check for previous word probability,
                 if it exists we get probability of previous word, else we assign it to zero '''
                if (idx_text - idx_word) >= 0:
                    prev_prob = chart[idx_text - idx_word][IDX_PROBABILITY]
                else:
                    prev_prob = 0

                '''dynamically update new prob'''
                updated_prob = prob + prev_prob
                '''check if text in chart or not OR updated probability is more than current probability,
                  update chart with updated probability if the condition is True'''
                if (idx_text not in chart) or (updated_prob > chart[idx_text][IDX_PROBABILITY]):
                    chart[idx_text] = [word, prev_prob + prob]

        ''' Get the best segmented text by iterate from the end index of our chart'''
        endindex = len(text) - 1
        segmented_text = []

        while endindex >= 0:
            word, prob = chart[endindex]
            segmented_text.append(word)
            endindex = endindex- len(word)

        # return from end of array
        return segmented_text[::-1]


#### Support functions (p. 224)
def iterative_segmentation(text,Pw,Pwords):
    '''Iterative segmentation function, return list of segmented text'''

    def heappush_list(h, item, key=lambda x: x):
        '''push entry to heap'''
        heapq.heappush(h, (key(item), item))

    def heappop_list(h):
        ''' pop out entry from heap'''
        return heapq.heappop(h)[1]

    def check_prev_entry(current_entry,chart):
        ''' check whether there is previous entry existing in chart already or not'''
        if current_entry[INDEX_STARTPOS] in chart:
            return True
        return False

    def get_prev_entry(current_entry,chart):
        ''' return previous entry if it exists '''
        if current_entry[INDEX_STARTPOS] in chart:
            return chart[current_entry[INDEX_STARTPOS]]
        return 'Error'

    def exist_in_heap(heap,entry):
        ''' check whether there is previous entry existing in heap already or not'''
        for entry_h in heap:
            if entry_h[1][INDEX_WORD] == entry[INDEX_WORD]:
                return True
        return False

    '''Initialize the HEAP'''
    heap = []
    for pword,value in dict(Pw).items():

        MAX_WORD_LENGTH = 8
        for word_length in range(1,MAX_WORD_LENGTH):
            # get the first word
            if (text[:word_length] == pword[0:word_length]) and len(pword)==word_length:
                # multiply by -1 to cast into positive
                # then we can get Min Heap (minimum value at the top of heap)
                each_entry = [pword,word_length-1,-1.0*log10(Pwords(pword)),None]
                # push entry into the heap, sorted based on probability
                heappush_list(heap, each_entry, key=operator.itemgetter(INDEX_PROBABILITY)) # sort by prob

    '''if HEAP is still empty, we add smoothing '''
    if len(heap) == 0 :
        # smoothing 1/size of dictionary
        smoothing_pro = 1 / len(list(dict(Pw).items()))
        # smoothing_pro = 1 / Pw.N

        entry_add = [text[0], 0, smoothing_pro, None]
        heappush_list(heap, entry_add, key=operator.itemgetter(INDEX_PROBABILITY))

    '''Iteratively fill in CHART for all i '''
    chart = {}
    count = 0

    while heap:

        # get top entry from the heap
        entry = heappop_list(heap)
        # multiply -1 back to get original value of prob (original = negative log prob)
        entry[INDEX_PROBABILITY] = -1.0*entry[INDEX_PROBABILITY]

        # init endindex = entry starting position
        endindex = entry[INDEX_STARTPOS]

        '''iterate and decide whether to add words to heap '''
        for pword,value in dict(Pw).items():

            # break if there is no more text
            if endindex+1 >= len(text):
                break

            # match word from dict based on the first index with new text
            if pword[0] == text[endindex+1]:

                # if (pword in text):
                # if (pword in text[endindex+1:]):
                if (pword in text[endindex+1:endindex+1+len(pword)]):

                    new_entry = [pword, endindex + len(pword), -1.0 * (entry[INDEX_PROBABILITY] + log10(Pwords(pword))),
                                     entry[INDEX_STARTPOS]]

                    # don't add new word if it is equal to popped word
                    if pword == entry[INDEX_WORD]:
                        continue
                    # if word is in heap already, don't add
                    if exist_in_heap(heap,new_entry):
                        continue
                    else:
                        # add new word to heap
                        heappush_list(heap, new_entry, key=operator.itemgetter(INDEX_PROBABILITY))  # sort by prob

        ''' add smoothing for word that does not appear in dict'''
        if len(heap) == 0 and endindex < len(text)-1:
            smoothing_pro = 1 / len(list(dict(Pw).items()))
            entry_add = [text[endindex+1], endindex+1, smoothing_pro, endindex]
            heappush_list(heap, entry_add, key=operator.itemgetter(INDEX_PROBABILITY))

        '''check if chart is empty and whether word related to popped entry is in chart already or not'''
        if chart and check_prev_entry(entry,chart):
            # get previous entry
            previous_entry = get_prev_entry(entry,chart)

            # if assign popped entry to chart belonging to previous entry
            # if popped entry probability > previous entry probability
            if entry[INDEX_PROBABILITY] > previous_entry[INDEX_PROBABILITY]:
                chart[endindex] = entry

            # if popped entry probability <= previous entry probability, do nothing
            if entry[INDEX_PROBABILITY] <= previous_entry[INDEX_PROBABILITY]:
                count += 1
                continue

        else:
            # add popped word to chart
            chart[endindex] = entry

        count += 1

    return get_segmented_text(chart)

def get_segmented_text(dict_text):
    ''' Get list of word from Dynamic programming table (chart) '''
    # if dict_text is empty, we return empty list
    if len(dict_text) < 1:
        return []

    last_entry = dict_text[max(list(dict_text.keys()))]

    # get last element
    list_result = []
    list_result.append(last_entry[INDEX_WORD])
    # get pointer from last element
    ptr_idx = last_entry[INDEX_BACKPOINTER]

    # loop while backpoint is not None
    while ptr_idx != None:
        entry = dict_text[ptr_idx]
        list_result.append(entry[INDEX_WORD])
        ptr_idx = entry[INDEX_BACKPOINTER]

    #reverse list
    list_result = list_result[::-1]
    return list_result

class Pdist(dict):
    "A probability distribution estimated from counts in datafile."
    def __init__(self, data=[], N=None, missingfn=None):
        for key,count in data:
            self[key] = self.get(key, 0) + int(count)
        self.N = float(N or sum(self.values()))
        self.missingfn = missingfn or (lambda k, N: 1./N)

    def __call__(self, key):
        if key in self:
            return self[key]/self.N
        else:
            return self.missingfn(key, self.N)

def datafile(name, sep='\t'):
    "Read key,value pairs from file."
    with open(name,encoding="utf8") as fh:
    # with open(name) as fh:
        for line in fh:
            (key, value) = line.split(sep)
            yield (key, value)

def punish_long_words(key, N,lambda_=0.03):
    '''Function to assign probability to based on length of word
    we can define lambda (hyperparameter) (default=0.03)'''
    prob = (1.0/N) if len(key) <=1 else 1e-200+ pow(1.0/( lambda_*N), len(key))
    return prob

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts [default: data/count_1w.txt]")
    optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts [default: data/count_2w.txt]")
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="file to segment")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    Pw = Pdist(data=datafile(opts.counts1w),missingfn=punish_long_words)
    segmenter = Segment(Pw)
    i = 1
    with open(opts.input,encoding='utf8') as f:
        for line in f:
            # print(" line: ",i, line)
            sentence =" ".join(segmenter.segment(line.strip()))
            print(sentence)

            # print("segmented sentence:",sentence)
            # print('-'*50)

            # if i ==3:
                # break
            i += 1
