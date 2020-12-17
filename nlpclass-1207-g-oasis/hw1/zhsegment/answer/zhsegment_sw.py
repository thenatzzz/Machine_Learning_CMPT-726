import re, string, random, glob, operator, heapq, codecs, sys, optparse, os, logging, math
from functools import reduce
from collections import defaultdict
from math import log10

# additional library
import operator

INDEX_PROBABILITY = 2
INDEX_WORD = 0
INDEX_STARTPOS = 1
INDEX_BACKPOINTER = 3

class Segment:

    def __init__(self, Pw):
        self.Pw = Pw

    def segment(self, text):
        "Return a list of words that is the best segmentation of text."
        if not text: return []
        segmentation = [ w for w in text ] # segment each char into a word
        # print('TEXT: ',text[0],self.Pw(text[0]))
        # print(segmentation, len(self.Pw),Pw.N,self.Pwords(text[0]))

        # call iterative_segmentation function
        segmentation = iterative_segmentation(text,self.Pw,self.Pwords)

        return segmentation

    def Pwords(self, words):
        "The Naive Bayes probability of a sequence of words."

        return self.Pw(words)
        return product(self.Pw(w) for w in words)

def product(nums):
    "Return the product of a sequence of numbers."
    return reduce(operator.mul, nums, 1)

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

        # get the first word
        if (text[0] == pword[0]) and pword in text:
            # multiply by -1 to cast into positive
            # then we can get Min Heap (minimum value at the top of heap)
            each_entry = [pword,0,-1.0*log10(Pwords(pword)),None]

            # push entry into the heap, sorted based on probability
            heappush_list(heap, each_entry, key=operator.itemgetter(INDEX_PROBABILITY)) # sort by prob

    '''if HEAP is still empty, we add smoothing '''
    if len(heap) == 0 :
        smoothing_pro = 1 / len(list(dict(Pw).items()))
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
                if (pword in text[endindex+1:]):

                    new_entry = [pword, endindex + len(pword), -1.0 * (entry[INDEX_PROBABILITY] + log10(Pwords(pword))),
                                     entry[INDEX_STARTPOS]]

                    # don't add new word if it is equal to popped word
                    if pword == entry[INDEX_WORD]:
                        continue
                    # if word is already in chart, don't add to heap
                    if check_prev_entry(new_entry,chart):
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

    list_result = []

    # get last element
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
        if key in self: return self[key]/self.N
        else: return self.missingfn(key, self.N)

def datafile(name, sep='\t'):
    "Read key,value pairs from file."
    with open(name,encoding="utf8") as fh:
    # with open(name) as fh:
        for line in fh:
            (key, value) = line.split(sep)
            yield (key, value)





if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts [default: data/count_1w.txt]")
    optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts [default: data/count_2w.txt]")
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="file to segment")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    Pw = Pdist(data=datafile(opts.counts1w))
    print("Pw.N: ",Pw.N, '\n\n')
    segmenter = Segment(Pw)
    i = 1
    with open(opts.input,encoding='utf8') as f:
        for line in f:
            # if i == 1:
                # i += 1
                # continue
            print(" line: ",i, line)
            sentence =" ".join(segmenter.segment(line.strip()))
            # print(" ".join(segmenter.segment(line.strip())))
            print(sentence)
            # print(sentence[0],' ***** ', Pw[sentence[0]]/Pw.N)
            print('-'*60)
            if i ==1:
                break
            i += 1

