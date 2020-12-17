from tqdm import tqdm
from os import listdir
import string
import pickle
from random import shuffle

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# split a document into news story and highlights
def split_story(doc):
    # find first highlight
    index = doc.find('@highlight')
    # split into story and highlights
    story, highlights = doc[:index], doc[index:].split('@highlight')
    # strip extra white space around each highlight
    highlights = [h.strip() for h in highlights if len(h) > 0]
    return story, highlights

# load all stories in a directory
def load_stories(directory):
    stories = list()
    for name in tqdm(listdir(directory)):
        filename = directory + '/' + name
        # load document
        doc = load_doc(filename)
        # split into story and highlights
        story, highlights = split_story(doc)
        # store
        stories.append({'story': story, 'highlights': highlights})
    return stories

# clean a list of lines
def clean_lines(lines):
    cleaned = list()
    # prepare a translation table to remove punctuation
    table = str.maketrans('', '', string.punctuation)
    for line in lines:
        # print(']]]]] ', line)
        # strip source cnn office if it exists
        index = line.find('(CNN) -- ')
        if index > -1:
            line = line[index+len('(CNN)'):]
        # tokenize on white space
        line = line.split()
        # convert to lower case
        line = [word.lower() for word in line]
        # remove punctuation from each token
        line = [w.translate(table) for w in line]
        # remove tokens with numbers in them
        line = [word for word in line if word.isalpha()]
        # store as string
        cleaned.append(' '.join(line))
    # remove empty strings
    cleaned = [c for c in cleaned if len(c) > 0]
    return cleaned

MIN_LENGTH = 10
def remove_short_sen(list_sen):
    # remove very short sentence eg. meta data
    list_rm = []

    # add sentence to remove
    for sen in list_sen:
        len_sen = len(sen.split(' '))
        if len_sen < MIN_LENGTH:
            list_rm.append(sen)
        else:
            break
    # remove those sentence if list_rm not empty
    if list_rm:
        for sen_rm in list_rm:
            list_sen.remove(sen_rm)

    return list_sen

def remove_junk_cnntag1(list_sen):
    # remove word 'new' at the begining of highlights
    i = 0
    for sen in list_sen:
        if sen.split(' ')[0] == 'new':
            list_sen[i] = sen[len('new')+1:]
        i += 1
    return list_sen

def remove_junk_cnntag2(list_sen):
    # remove word 'cnn' at the begining of stories
    first_word = list_sen[0]
    if first_word.split(' ')[0] == 'cnn':
        list_sen[0] = first_word[len('cnn')+1:]
    elif first_word.split(' ')[0][0:len('cnn')] == 'cnn':
        #if first word contains 'cnn' tag
        list_sen[0] = first_word[len('cnn'):]

    return list_sen

if __name__ == '__main__':
    # load stories
    # directory = 'data_zip/cnn/stories/'
    # directory = 'data_zip/cnn/samples/'

    # directory = 'data_zip/dailymail/stories/'
    # directory = 'data_zip/dailymail/samples/'

    # stories = load_stories(directory)
    # print('Loaded Stories %d' % len(stories))

    '''example of cleaning one data sample'''
    # example = {}
    # example['story'] = clean_lines(stories[index]['story'].split('\n'))
    # example['highlights'] = clean_lines(stories[index]['highlights'])
    # print(example['story'])
    # print(example['highlights'])

    '''clean all stories and highlights in every data sample '''
    # i = 0
    ## clean stories
    # for example in tqdm(stories):
        # example['id'] = 'id_dm_'+str(i)
        # example['story'] = clean_lines(example['story'].split('\n'))
        # example['highlights'] = clean_lines(example['highlights'])
        # i += 1

    '''save into pickle file'''
    # pickle.dump(stories, open('dm_dataset.pkl', 'wb'))

    ''' load from file: original paper 287k (CNN+DM):
    after processing:(CNN+DM=311964 ) DM=219503, CNN=92461 '''
    stories = pickle.load(open('data_zip/processedID_dm_dataset.pkl', 'rb'))
    # stories = pickle.load(open('data_zip/processedID_cnn_dataset.pkl', 'rb'))
    # print('Loaded Stories %d' % len(stories))

    # print(stories[92])
    # for i in range(980,1000):
        # print(stories[i]['highlights'])
        # print('------')
        # print(stories[i]['story'],i)
        # print(stories[i]['story'][0])
        # print()

    ''' For CNN dataset, remove junk from stories and highlights'''
    # for story in tqdm(stories):
        # story['highlights'] = remove_junk_cnntag1(story['highlights'])
        # story['story'] = remove_junk_cnntag2(story['story'])

    # stories[44]['highlights'] = remove_junk_cnntag1(stories[44]['highlights'])

    # print(stories[75742])
    # print(stories[164890])
    # for i in range(85,100):
        # print(stories[i]['highlights'])
        # print('------')
        # print(stories[i]['story'][0])
        # print()

    ''' remove very short sentence such as meta data'''
    # for story in tqdm(stories):
        # story['story'] = remove_short_sen(story['story'])

    ''' check min length of sentence in our inputs and remove'''
    for story in tqdm(stories):
        try:
            # if len(story['story'][0].split(' ')) < MIN_LENGTH:
            if len(story['highlights'][0].split(' ')) < 5:

                print(story['highlights'][:])
                print('0000000000')
        except:
            print(story['id'])
            # print(story['story'])
            # stories.remove(story)

    # print('Loaded Stories %d' % len(stories))

    ''' reindex data in case we remove some data'''
    # i = 0
    # for story in tqdm(stories):
        # story['id'] = 'id_dm_'+str(i)
        # story['id'] = 'id_cnn_'+str(i)
        # i += 1
    # print(stories[-1])

    '''create train dataset (train+val) and test dataset'''
    # stories_dm = pickle.load(open('data_zip/processedID_dm_dataset.pkl', 'rb'))
    # stories_cnn = pickle.load(open('data_zip/processedID_cnn_dataset.pkl', 'rb'))

    # print('Loaded Stories %d' % len(stories_dm))
    # print('Loaded Stories %d' % len(stories_cnn))

    # combined_stories = stories_cnn + stories_dm
    # print('Combined Stories %d' % len(combined_stories))

    ''' shuffle data (list) before creating train,test dataset '''
    # shuffle(combined_stories)
    # NUM_TEST = 10000
    # pickle.dump(combined_stories[:NUM_TEST], open('test_dataset.pkl', 'wb'))
    # pickle.dump(combined_stories[NUM_TEST:], open('train_dataset.pkl', 'wb'))

    # stories_tr = pickle.load(open('data_zip/train_dataset.pkl', 'rb'))
    # stories_te = pickle.load(open('data_zip/test_dataset.pkl', 'rb'))
    # print('Loaded Stories %d' % len(stories_tr)) # 301964
    # print('Loaded Stories %d' % len(stories_te)) # 10000

    # print(stories_te[-1])
