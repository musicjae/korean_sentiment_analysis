from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

def preprocess_fn(dt,mode='train'):

    TRAIN_MAXLEN = 128
    TEST_MAXLEN = 128

    sens = ["[CLS]" + str(sen) + "[SEP]" for sen in dt]
    tokenized_text = [tokenizer.tokenize(sent) for sent in sens]

    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text]

    if mode == 'train':
        input_ids = pad_sequences(input_ids, maxlen=TRAIN_MAXLEN,dtype='long',truncating='post',padding='post')
    elif mode == 'test':
        input_ids = pad_sequences(input_ids, maxlen=TEST_MAXLEN,dtype='long',truncating='post',padding='post')

    return input_ids

### ==== attention mask ==== ###

def attention_mask(dt,mode='train'):
    attention_mask = []

    if mode == 'train':
        for seq in dt:
            seq_mask = [float(i>0) for i in seq]
            attention_mask.append(seq_mask)

        return attention_mask

def train_test_split_fn(text, label,mode='train'):

    if mode=='train':
        train_inputs, val_inputs, train_labels, val_labels = train_test_split(text,
                                                                              label,
                                                                              random_state=42,
                                                                              test_size = 0.1)
        return train_inputs, val_inputs, train_labels, val_labels





