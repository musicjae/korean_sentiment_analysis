import pandas as pd
from soynlp.tokenizer import RegexTokenizer
import stopword
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
tokenizer = RegexTokenizer()

train = pd.read_csv(r'C:\Users\USER\Desktop\gitgit\korean_sentiment_analysis\nsmc\ratings_train.txt',sep='\t') # 150000,3
test = pd.read_csv(r'C:\Users\USER\Desktop\gitgit\korean_sentiment_analysis\nsmc\ratings_test.txt',sep='\t') # 50000, 3

test_ls=[]

def preprocess(text):

    data = text.fillna(0)
    result = text.str.replace(pat = r'[^ ㄱ-ㅣ가-힣]+', repl=r' ', regex=True) # 한글을 제외한 나머지 제거
    result = result.str.replace(',','').astype(object)

    result = result.tolist() # Series를 list로

    return result

def labeling(text, data):
    result = []
    lb = list(data['label'])

    for t,l in enumerate(zip(text,lb)):

        result.append([l])

    return result

def tokenize(data):
    get_words_list = []
    get_sens_list = []
    stop_word_list = stopword.text
    cnt=0
    for sen in data:

        sen = str(sen)
        word = tokenizer.tokenize(sen)

        get_sens_list.append(word)

    return get_sens_list



#labeled_train = labeling(tokenize(preprocess(train['document'])),train)
#labeled_test = labeling(tokenize(preprocess(test['document'])),test)

train_x = tokenize(preprocess(train['document']))
train_y = list(train['label'])



class Mydataset(Dataset):

    def __init__(self,x,y):

        self.x = x
        self.y = y

    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.y)

#ds = Mydataset(train_x,train_y)

#dl = DataLoader(ds,shuffle=True,batch_size=32) # ==> maxpadding 문제 때문에 안 돌아감.


