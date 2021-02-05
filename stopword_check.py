from collections import Counter
from data import result

def test_anything(result,mode='counter'):

    if mode == 'counter':
        print(Counter(result[0]))

    elif mode == 'leng':
        print(len(result[0]))

print(test_anything(result,'leng'))
