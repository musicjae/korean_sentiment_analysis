from soynlp.tokenizer import RegexTokenizer
import konlpy

tok = konlpy.tag.Mecab()
tokenizer = RegexTokenizer()

print(tok.morphs('동일하게 테스트 중입니다'))
print(tokenizer.tokenize('테스트 중이다'))