"""
基于贝叶斯算法 拼写检查
 P(c)  文章中出现一个正确拼写词c的概率，也就是说，在英文文章中c出现的概率多大
 P(w|c)  用户键入c的情况下多大概率错敲成了w
 gramaxc 用来枚举所有可能的c 并选取最大的

"""
import re,collections

def wrods(text):
    return re.findall('[a-z]+',text.lower())

def train(features):
    model = collections.defaultdict(lambda :1)
    for f in features:
        model[f] += 1
    return model
NWORDS = train(wrods(open('dict.txt').read()))
alphabet = 'abcdefghijklmnopqresuvwxyz' #
print(NWORDS)

#编辑距离
# 输入一个单词 返回这个单词编辑距离为1的集合
def edits1(word):
    n = len(word)
    return set(
        [word[0:i]+word[i+1:] for i in range(n)]+
        [word[0:i]+word[i+1]+word[i]+word[i+2:] for i in range(n-1)]+
        [word[0:i]+c+word[i+1:] for i in range(n) for c in alphabet]+
        [word[0:i]+c+word[i:] for i in range(n+1) for c in alphabet]
        ) #自身少一个词的组合

# 边界距离等于2

def edits2(word1):
    return set([e2 for e1 in edits1(word1) for e2 in edits1(e1)])

def known(words):
    return set(w for w in words if w in NWORDS)
def correct(word):
    candidates = known([word]) or known(edits1(word)) or known(edits2(word)) or [word]
    return max(candidates,key=lambda w:NWORDS[w])

a = input('请输入单词：')
print('是不要输入',correct(a))