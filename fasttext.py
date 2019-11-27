import torch
import torch.nn as nn
import gensim
from gensim.test.utils import datapath
from gensim.models import FastText
from tqdm import tqdm

class DeepLang:
    def __init__(self, name, dim_siz=20):
        self.name = name
        self.model = FastText(size=dim_siz, window=3, min_count=1, iter=10)
        # self.model.build_vocab(sentences=['SOS', 'EOS', '<NONE>', '<LINE>']) # 낱말
        self.model.build_vocab(sentences=[['SOS'], ['EOS'], ['<NONE>'], ['<LINE>']]) # 단어
        print('[INFO] Default words:',self.model.wv.index2entity)
        self.total_words = self.model.corpus_total_words
        print('[INFO] Total words in the vocab', self.total_words)

    def buildVocab(self, corpus_path):
        self.model.build_vocab(corpus_file=corpus_path, update=True)
        self.total_words = self.model.corpus_total_words
        print('[INFO] Total words in the vocab', self.total_words)
    
    def buildPOSVocab(self, corpus_path):
        from konlpy.tag import Kkma
        tagger = Kkma()
        corpus_sentences = []
        corpus_sentences_torkenized = []
        with open(corpus_path, 'r', encoding='utf-8') as cf:
            corpus_sentences = cf.readlines()
            print('[INFO] Torkenizing input corpus')
            corpus_sentences_torkenized = [tagger.morphs(sentence.strip()) for sentence in tqdm(corpus_sentences) if sentence.strip() != ' ']
            for cst_idx, cst in enumerate(corpus_sentences_torkenized):
                if cst == []:
                    corpus_sentences_torkenized.pop(cst_idx)
        self.model.build_vocab(sentences=corpus_sentences_torkenized, update=True)
        self.total_words = self.model.corpus_total_words
        print('[INFO] Total words in the vocab / FASTTEXT', self.total_words)
        print('[INFO] Words:',self.model.wv.index2entity)

    def doEmbedding(self, corpus_path, POSVob=False):
        self.buildVocab(corpus_path)
        if POSVob: self.buildPOSVocab(corpus_path)
        self.total_words = self.model.corpus_total_words  # number of words in the corpus
        self.model.train(corpus_file=corpus_path, total_words=self.total_words, epochs=5)
        self.saveModel()
        
    def saveModel(self, save_model_path='pretrained_fasttext.bin'):
        self.model.wv.save_word2vec_format(save_model_path, binary=True) # save model

    def loadModel(self, save_model_path='pretrained_fasttext.bin'):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(save_model_path, binary=True)

    def createTorchEmbedding(self):
        weights = torch.FloatTensor(self.model.vectors)
        embedding = nn.Embedding.from_pretrained(weights)
        embedding.requires_grad = False
        return embedding

    def vector2word(self, vector): # index2word
        return self.model.most_similar([vector], topn=1)#[0][0]

    def word2vector(self, word): # word2index
        if word not in self.model.wv.index2entity:
            print('[INFO] This word is not in vocabs:',word)
        return self.model[word]
    
    def positive_negative(self, postive_words, negative_words):
        related_words = self.model.most_similar(positive=postive_words, negative=negative_words)
        return related_words

if __name__ == "__main__":
    lang = DeepLang('Input')
    lang.doEmbedding(corpus_path='corpus.txt', POSVob=True)
    test_vec = lang.word2vector('똥개')
    test_vec = lang.word2vector('윤의녕')
    test_vec = lang.word2vector('나무')
    print(test_vec)
    print(lang.vector2word(test_vec))