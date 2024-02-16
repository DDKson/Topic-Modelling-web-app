import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from underthesea import word_tokenize
from utils.data_preprocessing import *
from gensim.models.ldamodel import LdaModel
from gensim.models.lsimodel import LsiModel
# from top2vec import Top2Vec

# from octis.evaluation_metrics.diversity_metrics import TopicDiversity
# from octis.evaluation_metrics.coherence_metrics import Coherence

# from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

with open("./utils/vietnamese-stopwords.txt", encoding="utf-8") as f:
    STOPWORDS = f.readlines()
    STOPWORDS = [remove_all_tag(i).strip() for i in STOPWORDS]
STOPWORDS.extend(["negative", "positive"])


class UnsupervisedModels:
  def __init__(self, data, model = None, clean_data = None):
    self.data = data
    self.model_name = model
    self.evaluation = {}
    self.clean_data = clean_data
    self.id2word = None
    self.corpus = None
    self.trained_model = None
  def preprocess(self):
    self.data["comment"] = self.data["comment"].astype(str)
    self.clean_data = self.data.copy()

    def tokenize(x):
      x = word_tokenize(x)
      x = [word for word in x if word not in STOPWORDS]
      return x
    self.clean_data["clean_comment"] = self.clean_data["comment"].apply(lambda x: cleaning(x))
    self.clean_data["tokens"] = self.clean_data["clean_comment"].apply(lambda x: tokenize(x))
    self.clean_data = self.clean_data[self.clean_data["tokens"].apply(len) > 0]
    return self.clean_data


  def train_gensim_models(self, num_topics = 10, passes = 15, evaluate = False):
    if self.clean_data is None:
      clean_data = self.preprocess()
    else:
      clean_data = self.clean_data.copy()
    data_words_without_stopwords = list(self.clean_data["tokens"])
    self.id2word = corpora.Dictionary(data_words_without_stopwords)
    self.corpus = [self.id2word.doc2bow(text) for text in data_words_without_stopwords]

    if self.model_name == "LDA":
      model = LdaModel(self.corpus, num_topics=num_topics, id2word=self.id2word, passes=passes, random_state = 60, per_word_topics=True)
    elif self.model_name == "LSI":
      model = LsiModel(self.corpus, num_topics=num_topics, id2word=self.id2word)

    if evaluate:
      coherence_model = CoherenceModel(model=model, texts=data_words_without_stopwords, dictionary=self.id2word, topn = 10, coherence = "c_v", processes = 1)
      coherence = coherence_model.get_coherence() # higher score = better

      unique_words = set()
      topics = {'Topic_' + str(i): [token for token, score in model.show_topic(i, topn=10)] for i in range(0, model.num_topics)}
      for topic in topics:
          unique_words = unique_words.union(set(topic[:10]))
      td = len(unique_words) / (10 * len(topics))
      self.evaluation = {"Coherence Score": coherence, "Topic Diversity": td}
    self.trained_model = model
    return model

  def train_bertopic(self, vectorizer, evaluate = False, num_topics = None):
    if self.clean_data is None:
      clean_data = self.preprocess()
    else:
      clean_data = self.clean_data.copy()

    if not num_topics:
      num_topics = "auto"

    model = BERTopic(vectorizer_model=vectorizer, language="multilingual", nr_topics=num_topics)
    topics, _ = model.fit_transform(clean_data["clean_comment"])
    if evaluate:
      all_words = [word for words in clean_data["tokens"] for word in words]
      bertopic_topics = [
          [
              vals[0] if vals[0] in all_words else all_words[0]
              for vals in model.get_topic(i)[:10]
          ]
          for i in range(len(set(topics)) - 1)
      ]

      output_tm = {"topics": bertopic_topics}
      data_words = list(clean_data["tokens"])
      coherence = Coherence(texts = data_words, topk=10, measure="c_v")
      topic_diversity = TopicDiversity(topk=10)
      self.evaluation = {"Coherence Score": coherence.score(output_tm), "Topic Diversity": topic_diversity.score(output_tm)}
    return (model, topics, _)

  def get_prob_features(self):
    rows = []
    for i in range(len(self.corpus)):
      row = {}
      probs = self.trained_model[self.corpus[i]][0]
      for index, prob in enumerate(probs):
        if prob:
          Topic_name = "Topic_" + str(prob[0])
          row[Topic_name] = prob[1]
      rows.append(row)
    return rows

  def train_top2vec(self, evaluate = False, nr_topics = None):
    if self.clean_data is None:
      clean_data = self.preprocess()
    else:
      clean_data = self.clean_data.copy()
    corpus = list(clean_data["clean_comment"])
    top2vec = Top2Vec(documents=corpus, speed="learn", workers=4, embedding_model= "distiluse-base-multilingual-cased", tokenizer = word_tokenize)
    if evaluate:
      tokenized = list(clean_data["tokens"])
      if nr_topics:
        top2vec.hierarchical_topic_reduction(nr_topics)
        topic_words, _, _ = top2vec.get_topics(reduced=True)
      else:
        topic_words, _, _ = top2vec.get_topics()
      topics_old = [list(topic[:10]) for topic in topic_words]
      all_words = [word for words in tokenized for word in words if words]
      topics = []
      for topic in topics_old:
          words = []
          for word in topic:
              if word in all_words:
                  words.append(word)
              else:
                  print(f"error: {word}")
                  words.append(all_words[0])
          topics.append(words)
      output_tm = {
            "topics": topics,
        }
      data_words = list(tokenized)
      coherence = Coherence(texts = data_words, topk=10, measure="c_v")
      topic_diversity = TopicDiversity(topk=10)
      self.evaluation = {"Coherence Score": coherence.score(output_tm), "Topic Diversity": topic_diversity.score(output_tm)}
    return top2vec
  def get_evaluation(self):
    return self.evaluation