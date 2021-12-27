"""
Text classification using metalearner (few shot learning)
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.base import TransformerMixin
from sklearn.pipeline import make_pipeline
from transformers import BertTokenizer, BertModel
from sklearn.base import ClassifierMixin
import torch
import tqdm
import numpy as np
from annoy import AnnoyIndex
import pandas as pd


data = fetch_20newsgroups(
    subset="train",
    categories=[
        "alt.atheism",
        "comp.graphics",
    ],
)

X, y = data.data, data.target

# pipeline = make_pipeline(
#     TextTransformer(),
#     KNeighborsClassifier(),
#     verbose=True
# )


class FastANN(ClassifierMixin):
    def __init__(self, n_neighbors=5, n_trees=100, metric='angular'):
        self.n_neighbors = n_neighbors
        self.n_trees = n_trees
        self.metric = metric
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained("bert-base-uncased")

    def fit(self, X, y):
        self.y = y
        self.ss = pd.Series(y).value_counts() * 0
        v = self.model(torch.tensor(self.tokenizer.encode(X[0][:500])).unsqueeze(0))[1].squeeze(0)
        self.ann = AnnoyIndex(v.shape[0], self.metric)
        self.ann.on_disk_build("test.ann")
        for i in tqdm.tqdm(range(len(X))):
            v = self.model(torch.tensor(self.tokenizer.encode(X[i][:500])).unsqueeze(0))[1].squeeze(0)
            self.ann.add_item(i, v)
        
        self.ann.build(self.n_trees)
        return self
    
    def predict_single_proba(self, x):
        v = self.model(torch.tensor(self.tokenizer.encode(x[:500])).unsqueeze(0))[1].squeeze(0)
        indx = self.ann.get_nns_by_vector(v, self.n_neighbors)
        ss = (self.ss + pd.Series(y[indx]).value_counts(normalize=True)).fillna(0).values
        return ss

    def predict_(self, X):
        pred = []
        for i in tqdm.tqdm(range(len(X))):
            pred.append(self.predict_single_proba(X[i]))
        return np.stack(pred, 0)
    
    def predict_proba(self, X):
        return self.predict_(X)
    
    def predict(self, X):
        return np.argmax(self.predict_(X), axis=1)

model = FastANN()
model.fit(X, y)
print(model.score(X, y))

