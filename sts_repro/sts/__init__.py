# Structural Topic and Sentiment-Discourse (STS) Model
# Faithful reproduction of Chen & Mankad (2024/2025) Management Science
# Implementation in Python with variational inference (Laplace approximation),
# multinomial→Poisson regression with sample aggregation, and anchor-words initialization.
# Author: Aime

from .model import STSModel
from .data import Corpus, build_vocabulary, vectorize_corpus
from .anchor import AnchorInitializer
from .em import EMRunner
from .viz import plot_top_words, plot_prevalence_sentiment
