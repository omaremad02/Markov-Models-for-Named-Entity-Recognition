# Markov Models for Named Entity Recognition (HMM & MEMM)

**Named Entity Recognition** on the CoNLL-style English dataset (tags: `PER`, `ORG`, `LOC`, `MISC`, `O`) using two probabilistic sequence models, built from scratch for the *Natural Language Processing* course and evaluated with confusion matrices and per-tag F1 scores.

## Models
### 1. Hidden Markov Model (`Hidden_Markov-Model-NER.ipynb`)
- Bigram HMM: transition and emission probabilities estimated from the training corpus
- Two decoders: **greedy decoding** and **Viterbi decoding** (dynamic programming)

### 2. Maximum Entropy Markov Model (`Max-Entropy-Markov-Model-NER.ipynb`)
- A discriminative model trained with logistic regression (scikit-learn) on hand-crafted, sparse one-hot features
- Features: the words in a ±4-word context window and up to 6 previous tags
- **Regex features** for word shape: capitalisation, all caps, digits, punctuation, etc.
- Compares performance with and without the regex features

## Results (test set, greedy HMM)
| Tag | O | PER | ORG | LOC | MISC |
|---|---|---|---|---|---|
| F1 | 0.97 | 0.64 | 0.58 | 0.80 | 0.74 |

## Pipeline
Data loading and normalisation (digits replaced with zeros) → word and tag vocabularies → training → tagging the test corpus → confusion matrix and F1 scores (scikit-learn).

## Tech
Python · NumPy · SciPy (sparse) · scikit-learn · NLP · HMM · Viterbi · MEMM · feature engineering
