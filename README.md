# Finance complaints classification
Predicting how consumer finance complaints get resolved, using user complaints.

## Supervised Learning Task

Predict how a company will resolve a consumer complaint, based on the complaint's
text and associated metadata.

**Target variable:** `Company response to consumer`

**Task type:** Multi-class classification (4 classes after removing rare/unstable
categories 


## How to run a project? 

1. Go to `data_loading.py file`. 
2. Run the command: `python data_loading.py`
3. Run the command: `python preprocessing.py`. 
4. Run the command: `python hyperparameters_tuning.py` for finding best params with RandomizedSearchCV.
5. Run the command: `python XGBoost_classifier.py` for baseline model without hyper parameters tuning + extended model with Hyperopt method for parameters tuning.
6. Run the command: `python neural_network.py`.
7. Run the command: `python extended_neural_network.py`.
8. 

## Streamlit deployment

## Exploratory data analysis

## Time Series analysis

## Feature Engineering

### Columns I dropped entirely

| Column | Reason |
|---|---|
| `Tags` | 86.9% missing |
| `Company public response` | I identified this as a leakage risk — this field reflects the company's explanation for how it already handled the complaint, so it would only exist *after* the resolution decision, not before. It was also ~50% missing/boilerplate ("Company chooses not to provide a public response") |
| `Submitted via` | Constant value in my sampled data (100% "Web") — zero variance, no predictive value |
| `Complaint ID` | Row identifier, no predictive content |
| `Date received`, `Date sent to company` (as raw columns) | Not usable directly by any model; I only retained them through the derived features below |

### Text feature: `Consumer complaint narrative`

This is my primary model input. I applied this cleaning pipeline before vectorization:
1. **Redaction token removal** — CFPB masks personal information as `XXXX`/`XX`
   placeholders. I found these dominated my n-gram frequency analysis
   (`"xxxx xxxx"` appeared 1.13M times, more than any real word), so I removed
   them via regex before any further processing to avoid polluting the vocabulary.
2. **Tokenization** — word-level, via NLTK.
3. **Stopword removal** — standard English stopword list.
4. **Lemmatization** — I reduce words to their dictionary form (e.g. "charged" →
   "charge"), so inflected forms of the same word aren't split into separate
   vocabulary entries.
5. **Vectorization** — TF-IDF, `max_features=10,000`, unigrams + bigrams. I
   chose this cap after checking my vocabulary composition: the full cleaned
   vocabulary contained 80,333 unique tokens, but 77% appeared five times or
   fewer across my ~100,000-document sample, indicating a long tail of rare,
   non-generalizable terms. I set `max_features=10,000` to retain the more
   frequent, informative portion while excluding this tail.

**Note:** I only apply this cleaning pipeline (steps 1–4) to my classical models
(Logistic Regression, XGBoost). For my fine-tuned DistilBERT model, I use raw,
uncleaned narrative text with BERT's own subword tokenizer instead — stopword
removal and lemmatization are known to hurt transformer performance, since these
models are pretrained on natural, unprocessed language.

### Engineered numeric features

- **`word_count`** — narrative length in words. My EDA showed complaints
  resulting in monetary relief had a notably higher median length (~200 words)
  and greater variability than all other outcome categories; among complaints
  exceeding 526 words (my IQR outlier threshold), the monetary relief rate was
  6.1% versus 3.8% overall — a ~60% relative increase, which is why I included
  this as a feature.
- **`processing_days`** — gap between `Date received` and `Date sent to company`.
  I chose this because it's available prior to resolution, so there's no
  leakage concern.
- **`year`** — extracted from `Date received`. My EDA revealed a substantial
  shift in the response-label distribution around 2021 ("Closed with
  explanation" dropped from ~88% to ~60% of complaints, while "Closed with
  non-monetary relief" rose from ~10% to ~37%), so I expected `year` to carry
  real predictive signal. **Caveat I want to flag:** this may partly reflect a
  change in CFPB's labeling taxonomy or policy era rather than complaint
  content itself, and a model relying heavily on it may generalize poorly to
  complaints outside my training date range. My feature importance analysis
  showed `year` contributing meaningfully (XGBoost importance ≈ 0.0098) without
  dominating the top features, which suggests my model relies primarily on
  text/categorical content rather than solely on this temporal artifact.

I standardized all numeric features (zero mean, unit variance) before feeding
them into Logistic Regression; I didn't scale them for XGBoost, since tree-based
splits are unaffected by monotonic transformations.

### Categorical features

- **`Product`, `Sub-product`, `State`** — I one-hot encoded these.
- **`Sub-product`, `Sub-issue`** — I found the missingness here was structural,
  not random: certain `Product` categories (e.g. "Credit reporting", "Payday
  loan") simply have no sub-category in CFPB's taxonomy, while `Credit card`
  showed a genuine partial split (16% missing). I filled missing values with
  `"Not applicable"` rather than imputing a mode value, to preserve this as a
  meaningful category rather than fabricating a false sub-product.
- **`State`** — I filled the small number of missing values (0.4%) with
  `"Unknown"`.
- **`Company`** — I found this was long-tail distributed (the top 3 companies —
  TransUnion, Equifax, Experian — accounted for roughly 45% of my entire
  sample; thousands of companies appeared only once or twice). Rather than
  one-hot encoding thousands of near-unique categories, I grouped values
  outside the top 20 most frequent companies into an `"Other"` category.

### Feature space summary

My final feature matrix contained 10,187 total features: 10,000 TF-IDF text
features (capped as described above), ~184 one-hot encoded categorical features
across `Product`/`Sub-product`/`State`/`Company`, and 3 engineered numeric
features (`word_count`, `processing_days`, `year`).

### What I considered but didn't include

- **SVD/dimensionality reduction on TF-IDF** — I considered this as a
  potential improvement for XGBoost specifically (tree-based models generally
  handle dense inputs better than very high-dimensional sparse ones), but I
  didn't apply it to my reported models, since I wanted to preserve
  word-level interpretability for feature importance analysis.
- **Frequency encoding for `Company`** (as an alternative to bucketing into
  "Other") — I found this technique used in prior work on this dataset, but I
  didn't implement it here, though I think it's a viable alternative worth
  noting.

## Baseline models
### Extreme Gradient Boosting
I chose XGBoost as my first baseline because it is a strong, well established
model for classification tasks that combine text and tabular features, and it
was one of the model types listed in my assignment as an expected option.

For my input features, I fed XGBoost a combination of TF IDF vectors from the
complaint narrative, one hot encoded categorical columns (Product, Sub product,
State, and Company), and three engineered numeric features (word count,
processing days, and year). I built this whole pipeline as a single scikit
learn Pipeline with a ColumnTransformer, so that text cleaning, vectorization,
and encoding all happen consistently and are always fit only on my training
data, avoiding any leakage into my validation or test sets.

Since my target variable is heavily imbalanced (Closed with explanation makes
up about 72% of my sample, while Untimely response is under 1%), I applied
sample weighting through compute_sample_weight, so that the model is penalized
more for misclassifying my rare classes rather than simply optimizing for the
dominant class.

I also attempted hyperparameter tuning with RandomizedSearchCV, scored on
macro F1 rather than accuracy or weighted F1, since those metrics would still
be dominated by my majority class. I initially tried a full search with 30
candidates and 3 fold cross validation, but individual fits were taking up to
22 minutes each, which made the full search impractical given my timeline. I
reduced the search to a 20,000 row subsample of my training data with a
narrower hyperparameter range and fewer iterations, which brought the search
down to a manageable runtime. Interestingly, the tuned model performed
slightly worse than my untuned baseline (macro F1 of 0.385 versus 0.409), which
I believe reflects both the narrower search space I had to use for speed and
the fact that XGBoost's defaults are already fairly strong for this kind of
data. I kept my baseline model as my reported XGBoost result for this reason,
and documented the tuning attempt as a negative result rather than discarding
it.

### Feed-forward neural network
I chose a feed forward neural network as my second baseline model, since it
gave me a genuinely different architecture to compare against my tree based
model, and Neural Networks (PyTorch/TensorFlow) were explicitly listed as a
valid option in my assignment.

Rather than building a separate feature pipeline for this model, I reused the
exact same ColumnTransformer output I had already built for XGBoost, meaning
the neural network receives the same TF IDF, one hot encoded, and scaled
numeric features. I made this choice deliberately, since it let me isolate the
effect of model architecture on performance, without also changing what
information the model has access to.

Because my feature matrix is dominated by a large, sparse TF IDF matrix
(around 10,000 of my roughly 10,187 total features), I could not simply load
everything into memory as a dense array without using several gigabytes of
RAM. I wrote a small batching function that converts only one batch at a time
from sparse to dense, which kept memory use manageable throughout training.

For class imbalance, I used a weighted CrossEntropyLoss, computed with
compute_class_weight, which is the PyTorch equivalent of the sample weighting
I used for XGBoost.

My first version of this model was a simple two hidden layer network trained
for 10 epochs, which reached a macro F1 of 0.4134 on my test set, my best
result at that point across all models I had tried. When I increased training
to 25 epochs, my validation macro F1 improved noticeably, but my test macro F1
barely moved (0.4134 to 0.4102), with the gains concentrated almost entirely
in my majority class. I took this as a sign that I was reaching a ceiling set
by my class imbalance and data volume, rather than by how long I trained the
model.

Based on that finding, I built an extended version of this model with a deeper
architecture, batch normalization, a learning rate scheduler, early stopping,
and gradient clipping, and I saved its results separately so I could compare
it directly against my simpler baseline rather than overwriting my original
result.
### Generative Large Language Models via API calls

## Evaluation metrics

## Extended models

## Models comparison

## Feature importance

## Analysis of errors

## Results

## Tracking with mlflow

