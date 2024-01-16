# MentalBERT: Publicly Available Pretrained Language Models for Mental Healthcare

## Author

**Shaoxiong Ji**: Aalto University, Finland
**Tianlin Zhang**: The University of Manchester, UK
**Jie Fu**: Mila, Quebec AI Institute, Canada
**Prayag Tiwari**: Aalto University, Finland
**Erik Cambria**: Nanyang Technological University, Singapore

<hr>

## Article

**Posted Date**: October 29th, 2021

**DOI**: https://arxiv.org/abs/2110.15621

<hr>

## Abstract

This paper trains and release two pretrained masked language models, i.e., MentalBERT and MentalRoBERTa, to benefit machine learning for the mental healthcare research community.  
Besides, we evaluate our trained domain-specific models and several variants of pretrained language models on several mental disorder detection benchmarks and demonstrate that language representations pretrained in the target domain improve the performance of mental health detection tasks.

## 1 Introduction

Suicide attempters have been reported as suffering from mental disorders, with an investigation on a shift from mental health to suicidal ideation conducted by language and interactional measures (De Choudhury et al., 2016).

Machine learning-based detection techniques can empower healthcare workers in early detection and assessment to take an action of proactive prevention.

The seminal work on a pretrained language model called BERT (Devlin et al., 2019) utilizes bidirectional transformerbased text encoders and trains the model on a largescale corpus.

Our paper trains and releases two representative bidirectional masked language models, i.e., BERT and RoBERTa (Liu et al., 2019), with corpus collected from social forums for mental health discussion.

## 2 Methods and Setup

Note that we aim to provide publicly available pretrained text embeddings as language resources and evaluate the usability in downstream tasks rather than propose novel pretraining techniques.

### 2.1 Language Model Pretraining 

We follow the standard pretraining protocols of BERT and RoBERTa with Huggingface’s Transformers framework (Wolf et al., 2020). 

- For the pretraining of RoBERTa based MentalBERT, Apply the dynamic masking mechanism that converges slightly slower than the static masking.
- Adopt the training scheme similar to the domain-adaptive pretraining 

**Advantange**
 - Utilize the learned knowledge from the general domain
 - Save computing resources 

**Experiment Env.**
- Four Nvidia Tesla v100 GPU
- Set the batch size to 16 per GPU
- Evaluate every 1,000 steps
- Train for 624,000 iterations
- Takes approximately 8 days

### 2.2 Pretraining Corpus

We identified and crawled subreddits related to mental health domains on Reddit.

The selected mental health-related subreddits include "r/depression", "r/SuicideWatch", "r/mentalillness/", and "r/mentalhealth".

Eventually, we make the training corpus with a total of 13,671,785 sentences.

### 2.3 Downstream Task Fine-tuning

We apply the pretrained MentalBERT and MentalRoBERTa in binary mental disorder detection and multi-class mental disorder classification of various mental disorders such as stress, anxiety, and depression. We fine-tune the language models in a downstream tasks. Specifically, we use the embedding of the special token \[CLS] of the last hidden layer as the final feature of the input text.

**MLP**
- Activation function: Hyperbolic tangent activation
- learning rate: 1e-05(transformer text encoder), 3e-05(classification layer)
- optimizer: Adam

## Results

### 3.1 Datasets

We evaluate and compare mental disorder detection methods on different datasets with various mental disorders (e.g., depression, anxiety, and suicidal ideation) collected from popular social platforms (e.g., Reddit and Twitter).

![img](https://github.com/Sameta-cani/papers/blob/main/imgs/Pasted%20image%2020240116213232.png)

### 3.2 Baselines

Note that the aim of this paper is not to achieve state-of-the-art performance but to demonstrate the usability and evaluate the performance of our pre-trained models, though we have achieved competitive performance in some datasets when compared with the SOTA.

### 3.3 Results and Discussion 

**Evaluate Metric**
- F1-score ($\because$ Mental disorder detection is usually a task with unbalanced classes)
- Recall score ($\because$ To reduce the FN)

![img](https://github.com/Sameta-cani/papers/blob/main/imgs/Pasted%20image%2020240116214849.png)

![img](https://github.com/Sameta-cani/papers/blob/main/imgs/Pasted%20image%2020240116215005.png)

#### Discussion 

When comparing the domainspecific pretrained models for mental health with models pretrained with general corpora, MentalBERT and MentalRoBERTa gain better performance in most cases. Those results show that continued pretraining on the mental health domain improves prediction performance in downstream tasks of mental health classification.

## 4 Conclusion and Future Work

Our paper is a positive attempt to benefit the research community by releasing the pretrained models for other practical studies and with the hope to facilitate some possible real-world applications to relieve people’s mental health issues. 

However, we only focus on the English language in this study since English corpora are relatively easy to obtain. In the future work, we plan to collect multilingual mental health-related posts, especially those less studied by the research community, and train a multilingual language model to benefit more people speaking languages other than English.
