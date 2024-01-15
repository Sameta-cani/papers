# Summary

## 1. Problem Definition

- Alzheimer's Disease (AD) is a significant public health concern, especially due to its increasing prevalence in the aging population. The paper identifies the need for a non-invasive, cost-effective method for early AD diagnosis, as current methods like MRI and PET scans are expensive and invasive.

## 2. Existing Research

- The paper reviews various technologies for AD detection, including molecular biomarkers combined with deep learning on gene expression datasets. It discusses the limitations of traditional machine learning algorithms, which require extensive export knowledge and often lack portability and accuracy.

## 3. Experiment

### 3-1. Data Source

- The study utilizes the ADReSS dataset, which includes audio recordings and their corresponding transcrips from both Alzheimer's Disease patients and a control group of normal individuals.

### 3-2. Model Development

- A transfer learning approach is employed, using a pre-trained BERT model (specifically, distilBert) for text embedding. This process involves converting sentences or transcripts into 768-dimensional vectors.
- The extracted features from the BERT model are then used in a logistic regression classifier for the final classification task.

### 3-3. Performance Evaluation

- The model's performance is evaluted based on several metrics, including accuracy, precision, recall, and F1-score. These metrics help determine the model's effectiveness in correctly identifying cases of Alzheimer's Disease from the speech and text data.
- The study compares the performance of the proposed model with other classifiers like CNN, random forest, SVM, and AdaBoost to assess its relative effectiveness.

### 3-4. Experimental Focus

- The experiment primarily focuses on analyzing the speech patterns and language use in the transcripts, looking for markers that are indicative of cognitive impairment associated with Alzheimer's Diseas.
- The goal is to demonstrate the feasibility of using speech and language processing as a non-invaisve, cost-effective tool for early detection of Alzheimer's Desease.

## 4. Proposed Solution

- The paper proposes a novel model architecture that combines deep learning with machine learning. The distilBert model is used to extract deep semantic features, which are then passed to a logistic regression model for classification.
- The approach aims to provide a reliable, low-cost, and convenient solution for AD screening, leveraging the strengths of transfer learning and the attention machanism in BERT for focusing on specific markers related to AD.

<hr>

# A Transfer Learning Method for Detecting Alzheimer's Disease Based on Speech and Natural Language Processing

## Author 

**Ning Liu**:  School of Public Health, Hangzhou Normal University, Hangzhou, China
**Kexue Luo**: Department of Mathematics and Computer Science, Fujian Provincial Key Laboratory of Data-Intensive Computing, Quanzhou Normal University, Quanzhou, China
**Zhenming Yuan**: Tongde Hospital of Zhejiang Province Geriatrics, Hangzhou, China
**Yan Chen**: School of Information Science and Technology, Hangzhou Normal University, Hangzhou, China, 5 International Unresponsive Wakefulness Syndrome and Consciousness Science Institute, Hangzhou Normal University, Hangzhou, China

<hr>

## Article

**Keywords**: transfer learning, Alzheimer's disease, natural language processing, BERT, machine learning

**Posted Date**: April 19th, 2022

**DOI**: [Frontiers | A Transfer Learning Method for Detecting Alzheimer's Disease Based on Speech and Natural Language Processing (frontiersin.org)](https://www.frontiersin.org/articles/10.3389/fpubh.2022.772592/full)

<hr>

## Abstract

- we developed a transfer learning model based on speech and natural language processing (NLP) technology for the early diagnosis of AD.
- The lack of large datasets limits the use of complex neural network models without feature engineering, while transfer learning can effectively solve this problem.
- Concretely, a distilled bidirectional encoder representation (distilBert) embedding, combined with a logistic regression classifier, is used to distinguish AD from normal controls.
- The accuracy of the proposed model is 0.88, which is almost equivalent to the champion score in the challenge
- As a result, the transfer learning method in this study improves AD prediction, which does not only reduces the need for feature engineering but also addresses the lack of sufficiently large datasets.

## Introduction

In contrast to earlier studies with manual expert-wise feature extraction in this field, this study used a reliable deep learning model to automatically find suspicious AD symptom features from speeches. Specifically, a pre-trained distilBert language model was used as a feature extractor to obtain the features of the input sentence or document, and a simple logistic regression classifier, which has a good effect on binary classification, was used to classify AD from normal controls.

In addition, a grid search strategy was used to tune the parameters to obtain the best parameters of the model. The results show that this method worked better on ADRess datasets in 2020, with an accuracy of 0.88.

The main contributions of this study are as follows:

1. A simple and effective model of AD diagnosis based on transcripts without complicated expertise is designed and implemented effectively.
2. A novel model architecture that combines deep learning with machine learning is proposed, and the best performance on the ADRess dataset is obtained.
3. Our proposed approach has the advantages of reliability, low cost, and convenience and can provide a feasible solution for the screening of AD.

## Related Works

Because language functions play an important role in the detection of cognitive deficits at different stages, the combination of NLP technology and deep learning provides an accurate and convenient solution for the detection of AD and MCI.

**basic study**

- Luz et al. - 34 linguistic features combined with linear discriminant analysis, and obtained the best accuracy of 0.75 on the test dataset.
- ??? - Acoustic features only obtained an accuracy of approximately 0.5 on the classfiers used frequently.
- Balagopalan et al. - acoustic and text-based feature extraction and the BERT model, which obtained the best accuracy of 0.8332.
- Syed et al. and Yuan et al. - achieved accuracies of 85.45 and 89.6% using acoustic and linguistic features, respectively.
- Syed et al. - used acoustic features, and obtained an accuracy of 76.85%.
- Luz et al. - used a combination of phonetic and linguistic features without human intervention and obtained an accuracy of 78.87%.

Most of these earlier studies were based on features designed by experts and were unable to learn more informative and discriminative features, so a relatively poor performance was obtained.

**latest deep-learning methods**

- Mahajan et al. - POS tags and Glove as inputs on CNN-LSTM model and obtained the best accuracy of 0.6875. Then, they replaced unidirectional LSTM with bidirectional LSTM layers and obtained the best accuracy of 0.7292.
- Fritsch et al. - enhanced $n$-gram language models to create neural network models with LSTM cells, and an accuracy of 85.6% was obtained to classify HCs and AD on the Pitt dataset.
- Pan et al. - used a glove word embedding sequence as the input, combined with gated recurrent unit layers and a stacked bidirectional LSTM to diagnose AD on the Pitt dataset.
- Roshanzamier et al. - demonstrated that the combination of BERT<sub>Large</sub> and logistic regression had the best performance in the classification problem. They used the Pitt DementiaBank dataset and data augmentation technology to enhance the classification performance and obtained a SOTA accuracy of 88.08%.

In addition, many studies have used multimodal datasets to detect AD and MCI, and more accurate and differentiated information may be obtained from different models.

Overall, strong representation learning ability and discriminative classifiers, multimodal information, and transfer learning are all effective factors in the accurate diagnosis of AD and MCI.

## Methods

### Transfer Learning

The general flow of using a pre-trained model for classification consists of the following steps:

1. Training a general language model on a large dataset.
2. Fine-tuning a pre-trained language model on the target dataset.
3. Using a target-specific pre-trained language model for classification.

In this papaer, we argue that the attention mechanism allows the model to focus on some parts of the transcripts for decision-making, which is suitable for AD diagnosis because it can capture specific markers related to AD. We used a pre-trained BERT model for text embedding, which converts original sentences or transcripts  to 768-dimensional vectors.

### Overall Classification Framework

The entire model architecture in this study mainly consists of two sections: the distilBert model and the logistic regression classifier.

DistilBert Model distills the BERT base from 12 layers to 6 layers and removes token-type embeddings and poolers. It can reach 60% of the faster speed and 40% smaller architecture but retains 97% language understanding capability of the BERT model. 

Specifically, the pretrained distilBert model is used as the feature extractor, the output layer of which is replaced by a logistic regression classifier for binary classification.

![[Pasted image 20240115180117.png]]

The algorithm description of the entire process is presented below.

![[Pasted image 20240115180815.png]]

DistilBert can capture long-distance dependencies by learning the global semantic message of input text thoroughly because it has some mechanisms, such as a multi-head self attention and location code.

The process is repeated six times and a 768-dimensional semantic feature vector is obtained. The transcripts in this study are a section of the description on a picture, the maximum lengh of which is no more than 500, so the length of word embedding is set as 500, considering speed and semantic completion.

### Grid Search 

In this study, the GridSearchCV function in the scikit-learn tool, including grid search and cross-validation, is used to search for the best parameters of the logistic regerssion model. Considering speed and accuracy, the search scope of the GridSearchCV function ranges from 0.0001 to 100, and the step is set as 20.

## Experiments

### ADReSS Datasets

![[Pasted image 20240115181958.png]]

The study is a picture description task from the Diagnostic Aphasia Examination, and participants are asked to describe a picture as detailed as possible.

An example of a transcript from the dataset is shown below.

*A boy and a girl are in a kitchen with their mothers. The little boy is getting a cookie for the little girl, but he is on a stool and is about to fall. The mother is washing dishes. She is obviously thinking of something else because the water pours out over the sink. She finished with some dishes. It seems to be summer because there are bushes. The window is open. There seems to be some kind of breeze because the curtains on the sill there blow. It must be fairly hot. The mother is in a sleeveless dress. The children are in short sleeve tops and have sandals. The little boy has tennis shoes. The mother obviously is unaware of what the children are doing. She will be aware of this shortly. How much more do you want to do?*

![[Pasted image 20240115182449.png]]

The age distribution of the two groups at different intervals is presented in **Table 1.** 

![[Pasted image 20240115182529.png]]

The average values and standard deviations of age and mini-mental state examination (MMSE) scores are shown in **Table 2.**

### Experiment Results 

**Experiment environment**

- Windows 10
- Intel(R) Core I i5-6500 CPU @3.20 GHz, 3.19 GHz CPU, 44. GB RAM
- scikit-learn(logistic regression), NumPy, Pandas
- Python 3.6.13

The experiment used the accuracy, precision, recall, and F1-score as indices to evaluate the perform of the model(**==I think the Precision expression is wrong. Shouldn’t it be TN ->TP?==**).

$$
\text{Accuracy} = \frac{TN + TP}{TN + FP + FN + TP}
$$
$$
\text{Precision} = \frac{TN}{TN + FP}
$$
$$
\text{Recall} = \frac{TP}{TP + FN}
$$
$$
\text{F1-Score} = \frac{2TP}{2TP + FP + FN}
$$

![[Pasted image 20240115184839.png]]

The champion used two models, acoustic and text, and combined the ERNIE model with discriminated markers to improve representation learning(accuracy of 0.896). We modified the model architecture of the distilBert model to achieve a strong classification performance using only text(accuracy of 0.88).

![[Pasted image 20240115185748.png]]

**Table 5** shows the comparative results to check the influence of different classifiers.

## Discussion

The best performance indicates that our model has learned useful features for classification, which not only reduces the need for expert-defined linguistic features but also makes it possible for accurate, complex, and comprehensive features to be extracted from the dataset.

Whether our model is suitable for the evaluation of MMSE scores needs to be further verified.

Using automatic speech recognition (ASR)-generated transcripts directly without the need for further annotation, our method has more advantages than the manual feature extraction method.

The largest limitation of our study is the difficulty to interpret the performance of a model with so many parameters. That is, our model cannot understand the reason for a wrong verdict, but we can identify the words that the network has paid more attention to in the case of a correct prediction. This function is particularly useful because such an interpretation can reveal the important linguistic attributes of patients with AD, which can help in speech therapy and communication with patients with AD.

The practice of pre-trained and fine-tuning paradigms has achieved excellent performance in many downstream tasks. However, there are still some problems that need to be solved in large models, such as the dataset quality, huge training energy consumption, carbon emission problems, and a lack of common sense and reasoning ability. 

## Feature works 

In the future, we will focus on the following two directions for AD diagnosis.
1. Focus on implicit sentiment analysis for AD diagnosis using a contrastive learning method.
2. Commit to improving multilingual AD recognition using cross-lingual transfer learning, including the multilingual BERT and transformer models.