# Automatic Detection of Psychological Distress Indicators and Severity Assessment from Online Forum Posts

## Authors

**Shirin Saleem**: Raytheon BBN Technologies, 10 Moulton St, Cambridge, MA, U.S.A.

**Rohit Prasad**: Raytheon BBN Technologies, 10 Moulton St, Cambridge, MA, U.S.A.

**Maciej Pacula**: Raytheon BBN Technologies, 10 Moulton St, Cambridge, MA, U.S.A.

**Michael Crystal**: Raytheon BBN Technologies, 10 Moulton St, Cambridge, MA, U.S.A.

**Brian Marx**: National Center for PTSD at VA Boston Healthcare Sytem, Boston, MA, U.S.A., Boston University School of Medicine, Boston, MA, U.S.A.

**Denise Sloan**: National Center for PTSD at VA Boston Healthcare Sytem, Boston, MA, U.S.A., Boston University School of Medicine, Boston, MA, U.S.A.

**Jennifer Vasterling**: National Center for PTSD at VA Boston Healthcare Sytem, Boston, MA, U.S.A., Boston University School of Medicine, Boston, MA, U.S.A.

**Theodore Speroff**: VA Tennessee Valley Healthcare System, Nashville, TN, U.S.A., Vanderbilt University School of Medicine, Nashville, TN, U.S.A.

<hr>

## Article 

**Keywords**: Psychological Distress, Web forums, Text classification, Annotator rationales, Support Vector Machines, Probabilistic Logic, Markov Logic Networks.

**Posted**: COLING 2012, Mumbai, December 2012.

**Original**: https://aclanthology.org/C12-1145.pdf

<hr>

## Abstract

This paper describes a novel system that analyses forum posts to: 
1. detect distress indicators that directly map to the Diagnostic and Statistical Manual of Mental Disorders (DSM) IV constructs 
2. assess the severity of distress for prioritizing individuals who should seek clinical help (i.e. triage).

Improvements in multi-label classification accuracy using human-generated rationales in support of annotated distress labels.

For triage assessment, we demonstrate the effectiveness of Markov Logic Networks (MLNs) in dealing with noisy distress label detections and encoding expert rules.

## 1 Introduction

Web-forum discussions of symptoms, thoughts and experiences are open, descriptive, and honest, making them an ideal source for observing communications of individuals for assessing psychological status. 

In this paper, we present a multi-stage text classification system for assessing psychological status of individuals based on their text postings on online web forums. Specifically, our system combines state-of-the-art NLP and machine learning techniques to: 
1. extract fine-grained psychological distress indicators/labels derived from Diagnostic and Statistical Manual of Mental Disorders (DSM) IV (American Psychiatric Association, 2000)
2. assesses the severity of distress that can be used to triage individuals who should seek clinical help

### 1.1 Previous Work

In this paper, we focus on noisy and informal text messages that occur in web forums. The work presented in this paper is the first to assess psychological states through web forum texts.

Several rule-based approaches have been studied to detect PTSD and mTBI in clinical narratives, but these approaches are highly dependent on annotations, lack consistency, and require expert knowledge. Therefore, we use a statistical model that encodes domain knowledge by learning weights for domain rules from the data.

### 1.2 Novel Contributions

1. We describe a suite of features and classifiers trained on expert-annotated text to detect distress indicators. $\rightarrow$ relative improvement of 14.6% over using plain text features.
2. Use of Markov Logic Networks (MLNs) to incorporate domain-specific rules, and handle the inherent noise in the data. $\rightarrow$ improve the triage classification accuracy, and provide a robust approach for inferring triage codes from noisy distress label detection as well as potentially contradictory domain rules.

## 2 Corpus for Experimentation

Our corpus consists of threads downloaded from an online forum for veterans with post-combat psychological issues.

In consultation with psychologists, a codebook of 136 psychological distress labels spanning PTSD, mTBI, and depression symptoms was developed. Codes/labels were mostly derived from the DSM-IV guidelines (American Psychiatric Association, 2000).

![[Pasted image 20240118150443.png]]
Expert psychologists next annotated each author in a thread with a triage code that indicates treatment acuity or the priority assigned to a referral for additional treatment.
- TR1: Indicating current or imminent danger to self or others $\rightarrow$ emergency intervention or urgent care evaluation
- TR2: Indicating behavioural disturbances, distress, functional impairment and/or suicidal/homicidal ideation without any imminent danger to self or others $\rightarrow$ non-urgent treatment referral
- TR3: There is no evidence of current behavioural disturbance, distress or functional impairment. $\rightarrow$ no recommendation for treatment

**TR1 is rarely observed in online forums because sensitive content is moderated and deleted. Therefore, this paper is limited to distinguishing between TR2 and TR3, but can also be extended to TR1 detection.** 

## 3 Approach Overview

![[Pasted image 20240118151546.png]]

## 4 Multi-label Distress Classification 

### 4.1 Classifier

problem transformation methods using binary one-versus-all Support Vector Machines (SVMs) that detect the presence or absence of each of the fine-grained distress labels.

**Why?** Given the large size of our label set (118 observed labels out of 136 total), we could not find a memory-efficient way to use many of the algorithm adaptation methods.

### 4.2 Features 

 In our experiments, we explored a variety of features that look beyond the identity of the words in the message. These include message-level features computed based on the content of individual messages as well as threadlevel features that exploit the structure of the discussion thread and look at other messages in the thread. In all cases, the features are binary, integer, or real valued

## 5 Psychological Triage Models for Severity Assessment

Goal: find authors who might require treatment or medical evaluation based on any behavioural disturbances, distress, functional impairments and/or suicidal or homicidal ideation. 

We explored **two approaches** to address this problem:
1. Uses an SVM trained on the words and predicted distress labels for the messages posted by the author. 
2. Uses Markov Logic Networks (MLNs) to encode domain knowledge using probabilistic first order rules with associated weights.

In our system, the **MLN computes the probability** of a triage code using: 
1. the distribution of words in the messages posted by an author 
2. the predicted distress labels 
3. domainspecific rules that encode dependencies between the text, distress labels and the triage.

MLNs have two key **advantages** for our application:
1. the use of statistical inference provides robustness to noise in the text and label predictions, and potential contradictions in the domainspecific rules.
2. the relative weights for the domain-specific rules can be automatically learned from the training data.

### 5.5 Alchemy 

An implementation of learning and inference algorithms for MLNs, for our experiments. 

To learn the weights of the domainspecific rules, we used discriminative training, which maximizes the conditional likelihood of target labels (in our case the triage codes) given the observed variables (in our case the message words and distress labels). 

Alchemy uses an approach referred to as pre-conditioner scaled conjugate gradient for discriminative weight learning. 

The inference is performed using MaxWalkSAT

## 6 Experimental Results

### 6.1 Inter-annotator Agreement

We measured inter-annotator agreement between multiple annotators using the Fleiss Kappa statistic, and the measured value was 0.71, indicating good agreement.

### 6.2 Multi-label Distress Classification 

![[Pasted image 20240118155426.png]]

SVM parameters were tuned based on 10-fold cross-validation on the training set were threads were randomly distributed across 10 different subsets.

 For our experiments with SVMs, we used the Weka machine learning software (Hall et. al, 2009) with the Radial Basis Function (RBF) kernel. We performed gridsearch to find the best regularization (C) and gamma (g) parameters on the cross-validation set. For the baseline experiment with SVMs, each message was treated as a bag of words with normalized (TF-IDF) frequencies. Next, the remaining features described in section 4.1 were incrementally added to the baseline feature set of the SVM classifier.

![[Pasted image 20240118160223.png]]

**We found that our approach(23.5 AUC) of using the rationales by extracting label specific domain phrase features out-performed the contrastive approach(22.3 AUC) in (Zaiden et. al, 2008).**

![[Pasted image 20240118160418.png]]

It is to be noted that the dataset has a high class imbalance. Hence, a large number of labels perform poorly merely due to the lack of sufficient training data.

We demonstrated this in the inter-annotator agreement study where we found only moderate agreement between annotators in the coding of these distress labels.

### 6.3 Triage Assessment

![[Pasted image 20240118160842.png]]As can be observed, MLNs provide statistically significant gains over SVMs by using domain-specific rules for combining information from text as well as the distress label detections.

![[Pasted image 20240118160941.png]]

## 7. Conclusions and Future Work

Incorporating rationales from domain experts for the label annotations helps improve the multi-labeling performance, and presented a novel feature to exploit the rationale annotations. 

**Future Work**
- How to Take Advantage of Label Dependencies
- Contextual features for classification that exploit information from previous messages within a threads 
-  Validate the system on text data from subjects diagnosed with PTSD and compare the outcomes on a control group that does not suffer from PTSD