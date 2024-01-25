# Summary
## 1. Problem Definition

The primary problem addressed in the paper is the prediction of suicidal sentiments on social networks, particularly on Twitter. The challenge lies in effectively identifying tweets that express or indicate a risk of suicide, which is a significant public health concern.

## 2. Motivation

The motivation behind this research is twofold: firstly, to leverage the vast amount of data available on social networks for the benefit of mental health surveillance and intervention; and secondly, to improve the accuracy and effectiveness of suicide risk prediction using advanced computational techniques. By doing so, the researchers aim to contribute to preventative mental health strategies.

## 3. Method

The methodology involves a combination of machine learning algorithms and semantic sentiment analysis. The machine learning component utilizes Weka for classification tasks, while the semantic analysis is based on constructing a specific vocabulary associated with suicide and implementing an algorithm that compares semantic similarities between tweets. This approach leverages WordNet, a lexical database for the English language, to understand the context and sentiment of the tweets.

## 4. Experiment

The experimental setup involved collecting a dataset of tweets using Twitter4J, an API for Twitter. This dataset was then processed and analyzed using the proposed method. The effectiveness of the method was evaluated in terms of accuracy, precision, and other relevant metrics to determine how well the model could identify tweets with suicidal sentiment.

## 5. Conclusion

The conclusion of the study highlights the effectiveness of the proposed method in predicting suicidal sentiments on Twitter. The authors note that their approach demonstrates promising results in terms of accuracy and precision, suggesting it could be a valuable tool in identifying individuals at risk of suicide on social media platforms. They also discuss the potential implications for mental health professionals and the importance of further research in this area.

# Machine Learning and Semantic Setiment Analysis based Algorithms for Suicide Sentiment Prediction in Social Networks

## Authors

**Marouane Birjali**: LAROSERI Laboratory, Department of Computer Sciences, University of Chouaib Doukkali, Faculty of Sciences, El Jadida, Morocco
**Abderrahim Beni-Hssane**: LAROSERI Laboratory, Department of Computer Sciences, University of Chouaib Doukkali, Faculty of Sciences, El Jadida, Morocco 
**Mohammed Erritali**: TIAD Laboratory, University of Sultan Moulay Slimane, Faculty of Sciences and Technologies, Béni Mellal, Morocco

<hr>

## Article

**Keywords**: Sentiment Analysis; Machine Learning; Suicide; Social Networks; Tweets; Semantic Sentiment Analysis

**Posted**: The 8th International Conference on Emerging Ubiquitous Systems and Pervasive Networks (EUSPN 2017)

**DOI**: https://www.sciencedirect.com/science/article/pii/S187705091731699X

<hr>

## Abstract

This paper addresses the new challenge of sentiment analysis in social networks, specifically focusing on the issue of suicide. It proposes a method to address the lack of terminological resources related to suicide by creating a vocabulary associated with it. The study also utilizes Weka, a data mining tool based on machine learning algorithms, to extract useful information from Twitter data collected via Twitter4J. The research includes an algorithm for semantic analysis between tweets using WordNet, and the experimental results demonstrate the effectiveness of this approach in predicting suicidal thoughts using Twitter data.

## 1. Introduction

The purpose of this paper is to propose a method of predicting suicidal ideas, to predict suicidal acts and ideas using data collected from social.

We present an algorithm that extracts information using Weka as a data mining tool and calculates the semantic similarity between tweets collected from Twitter as a training set based on semantic analysis resources using WordNet.

## 2. Related Works

In this work, we compute the semantic similarity of a tweet collected from Twitter and training data. In this context, we focus on pessimistic, bad thoughts and thinking through suicide. We evaluate our method and present the results to predict the semantic orientations of Twitter data.

## 3. Methodology of Suicidal acts analysis 

The methodology needs of this work can be divided into four main parts:
1. needs related to the manual construction of the vocabulary associated with suicide theme
2. collection of Twitter data
3. needs related to automatic classification using machine learning algorithms implemented in Weka
4. requirements for a semantic analysis of these sentiments to improve our results.

![[Pasted image 20240118185142.png]]

### 3.1. Vocabulary associated with Suicide act

1. Defining a manual vocabulary associated with the various themes related to suicide (fear, depression, harassment, etc.). 
2. Subsequently, this vocabulary is divided into several categories and sub-categories in order to easily determine the degree of threat of each tweet.

### 3.2. Data collection 

![[Pasted image 20240118185607.png]]

In this work, we collect the tweets using the search word defined in the part of the manual vocabulary. The treatment mechanism is used to collect sequential data from Twitter and the extraction is performed several times for more tweets. The token and access keys obtained, are necessary for the extraction of the tweets in real time from the Twitter page.

### 3.3 Machine learning classification 

 In this work, the training tweets is a set of tweets already affected by the act of suicide and built manually, it groups input vectors and their corresponding classes. From this training set, a classification model is defined which allows to classify the input characteristics vectors in the corresponding classes.

For the case of our work, we propose to study the textual content of tweets, especially associated with user profiles to accurately classify user profiles, so we do not consider the tweet of a particular user. On the other hand, we have targeted our tweets associated with vocabulary to consider the particular user.

### 3.4 Semantic analysis measure based on WordNet 

Based on WordNet database, where each term is associated with others.

The main task is to use the stored documents that contain terms and then check the similarity with the words that the user uses in their sentences.

The reason for using a semantic analysis in this subsection is to establish the semantic meaning of the new tweets set in relation to training tweets by using the semantic meaning of the elements (words) of each tweet.

The aim of this subsection is to implement the proposed algorithm of semantic analysis of tweets that can advance our research. The principal contribution is to propose an algorithm for computing the semantic similarity between training tweets and the new test tweets collected by Twitter4J, using WordNet as an external network semantic resource. In addition, this contribution will based on Leacock and Chodorow approache. This approach based on the combination of the method of informational content and the counting of the arcs method. In fact, the semantic measure proposed by Leacock and Chodorow is based on the shortest length between two syntaxes of WordNet. This measure is defined by the formula:

$$
Sim_{lc}(A, B) = -\text{log}(\frac{cd(a, b)}{2 * L})
$$

$L$: longest length, which separates the concept root, of ontology, of the concept more in bottom
$(A, B)$: shortest length that separates A of B.

To compute the semantic similarity between words of new tweets set and words of the tweets training set, we apply the following algorithm:

![[Pasted image 20240125125357.png]]

The proposed semantic analysis measure is presented by the following formula:

$$
Sim_{lc}(a, b) = \frac{\sum_{i=1}^n\sum_{j=1}^ma_i*b_j*Sim(i, j)}{\sum_{i=1}^n\sum_{j=1}^ma_i*b_j}
$$

$i$: represents the terms of the training tweets of b.
$j$: represents the terms of the new tweet a.
$qi$: frequency occurrence of the term i in training tweets q.
$dj$: frequency occurrence of the term j in new tweet a.
$Sim(i, j)$: semantic similarity measure between the two terms of new tweets i and training tweets j.

## 4. Experimental results and Analysis 

Use our algorithm to compute the semantic similarity between the new collected tweets and the training tweets already affected by the act of suicide.

The percentage of suspicious tweets at risk of suicide and tweets suspect to risk compared to suspect tweets(using the Weka tool).

![[Pasted image 20240125130414.png]]

![[Pasted image 20240125130626.png]]

![[Pasted image 20240125130813.png]]

![[Pasted image 20240125130836.png]]

## 5. Conclusion 

As part of this work, we present our potentially method based on different machine learning for using the social network Twitter as a preventive force in the fight against suicide. In addition, our work can analyze semantically the Twitter data based on WordNet.