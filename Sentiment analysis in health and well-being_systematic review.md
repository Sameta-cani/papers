
# Sentiment Analysis in Health and Well-Being: Systematic Review

## Authors

**Anastazia Zunic, MSc**: School of Computer Science & Informatics, Cardiff University, Cardiff, United Kingdom
**Padraig Corcoran**: School of Computer Science & Informatics, Cardiff University, Cardiff, United Kingdom
**Irena Spasic, phD**: School of Computer Science & Informatics

<hr>

## Article

**Keywords**: sentiment analysis, natural language processing, text mining, machine learning

**Posted Date**: January 1th, 2020

**DOI**: https://medinform.jmir.org/2020/1/e16023

<hr>

## Introduction

This review focuses specifically on applications related to health, which is defined as “a state of complete physical, mental, and social well-being and not merely the absence of disease or infirmity”.

To establish the state of the art in SA related to health and well-being, we conducted a systematic review of the recent literature. To capture the perspective of those individuals whose health and well-being are affected, we focused specifically on spontaneously generated content and not necessarily that of health care professionals.

## Methods

### Guidelines

Our methodology is based on the guidelines for performing systematic reviews described by Kitchenham [14]. 

### Research Questions

The overarching topic of this review is the SA of spontaneously generated narratives in relation to health and well-being. The main aim of this review was to answer the research questions given in Table 1.

**Table 1.** Research questions.

| ID | Question |
| ---- | ---- |
| RQ1 | What are the major sources of data? |
| RQ2 | What is the originally intended purpose of spontaneously generated narratives? |
| RQ3 | What are the roles of their authors within health and care? |
| RQ3 | What are their demographic characteristics? |
| RQ4 | What areas of health and well-being are discussed? |
| RQ5 | What are the practical applications of SA? |
| RQ6 | What methods have been used to perform SA? |
| RQ7 | What is the state-of-the-art performance of SA? |
| RQ8 | What resources are available to support SA related to health and well-being?  |
### Search Strategy

To systematically identify articles relevant to SA related to health and well-being, we first considered relevant data sources: the Cochrane Library [15], MEDLINE [16], EMBASE [17], and CINAHL [18]. 

The search performed on January 24, 2019, retrieved a total of 299 articles.

### Selection Criteria

![img](https://github.com/Sameta-cani/papers/blob/main/imgs/Pasted%20image%2020240125141351.png)

## Results

### Data Provenance

This section discusses the main properties of data used as input for SA in relation to research questions RQ1 and RQ2.

The majority of data were collected from the mainstream social multimedia and Web-based retailing platforms, which provide the most pervasive user base together with application programming interfaces (APIs) that can support large-scale data collection.

- Twitter(TwitterR, Twitter4J, Tweepy)
- Facebook
- Instagram
- YouTube
- Reddit
- Amazon

**Table 6.** Health-related websites and networks.

| Website | Description | Used in |
| ---- | ---- | ---- |
| RateMDs[55] | Allows users to post reviews about health care staff and services. | [56-58] |
| WebMD[59] | Publishes content about health and care topics, including fora that allow users to create or participate in support groups and discussions. | [23, 60, 61] |
| Ask a Patient[62] | Allows users to share their personal experience about drug treatments. | [61, 63] |
| DrugLib.com[64] | Allows users to rate and review prescription drugs. | [23, 61, 63, 65] |
| Breastcancer.org[66] | A breast cancer community of 218,615 members in 81 fora discussing 154,832 topics. | [67, 68] |
| MedHelp[69] | Allows users to share their personal experiences and evidence-based information across 298 topics related to health and well-being. | [21,53,54,70,71] |
| DailyStrength[72] | A social networking service that allows users to create support groups across 34 categories related to health and well-being. | [23,27] |
| Cancer Survivors Network [73] | A social networking service that connects users whose lives have been affected by cancer and allows them to share personal experience and expressions of caring. | [74, 76] |
| NHS website[77] (for merly NHS Choices) | The primary public facing website of the United Kingdom’s National Health Service (NHS) with  more than 43 million visits per month. It provides health-related information and allows patients to provide feedback on services. | [78] |
| DiabetsDaily [79] | A social networking service that connects people affected by diabetes where they can trade advice and learn more about the condition. |  [80] |
The fifth i2b2/VA/Cincinnati challenge in NLP for clinical data [81] represents an important milestone in SA research related to health and well-being. The challenge focused on the task of classifying emotions from suicide notes. The corpus used for this shared task contained 1319 written notes left behind by people who died by suicide.

### Data Authors 

This section discusses the characteristics of those who authored the types of narratives discussed in the previous section. We first discuss their roles within health and care in relation to research questions RQ3 followed by their demographic characteristics in relation to question RQ4.

We identified the following 5 roles:

**Table 7**. The roles of authors with respect to health and well-being.

| Role | Description | Studies |
| ---- | ---- | ---- |
| Sufferer | A person who is affected by a medical condition. | [21,23,27,46,53,54,60,61,63,65,67,68,70,71,74-76,101,102] |
| Addict | A person who is addicted to a particular substance. | [26,103-106] |
| Patient | A person receiving or registered to receive medical treatment. | [21,23,27,46,50,53,54,56-58,60,61,63,65,67,68,70,71,74-76,78,80,102,107,108] |
| CarerCarer | A family member or friend who regularly looks after a sick or disabled person |  [23,56-58,60,61,74-76] |
| Suicide victim  | A person who has committed suicide.  | [51,82-100] |

![img](https://github.com/Sameta-cani/papers/blob/main/imgs/Pasted%20image%2020240125144250.png)

### Areas and Applications

This section focuses on the areas of health and well-being encompassed by the given datasets in relation to research question RQ5. These areas provide context for the practical applications of SA, which are discussed in relation to question RQ6.

![img](https://github.com/Sameta-cani/papers/blob/main/imgs/Pasted%20image%2020240125144611.png)

![img](https://github.com/Sameta-cani/papers/blob/main/imgs/Pasted%20image%2020240125144627.png)

### Methods Used for Sentiment Analysis

This section studies a range of methods and their implementations that have been used to perform SA in relation to research question RQ7. We also describe their classification performance to establish the state of the art in relation to question RQ8.

Traditionally, lexicon-based SA methods classify the sentiment as a function of the predefined word polarities [28,31,37,43,50]. Lexicon-based methods are the simplest kind of rule-based methods. In general, rather than focusing on individual words, rule-based methods focus on more complex patterns, typically implemented using regular expressions [85,87,88,90,93-95,100,112]. Most often, these rules are used to extract features pertinent to SA, whereas the actual classification is based on machine learning algorithms. Table 11 provides information about specific machine learning algorithms used. Specific implementations of these algorithms that were used to support experimental evaluation are listed in Table 12.

**Table 13**. Classification performance.

| Study | Algorithm | Accuracy (%) | Precision (%) | Recall (%) | F-measure(%) |
| ---- | ---- | ---- | ---- | ---- | ---- |
| [110] | SVM | 70 | - | - | - |
| [82] | SVM | - | 55.72 | 54.72 | 55.22 |
| [83] | SVM | - | - | - | 53.31 |
| [84] | SVM | - | 49 | 46 | 47 |
| [85] | SVM + CRF + rules | - | 60.1 | 36.8 | 45.6 |
| [87] | KNN, DT + SVM + rules | - | 51.9 | 48.59 | 50.18 |
| [88] | SVM + rules | - | 41.79 | 55.03 | 47.5 |
| [89] | SVM, rules | - | 53.8 | 53.9 | 53.8 |
| [90] | rules | - | 45.98 | 44.57 | 45.27 |
| [91] | SVM | - | 46 | 54 | 49.41 |
| [92] | SVM | - | 55.09 | 48.51 | 51.59 |
| [93] | NB, rules, NB + rules | - | 87.09 | 55.74 | 56.4 |
| [94] | NB + rules | - | 54.96 | 51.81 | 53.34 |
| [95] | SVM, SVM + rules | - | - | - | 50.38 |
| [96] | ME | - | 57.89 | 49.61 | 53.43 |
| [97] | SVM + rules, NB, DT | - | 56 | 62 | 59 |
| [98] | SVM + NB + ME + CRF + lexicon | - | 58.21 | 64.93 | 61.39 |
| [99] | LR | - | 51.14 | 47.62 | 49.33 |
| [78] | SVM, NB, DT, bagging | 88.6 | - | - | 89 |
| [60] | NB | - | - | - | 54 |
| [74] | AdaBoost | 79.2 | - | - | - |
| [67] | SVM, AdaBoost, ME | 79.4 | - | - | - |
| [75] | AdaBoost | 79.2 | - | - | - |
| [61] | NB, ME, rules | - | 85.25 | 65 | 73.76 |
| [63] | NB, ME | - | 84.52 | 66.67 | 74.54 |
| [25] | SVM | 88.6 | - | - | - |
| [76] | SVM, LR, AdaBoost | 79.2 | - | - | - |
| [26] | SVM, NB, LR | - | 71.47 | 66.91 | 67.23 |
| [107] | SVM, NB, DT | - | - | - | 84 |
| [114] | SVM, NB | - | 63 | 82 | 73 |
| [28] | NB, lexicon-based | - | 75.8 | 74.3 | 73 |
| [30] | CNN | 76.6 | 73.7 | 76.6 | 73.6 |
| [106] | SVM + NB | 82.04 | - | - | - |
| [32] | SVM, NB, RF | - | 68.73 | 51.42 | 58.83 |
| [33] | SVM | - | 78.6 | 78.6 | 78.6 |
| [111] | LR, DT | 75 | 76.1 | - | - |
| [38] | NB | 80 | - | - | - |
| [41] | N-gram | - | 81.93 | 81.13 | 81.24 |
| [53] | SVM, NB, RF | - | - | - | 82.4 |
| [47] | SVM, KNN, DT | - | 58 | 99 | 73 |

In summary, the performance of SA of health narratives is much poorer than that in other domains, but it is yet unclear if this is because of nature of the domain, the size of training datasets, or the choice of methods.

### Resources 

In relation to research question RQ9, this section provides an overview of practical resources that can be used to support development of SA approaches in the context of health and well-being.

![img](https://github.com/Sameta-cani/papers/blob/main/imgs/Pasted%20image%2020240125151630.png)

This refers to a lack of lexicon regarding health and well-being.

## Discussion

The overarching topic of this review is the SA of spontaneously generated narratives in relation to health and well-being.

### Conclusions

This review explores the potential for advancing research in sentiment analysis (SA) related to health and well-being. Researchers need to systematically explore and test a variety of methods, and the creation and sharing of a large, anonymized dataset are necessary. This will enable benchmarking of existing methods and exploration of new approaches, especially in deep learning. The development of domain-specific sentiment lexicons can also enhance SA performance. Although many studies have focused on the automatic construction of domain-specific sentiment lexicons, none were identified in this review. Finally, systematic collection of demographic data is required for health-related applications of SA to understand the generalizability of the findings.
