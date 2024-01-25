# Application of Machine Learning Methods in Mental Health Detection: A systematic Review

## Author

**ROHIZAH ABD RAHMAN**: Center for Artificial Intelligence Technology, Faculty of Information Science and Technology, Universiti Kebangsaan Malaysia, Bangi 43600, Malaysia

**KHAIRUDDIN OMAR**: Center for Artificial Intelligence Technology, Faculty of Information Science and Technology, Universiti Kebangsaan Malaysia, Bangi 43600, Malaysia

**SHAHRUL AZMAN MOHD NOAH**: Center for Artificial Intelligence Technology, Faculty of Information Science and Technology, Universiti Kebangsaan Malaysia, Bangi 43600, Malaysia

**MOHD SHAHRUL NIZAM MOHD DANURI**: Department of Computer Science, Faculty of Science and Information Technology, Kolej Universiti Islam Antarabangsa Selangor, Kajang 43000, Malaysia

**MOHAMMED ALI AL-GARADI**: Department of Radiology, University of California, San Diego, CA 92093, USA

<hr>

## Article

**Keywords**: Machine learning, Feature extraction, Systematics, Stress, Mental disorders, Data mining, Social network services

**Posted Date**: October 6th, 2020

**DOI**: https://ieeexplore.ieee.org/abstract/document/9214815

<hr>

## Abstract 

본 논문은 온라인 소셜 네트워크(OSN)를 정신 건강 진단에 활용하는 효과를 비판적으로 평가합니다. 주로 2007년부터 2018년까지 발표된 논문들을 검토하며 기계 학습 기술과 데이터 분석 방법에 중점을 두고 있습니다. 이 연구는 OSN이 정신 건강 문제의 조기 감지를 위한 유망한 데이터 소스임을 강조하며, 이 분야에서의 혁신적인 알고리즘과 계산 언어학의 필요성, 그리고 정신 건강 전문가들과의 협력의 중요성을 강조하고 있습니다.

## 1. Introduction

온라인 소셜 네트워크(OSN)는 정신 건강 문제 감지 방법을 개발하기 위해 활용할 수 있는 방대한 양의 정보를 생성할 수 있습니다.

OSNs에서의 현재 정신 건강 감지에 대한 분석은 데이터 집합, 데이터 분석 방법, 특징 추출 방법, 분류기 성능(정확도 및 효율성), 도전 과제, 제한 사항 및 향후 연구를 이해하기 위해 필요합니다.

이 체계적인 검토의 목적은 OSNs에서 추출된 데이터를 기반으로 한 정신 건강 문제 감지에 대한 비판적인 평가 분석을 수행하는 것입니다. 이는 OSNs에서의 정신 건강 문제 감지의 능력을 탐구하고 그 도전 과제, 제한 사항 및 향후 연구를 포함하여 조사하는 것을 목표로 합니다.

## 2. Methods

### A. Identification and selection of studies

대부분의 정신 건강 문제 감지 연구에서 OSNs를 데이터 원천으로 사용한 연구들이 선정 대상에 포함되었습니다. 현재 연구는 이전 연구자들이 정신 건강 문제 감지에서 OSNs를 데이터 소스로 활용한 방법을 설명합니다.

- PRISMA의 지침을 준수했습니다.
- 전자 문헌 검색을 수행하였습니다.
- 영국 국립 기관에서 정의한 일반적인 정신 건강 질환 키워드를 사용하였습니다.
- 정신 건강 관련 주요 용어가 문헌에 포함되어 있는지 확인하기 위해 의학 주제어 목록(Medical Subject Headings)을 참고하였습니다.

$\rightarrow$ 22 편의 논문이 적합한 연구로 검토되었습니다.

![img](https://github.com/Sameta-cani/papers/blob/main/imgs/Pasted%20image%2020240118021614.png)

### B. Methodological quality assessment

Critical Appraisal Skills Program (CASP) 체크리스트를 채택하였습니다. 각 연구의 주요 특징과 한계를 분석하고 비교하여 각 연구의 장점과 약점을 나타냈습니다.

주요 특징과 한계는 다음을 기반으로 합니다:
- **데이터 추출**: 데이터 원천, 키워드, 데이터 추출 기간 및 지리적 위치
- **데이터의 질**: 정신 건강 문제와 관련된 데이터 집합
- **연구 디자인**: 적절한 방법론 적용
- **결과**: 명확한 연구 목표 및 결과

## 3. Results

### A. Finding and selecting studies

총 2770 편의 논문 중에 중복, 무관련성 및 **표 1**을 적용한 기준에 따라 22 편의 논문이 선택되었습니다.

![[Pasted image 20240118021614.png]]

**표 2**에 제시된 대로, 논문들의 주요 특성 및 내용 분석 개요가 논의되었습니다.

선택된 논문들의 개요는 데이터 집합 (데이터 원천, 키워드, 기간 및 지리적 위치), 데이터 분석 방법, 연구 목표, 특징 추출 방법, 기계 학습 기술 및 분류기 성능과 일치합니다.

### B. Description of selected studies

<font color="#245bdb">TABLE 2.</font> Studies related to the detection of mental health in OSNs.

|                  Author/s                  | Mental<br>Health<br>Types             | Data Source<br>(OCNs)                                                                                    | Keywords                                  | Durations                            | Geo-Location             |                                                                                                                                                                                Data Set                                                                                                                                                                                 |
|:------------------------------------------:| ------------------------------------- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ------------------------------------ | ------------------------ |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|          Lin et al. 2017<br>\[9]           | Stress                                | The authors collected 350 million tweets from Sina Weibo.                                                | The authors used tags and comments.       | Oct 2007-Oct2012                     | China                    |                                                                                                                         Four data set with the first consist of 19,000 tweets (with stress label) and 17,000 tweets (with non-stressed label).                                                                                                                          |
|        Kandias et al. 2017<br>\[14]        | Stress                                | Data extracted from Facebook.                                                                            | Not specified                             | Not specified                        | Greece                   |                                                                                          The data set was generated from Facebook users who provided their informed consent (405 fully crawled, 12.346 groups, 98.256 liked objects, 171,054 statuses, and 250.027 comments).                                                                                           |
|           Thelwall 2017<br>\[17]           | Stress                                | 3066 English tweets                                                                                      | Keywords from a variety of sources        | 1 month (July 2015)                  | Not specified            |                                                                                                                                 The data collected from 3000 stress-related tweets and classified under stress level with scale 1 to 5.                                                                                                                                 |
|          Shuai et al. 2017 \[20]           | Mental Disorders                      | Data were collected among 3126 OSNs users via Amazon's MTurk.                                            | Not specified                             | Not specified                        | Not specified            |                                                                            3126 OSNs users (1790 males and 1336 females); 389 users were labeled with social network mental disorders, 246 had "Cyber Relationship Addiction", 267 had "Information Overload", and 73 had "Net Compulsion".                                                                             |
|          O'Dea et al. 2015 \[21]           | Suicide                               | 14,701 suicide-related tweets collected from Twitter.                                                    | Several keywords were derived from \[47]. | February 18 and April 23, 2014       | Not specified            |                                                                                                             The total is 1820 tweets. Set A consist 829 tweets (training: 746, testing: 83) and Set B consist 991 tweets (Training: 891, and Testing: 100).                                                                                                             |
|           Lin et al. 2014 \[22]            | Stress                                | 600 million tweets collected from Sina Weibo.                                                            | Not specified                             | Oct 2009-Oct2012                     | China                    |                                                                                                                              600million tweets with five categories of stress such as affection, work, social, physiological, and others.                                                                                                                               |
|         Tsugawa et al. 2015 \[23]          | Stress                                | 3200 tweets collected from Twitter.                                                                      | Not specified                             | December 4, 2013 to February 8, 2014 | Japan                    |                                                                                                An experiment conducted with 219 participants. Only 214 participants were involved. Data from the participants (males: 121; females: 88) aged 16-55 years were analyzed.                                                                                                 |
|          Saleem et al. 2012 \[24]          | Distress                              | Data collected from web fora related to psychological issues.                                            | Not specified                             | Not specified                        | United States            |                                                                                                        136 psychological distress labels raning from PTSD to mild traumatic brain injury and depression symptoms developed via consultation with psychologists.                                                                                                         |
|       De Choudhury et al. 2013 \[26]       | Depression                            | 1583 data of crowd workers (crowdsource: MTurk) who shared their Twitter public profile                  | Not specified                             | September 15 to October 31, 2012     | United States            |                                                                                                                 637 participants provided access to their Twitter feeds. Subsequently, 476 users (243 males and 233 females) diagnosed with depression.                                                                                                                 |
|        Deshpande and Rao 2017 \[27]        | Depression                            | 10,000 tweets collected from Twitter.                                                                    | Words related to "poor mental well-being" | Not specified                        | Not specified            |                                                                              10,000 tweets collected to generate training and test datasets with a ratio of 80:20. The training set consists of words that suggest depression tendencies, such as "depressed", "hopeless", and "suicide".                                                                               |
|          Huang et al. 2014 \[25]           | Suicide                               | 53 verified suicidal users and over 30,000 posts from Sina Weibo. They obtained 614 true suicidal posts. | Not specified                             | Not specified                        | China                    |                                                                                                 614 suicidal posts obtained. To perform a 90 to 10 test, 6140 posts randomly selected from the set of non-suicidal users, for 6754 posts. Finally, 6704 posts obtained.                                                                                                 |
|      Wang, Zhang, and Sun 2013 \[28]       | Depression                            | Data collected from Sina Weibo.                                                                          | Not specified                             | August 1-15, 2012                    | China                    |                                                                                                                                                                  The data set was derived from \[48].                                                                                                                                                                   |
|           Xue et al. 2014 \[29]            | Pressure                              | The tweets of 459 middle school students (aged 14-20 years) collected from Sina Weibo.                   | Not specified                             | July 7, 2013                         | China                    |                                                                                                                               23 teenagers posted 300 to 1000 tweets and 10,872 tweets. The average number of tweets is 473 per teenager.                                                                                                                               |
|           Saha et al. 2016 \[30]           | Depression                            | Data crawled from 620,000 posts made by 80,000 users inn 247 online communities.                         | Not specified                             | Not specified                        | Not specified            |                                                                                                                                              Data from the Live Journal website contains 620,060 posts from 78,647 users.                                                                                                                                               |
| Wongkoblap, Vadillo, and Curcin 2018 \[31] | Depression                            | Data of "myPersonality" crawled from Facebook and derived from \[49].                                    | Not specified                             | Not specified                        | Not specified            |                                                                                                     Two datasets: 1) self-reported results from Satisfaction with Life Scale, and 2) self-diagnosed results from volunteers who answered the CES-D questionnaires.                                                                                                      |
|           Luo et al. 2018 \[32]            | Suicide                               | Data collected from 716,899 tweets from Twitter                                                          | Suicide-related terms                     | January-November 2016                | Not specified            |                                                                                                                                                                 191,473 precise suicide-related tweets                                                                                                                                                                  |
|           Tai et al. 2015 \[33]            | Mental Disorder                       | Data collected through online social platforms.                                                          | Related to depression and being healthy.  | Not specified                        | Not specified            |                                                                                                              Data from the depressed (keywords: bad, hate, and hard) and healthy (keywords: love, best, and hope) groups based on the extracted keywords.                                                                                                               |
|         Saravia et al. 2016 \[34]          | Mental Illness                        | Data collected from Twitter.                                                                             | Related to BPD and BD.                    | Not specified                        | Not specified            |                                                                                                  Data collected from community portals consist of 17 BPD and 12 BD. Each portal has 5000 followers in 145,000 accounts. After filtering, 278 BD and 203 BPD remained.                                                                                                   |
|    Chang, Saravia, and Chen 2016 \[15]     | Mental Disorder                       | Subconscious crowdsourcing.                                                                              | Keywords related to BPD and BD            | Not specified                        | Not specified            |                                                                                                            Data were collected from community portals consist 12 BD and 17 BPD. Then, 5000 followers downloaded from each portal with 145,000 user accounts.                                                                                                            |
|            Li et al. 2016 \[16]            | Stress                                | Data extracted from a microblog.                                                                         | Not specified                             | January 1, 2012 to February 1, 2015  | China                    | **Set 1**: Stressor event (273 from 1 January 2012 to 1 February 2015 related to study; 122 related events, such as examination, contest, and result notification). <br>**Set 2**: Post (124 students who actively used Tencent Weibo. Post from 1 January 2012 to 1 February 2015, 29,232 posts. The average post is 236, the maximum is 1378 and the minimum is 104.) |
| Coppersmith, Harman, and Dredze 2014 \[18] | Post-Traumatic Stress Disorder (PTSD) | 3200 tweets collected from Twitter.                                                                      | Related to PTSD                           | Not specified                        | United States (military) |                                                                                                                                    260 tweets indicated a diagnosis of PTSD. After filtering, only 244 users were positive samples.                                                                                                                                     |
|           Park et al. 2012 \[19]           | Depression                            | 65 million tweets collected from Twitter.                                                                | Related to depression                     | June to July 2009                    | United States            |                                                                         21,103 tweets consist the word "depression" from 14,817 users. 1000 of random tweets (500 tweets from each month) selected for content analysis. 165 participants and 69 participants (male=28, female=41) are active.                                                                          |

### C. Feature extraction

<font color="#245bdb">TABLE 3.</font> Feature extraction methods used in supervised machine learning studies.

| Author/s | Feature Extraction Method |
| ---- | ---- |
| Lin et al. 2017 \[9] | This study used LIWC2007 in categorizing to positive or negative emotion words. |
| Kandias et al. 2017 \[14] | The feature selection using frequency of TF-IDF and term occurrence. |
| Thelwall 2017 \[17] | All the features used were labeled unigrams, bigrams, and trigrams. |
| O'Dea et al. 2015 \[21] | This research used the basic features of word frequencies or unigrams and TF-IDF instead of simple frequency. |
| Lin et al. 2014 \[22] | This study used LIWC2007 for linguistic features. |
| Tsugawa et al. 2015 \[23] | This study used bag-of-words for word frequency and LDA for topic models. |
| Saleem et al. 2012 \[24] | The article adopted bag-of-words, unigrams, and TF-IDF. LIWC used for linguistic features. |
| De Choudhury et al. 2013 \[26] | This study used LIWC to determine 22 specific linguistic styles. |
| Deshpande and Rao 2017 \[27] | This research used part-of-speech (e.g., adjective, noun, and verb) tags and bag-of-words. |
| Huang et al. 2014 \[25] | This study used N-gram features (unigram, bigram, and trigram) and classified emotional words to positive or negative from a lexicon. Three types of part-of-speech tags (i.e., adjective, noun and verb) adopted. |
| Tai et al. 2015 \[33] | This research considered the unigram word feature and an LIWC lexicon to determine PTSD user. |
| Saravia et al. 2016 \[34] | This study adopted TF-IDF to model the linguistic features of patients and their pattern of life features of patients and their pattern of life features (e.g., age and gender). Then, TF-IDF was applied to unigrams and bigrams collected from all the patients' tweets. |
| Chang et al. 20116 \[15] | In this study, TF-IDF applied to unigrams and bigrams (calculated the frequency-of-word sequences) and LICW for linguistic features. |
| Coppersmith  et al. 2014 \[18] | LICW used to determine the linguistic style of users with PTSD. |

### E. Machine learning techniques 

<font color="#245bdb">Table 4.</font> Classifier performance of studies using supervised machine learning techniques.

| Authors | Objectives | Method of Data Analysis | Machine Learning/Deep Learning Technique | Classifier Performance |
| ---- | ---- | ---- | ---- | ---- |
| Lin et al. 2017 \[9] | This research proposed a hybrid model for detecting stress by using user content and social interaction in Twitter. | A hybrid model to analyze the data and compared with others mahcine/deep learning techniques. | - Hybrid (FGM + CNN)<br>- LR<br>- SVM<br>- RF<br>- Gradient-boosted DT<br>- DNN | The **hybrid model** (FGM + CNN) achieved the highest detection performance, i.e., an improvement of 6%-9% in the $F_1$ score. |
| Kandias et al. 2017 \[14] | This research study the stress level chronicity experienced based on the content posted by the OSNs users. | Content classification was analyzed using machine learning techniques. | - Multinomial NB<br>- SVM<br>- Multinomial LR | **SVM** was selected because it achieved more than 70% (precision, recall, and F score) and better F score values in most categories. |
| Thelwall 2017 \[17] | This research provided a stree and relaxation detection system posted in OSNs. | A new method called TensiStrength compared with machine and deep learning techniques. | - TensiStrength<br>- AdaBoost<br>- SVM<br>- NB<br>- J48 tree<br>- JRip rule<br>- LR<br>- DT | **TensiStrength** can possible to detect expressions of stress and relaxation with accuarcy level through user post in Twitter. |
| Shuai et al. 2017 \[20] | This research proposed a framework that can accurately identify potential cases of social network mental disorders. | A new method called SNMDD | - TSVM | **SNMDD** is a new method that possibility to identified a mental disorders' users through OSNs. |
| O'Dea et al. 2015 \[21] | This research examined the suicide level. | Data compared with human coding and machine learning techniques. | - LR<br>- SVM | **SVM** with TF-IDF and without filter was the best performing algorithm. |
| Lin et al. 2014 \[22] | This research proposed a method of stress detection through OSNs. | The primary model, called deep sparse neural network, compared with SVM and ANN while implmenting an Auto-Encoder. | - Deep Sparse Neural Network<br>- SVM<br>- ANN | **SVM** achieved good accuracy in linguistic attributes. **Deep Sparse Neural Network** achieved better results in social, linguistic, and visual attributes. |
| Tsugawa et al. 2015 \[23] | This research introduced a method that can recognize depression through user's activities in social media. | Models predict the risk of depression with several features obtained from user activities in Twitter using SVM. | - SVM | **SVM** achieved an accuracy of approximately 70%. |
| Saleem et al. 2012 \[24] | This research proposed a novel technique with a multistage text classification framework to assess psychological status in web fora. | SVM used for distress detection, and MLN was used for noisy distress label detection. | - SVM<br>- MLN | **MLN** provided statistically significant gains over SVM. |
| De Choudhury et al. 2013 \[26] | This research explored the potential of using OSNs to detect depressive user. | SVM used to predict depressed users. | - SVM | **SVM** classifier could predict depression with promissing results of 70% classification accuracy. |
| Deshpande and Rao 2017 \[27] | This research applied NLP to Twitter feeds for emotions related to depression. | SVM and NB classifier used for the class predction process. | - SVM<br>- NB | The result showed that **NB** gained an $F_1$ score of 83.29, while **SVM** scored 79.73. The precision and recall results were similar. The accuarcy of NB was 83% and that of SVM was 79%. |
| Huang et al. 2014 \[25] | This research proposed a real-time system to detect suicidal ideation users. | Data compared with machine learning techniques. | - SVM<br>- RF<br>- J48 tree<br>- LR<br>- Sequential minimal optimization<br> | **SVM** exhibited the best performance, with 68.3% (F-meausre), 78.9% (precision) and 60.3% (recall). |
| Wang, Zhang, and Sun 2013 \[28] | This research proposed a model to detect depressive users based on node and linkage features. | DT used as the classifier in a node feature only model. | - DT | The **DT** classifier with node features achieved the highest accuracy was 95% with increased to 15%. The addition of linkage features resulted in a considerably better performance than that with only node features. |
| Xue et al. 2014 \[29] | This research analyzed a psychological pressure experienced by adolescents by collecting data from a microblog. | Data compared with machine learning techniques. | - NB<br>- SVM<br>- ANN<br>- RF<br>- Gaussian process | The **Gaussian process** classifier achieved the highest detection accuarcy. |
| Wonkoblap, Vadillo, and Curcin 2018 \[31] | This research explored the relationship between life satisfaction and depression through OSNs. | Data compared with machine learning techniques. | - SVM with RBF<br>- LR<br>- DT<br>- NB | The accuracy of **SVM** with RBF kernel was the best model achieved to 68%. |
| Saravia et al. 2016 \[34] | This research proposed a novel data collection to build a predictive models base on users linguistic and behavioral patterns. | Data analyzed using an RF classifier. | - RF | The precision of **RF** was 96% for BPD and BD. |
| Chang, Saravia, and Chen 2016 \[15] | This research developed a model to determine mental disorders user base on users linguistic and behavioral patterns. | Data analyzed using an RF classifier. | - RF | The preformance of stressor event detection using **RF** was approximately 13.72% (precision), 19.18% (recall), and 16.50% ($F_1$ measure). |

## 4. Discussion and challenges 

**Discussion**
- 첫째, 정신 질환에 관한 연구 중에서 유용한 정보를 찾은 것은 소수 뿐이었으며, 정신 건강 문제를 가진 사람들이 자신을 고립시키고 다른 사람들과 의사 소통을 하지 않는 것과 관련된 내용이었습니다.
- 둘째, 연구자들은 OSNs에서 데이터 분석 중 정신 건강 문제 감지와 관련된 다른 언어 사용과 관련된 추가적인 발견을 하였습니다.

**Challenges**
- 비대면 의사 소통과 인간-컴퓨터 상호작용
- 언어 장벽
- 계정 개인정보 정책

### A. Quality of data sets and model interpretation

자체 데이터셋을 사용하는 장점은 정보가 특정 위치와 연구 목적에 특화되어 있을 수 있다는 것입니다. 그러나 이로 인해 편향성이 도입될 수도 있습니다.

이러한 모델을 해석하는 것은 어려울 수 있으며, 여러 복잡한 통계적 패턴이 다양한 특징을 통해 작은 지표와 관련될 수 있기 때문입니다.

### B. Mental health problem detection over time

흥미로운 동시에 어려운 작업 중 하나는 시간이 지남에 따른 정신 건강 문제 감지입니다. 다른 텍스트 분류 작업과는 달리 정신 건강 상태는 시간에 따라 상당히 다양할 수 있습니다. 예를 들어, OSNs 웹사이트에서 보고된 정신 건강 사례는 간단한 정신 건강 문제(즉, 약한 신호)로 시작하여 자살 사건(즉, 강한 신호)으로 끝날 수 있습니다. 따라서 시간이 지남에 따라 변화하는 다양한 정신 건강 시나리오를 고려해야 할 때 머신 러닝 모델을 구축해야 합니다. 모델은 약한 신호를 감지하고 시간이 지남에 따라 지속적으로 진화하는 정신 건강 감지 사례를 효과적으로 처리해야 합니다.

### C. Multicategories of mental health problems

정신 건강 문제를 분류하는 것은 연구자들이 다양한 특징 선택 과정을 수행해야 하기 때문에 어려운 작업입니다. 결과적으로 연구자들은 정신 건강 문제의 범주를 일반화하거나 구체화하기를 선택합니다.

### D. Data preprocessing

새로운 데이터 집합을 준비하는 것은 정신 건강 문제 감지에서의 한 가지 어려움입니다.

### E. Length of posts 

한정된 글 길이로부터 정신 건강 문제를 이해하는 것은 정신 건강 문제 감지를 위한 머신 러닝 모델 개발 시 고려해야 할 중요한 도전 중 하나입니다.

### F. Multilingual content

다국어 텍스트의 분석은 앞으로 다루어져야 할 흥미로운 도전 과제입니다.

### G. Data sparsity

데이터 희소성 문제에 대처하는 것은 머신 러닝 모델의 성능에 부정적인 영향을 미칠 수 있기 때문에 중요합니다.

### H. Publicy available data sets

사용자 개인정보를 보호하면서 공개용 데이터 집합을 생성하는 것은 앞으로 다뤄져야 할 또 다른 도전 과제입니다.

### I. Data quantity and generalizability

결국, 대부분의 사용자 패턴을 포괄적으로 다루기 위해 큰 데이터셋이 필요합니다. 광범위하고 다양한 데이터를 수집함으로써 딥러닝 모델을 보다 일반화된 모델로 만들고 편향에 민감하지 않도록 할 수 있습니다.

### J. Ethical code

연구자들은 OSNs에서 데이터를 수집하기 전에 윤리 규범을 완전히 이해하고, OSNs 사용자 및 제공업체에게 허가 요청을 보내는 등의 좋은 연구 관행을 적용해야 합니다.

## 5. Limitations

논문 선정
- 네 개의 학술 논문 데이터베이스만 사용하였습니다.
- 영어로 출판된 정신 건강 문제와 관련된 논문만을 포함하였습니다.

## 6. Future implications

1. 나라와 지역을 기반으로 새로운 방법을 구현하고 새로운 데이터 집합을 생성하는 것은 앞으로의 정신 건강 감지 연구를 개선할 수 있을 것입니다.
2. 언어 장벽 문제와 OSNs에서 다른 언어 추출은 지리적 위치와 모국어를 기반으로 한 연구에 대한 잠재적인 미래 방향을 제공합니다. 왜냐하면 많은 OSNs 사용자가 텍스트에서 영어만이 아닌 자국어를 사용하기 때문입니다.
3. OSNs에서 텍스트만이 아닌 다른 유형의 데이터(예: 사진, 오디오, 비디오)를 사용하는 것은 미래 연구에서 탐구할 수 있는 잠재적인 분야 중 하나일 수 있습니다.

## 7. Conclusion

이 연구는 OSNs가 정신 건강 문제 감지의 데이터 원천으로 높은 잠재력을 가지고 있지만, 얼굴 대 얼굴 인터뷰, 자체 보고 또는 설문 조사 배포를 기반으로 한 전통적인 정신 건강 감지 방법으로 대체될 수 없다는 결론을 내립니다. 그러나 OSNs는 보완적인 데이터를 제공할 수 있으며, 두 가지 접근법을 결합하여 정신 건강을 감지하는 것은 미래 연구를 개선할 수 있습니다.

미래에 정신 건강 문제 감지의 정확도와 정밀도를 높이기 위해 다음과 같은 접근을 취할 수 있습니다:
1. 포괄적인 채택
2. 혁신적인 알고리즘
3. 계산 언어학