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

This paper critically evaluates the effectiveness of using Online Social Networks for mental health detection. It primarily reviews papers published between 2007 and 2018, focusing on machine learning techniques and data analysis methods. The study highlights that OSNs are a promising data source for the early detection of mental health issues, emphasizing the need for innovative algorithms and computational linguistics in this area, as well as the importance of collaboration with mental health professionals.

## 1. Introduction

OSNs can generate a massive amount of information that can be used to develop an approach for mental health problem detection.

An analysis of the current mental health detection in OSNs is required to comprehend data sets, data analysis methods, feature extraction method, classifier performance (i.e., accuracy and efficiency), challenges, limitations, and future work. 

The purpose of this systematic review is to conduct a critical assessment analysis of mental health problem detection based on data extracted from OSNs. It intends to explore the competence of mental health problem detection in OSNs, including its challenges, limitations, and future work.

## 2. Methods

### A. Identification and selection of studies

Most studies that used OSNs as their data source in mental health problem detection were included in the selection. The current study explains how previous researchers used OSNs as their data source in mental health problem detection.

- followed the guidelines of the PRISMA.
- use an eletronic literature search
- use common mental health disorder keywords as defined by the UK National Institute.
- reffer to the Medical Subject Headings to ensure that the key terms used in mental health are inclusive in the literature.

$\rightarrow$ 22 articles were revised as possible suitable studies.

![[Pasted image 20240117141348.png]]

### B. Methodological quality assessment

Adopted the Critical Appraisal Skills Program (CASP) checklist. The major features and limitations were analyzed and compared to indicate the strengths and weaknesses for each of the studies.

The main features and limitations are based on:
- **extraction of data**: data source, keywords, duration, and geographical location of data data extracted
- **quality of data**: data set related to mental health problems
- **study design**: suitable methodology applied
- **results**: clear study objectives and outcomes

## 3. Results

### A. Finding and selecting studies

From a total of 2770 articles, 22 articles were selected by applying the criteria of redundancy, irrelevance, and Table 1.

![[Pasted image 20240118021614.png]]

As presented in Table 2, major characteristics and a summary of the content analysis of the articles are discussed.

The summary of the selected articles were in accordance with the data set (data sources, keywords, duration and geographical location), method of data analysis, study objectives, feature extraction method, machine learning techniques, and classifier performances. 

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
- First, only a few studies on mental illness found useful information, such as people with mental health problems isolating themselves and not communicating with others.
- Second, the researchers found an additional finding related to the use of different languages in mental health problem detection during data analytics in OSNs.

**Challenges**
- non-face-to-face communication and human-computer interaction.
- language barrier
- account privacy policy

### A. Quality of data sets and model interpretation

Using your own dataset has the advantage that the information is location-specific and specific to your research purposes, but it can also introduce some bias.

Interpreting such models is likely difficult because several complex statistical patterns that correlate to minor indicators across various features are involved.

### B. Mental health problem detection over time

One of the interesting and challenging tasks is mental health problem detection over time. In contrast with other text classification tasks, mental health status can vary significantly over time. For example, a mental health case reported in an OSNs website may begin with a simple mental health issue (i.e., a weak signal) and end with a suicide case (i.e., a strong signal). Consequently, different mental health scenarios that change over time should be considered while building machine learning models. A model should be effective in detecting weak signals and in continuously evolving mental health detection cases over time.

### C. Multicategories of mental health problems

Categorizing mental health problems is difficult because numerous feature selection processes should be performed by researchers. Consequently, researchers choose to either generalize or specify the category of a mental health problem.

### D. Data preprocessing

The preparation of a new data set is one of the challenges in mental health problem detection.

### E. Length of posts 

Understanding mental health problems from a limited post length is one of the important challenges that must be considered when developing machine learning models for mental health problem detection.

### F. Multilingual content

The analysis of multilingual texts is an exciting challenge that must be addressed in the future.

### G. Data sparsity

Addressing the data sparsity problem is crucial because it may negatively affect the performance of a machine leaning model.

### H. Publicy available data sets

The creation of publicly available data sets while preserving user privacy is another challenge that must be addressed in the future.

### I. Data quantity and generalizability

Consequently, a large dataset is required to comprehensively cover the patterns of most users to have more generalized models. Collecting extensive and diverse data can make deep learning models more generalizable and less vulnerable to bias.

### J. Ethical code

Researchers should fully understand the ethical code of conduct before collecting data from OSNs and should apply good research practice by sending permission requests to OSNs users and providers.

## 5. Limitations

Selection of articles
- only four journal databases
- only articles publlished in English and related to mental health problems were included.

## 6. Future implications

1. Implementing a new method and creating a new data set based on countries and localities may improve research on mental health detection in the future. 
2. The language barrier issue and extracting other languages from OSNs provide a potential future direction for research based on geographical location and native language because many OSNs users use their native language, instead of only English, in texting. 
3. Using another type of data (e.g., pictures, audios, and videos), instead of texts only in OSNs can be one of the potential areas for exploration in future research.

## 7. Conclusion

 This study concludes that OSNs exhibit high potential as data sources of mental health problems detection but can never be substituted with traditional mental health detection methods based on face-to-face interviews, self-reporting, or questionnaire distribution. Nevertheless, OSNs can provide complementary data, and the combination of the two approaches in detecting mental health can improve future research.

To increase the accuracy and precision of mental health problem detection in the future:
1. comprehensive adoption
2. innovative algorithms
3. computational linguistics