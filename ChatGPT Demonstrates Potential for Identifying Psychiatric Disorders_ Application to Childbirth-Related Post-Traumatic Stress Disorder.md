
# Summary
## 1. Problem Definition

- The study focuses on the challenge of diagnosing psychiatric disorders, specifically Childbirth-Related Post-Traumatic Stress Disorder (CB-PTSD), which affects millions of women annually and lacks a standard screening protocol.
- The paper highlights the potential of using advanced text-based computational methods and Machine Learning (ML) for early detection of CB-PTSD using childbirth narratives.

## 2. Existing Research

- The paper reviews the current state of Natural Language Processing (NLP) and ML in diagnosing psychiatric conditions.
- It discusses previous models like MentalBERT and MentalRoBERTa, which were trained for mental healthcare research, and their effectiveness in identifying stress, anxiety, and depression.
- The study acknowledgeds the limitations of existing models and the need for more specialized models for specific psychiatric conditions like CB-PTSD.

## 3. Experiment

### 3-1. Research Design

- The study is part of a larger research project focusing on the impact of childbirth experiences on maternal mental health.
- Women over 18 years old who had given birth to a living infant within the past six months participated anonymously in a web survey, providing information about their mental health and childbirth experiences.
- At the end of the survey, participants had the opportunity to narrate their childbirth story, and these narratives were collected on average 2.73 ± 1.82 months postpartum.

### 3-2. Measurement Method 

- Open-ended, unstructured text-based narratives about childbirth were collected.
- Participants were asked to provide brief narratives about their childbirth experience, focusing particularly on the most stressful aspects.

### 3-3. Narrative Analysis

- The research team compared three models using OpenAI's API.
- The first model used gpt-3.5-turbo-16k for zero-shot classification without additional examples.
- The second model used gpt-3.5-turbo-16k for few-shot learning with a few examples.
- The third model trained a neural network machine learning model using the text-embedding-ada-002 model to classify CB-PTSD.

### 3-4. Model Evaluation

- The first and second models were evaluated across the entire dataset.
- The third model was trained on the Train set and evaluated on the Test set.
- The models were compared using F1 scores and Area Under the Curve (AUC).

## 4. Proposed Solution

- The research explores the capabilities of ChatGPT in analyzing childbirth narratives to identify potential markers of CB-PTSD.
- A new ML model utilizing ChatGPT's knowledge is developed for narrative classification to identify CB-PTSD.
- The model is tested against six previously published large language models (LLMs) trained on mental health or clinical domains data. The new model demonstrates superior performance (F1 score: 0.82), suggesting that ChatGPT can effectively identify CB-PTSD.
- The approach can be generalized to assess other mental health disorders.

# ChatGPT Demonstrates Potential for Identifying Psychiatric Disorders: Application to Childbirth-Related Post-Traumatic Stress Disorder

## Author

**Alon Bartal** : Bar-llan University
**Kathleen Jagodnik** : Bar-llan University, Harvard Medical School & Massachusetts General Hospital
**Sabrina Chan** : Massachusetts General Hospital 
**Sharon Dekel** : Harvard Medical School & Massachusetts General Hospital 

<hr>
## Article

**Keywords**: Birth narratives, ChatGPT, Childbirth-related post-traumatic stress disorder (CB-PTSD), Maternal mental health, Postpartum psychopathology, Pre-trained large language model (PLM)

**Posted Date**: October 19th, 2023

**DOI**: https://doi.org/10.21203/rs.3.rs-3428787/v1

<hr>

## Abstract

- The study demonstrates the promise of using natural language processing (NLP) and machine learning (ML) for free-text analysis in diagnosing psychiatric disorders.
- ChatGPT shows initial feasibility for this purpose, but whether it can accurately assess psychiatric disorders remains undetermined.
- The research examines the utility of ChatGPT in identifying Childbirth-Related Post-Traumatic Stress Disorder (CB-PTSD), a postnatal psychiatric condition affeting millions of women annually, lacking standard screening protocols.
- The team explores ChatGPT's potential by analyzing childbirth narratives as the sole data soucre. They develop an ML model utilizing ChatGPT's knowledge to identify CB-PTSD through narrative classification.
- This model outperforms six previously published large language models (LLMs) trained on metal health or clinical data, showing superior performance (F1 score: 0.82). This suggests that ChatGPT could be used to identify CB-PTSD.
- Our modeling approach can be generalized to assess other mental health disorders.

## 1 Introduction

Combined with ML models, language models have been reported as useful in the classification of psychiatric conditions. Two studies(MentalBERT and Mental-RoBERTa, The mental-xlnet-base-cased LLM) fount that language representations pretrained in the target domain improve model performance on mental health detection tasks.

ChatGPT (GPT-3.5 and GPT-4) have shown significant language  processing capabilities in the realm of mental health analysis.

Untreated CB-PTSD is associated with negative effects in the mother and, by extension, her child, and these consequences carry significant societal costs. Early treatment for CB-PTSD facilitates improved outcomes. Self-diagnosis, one of the ways to determine CB-PTSD, has several disadvantages. Therefore, relying on self-reported symptoms can compromise the identification of CB-PTSD.

In this context, the narrative style and language that individuals use when recounting traumatic events have been suggested to provide deep insights into their mental well-being. The words in individuals' traumatic narratives may reflect post-trauma adjustment even before deep psychological analysis occurs.

We previously used the embeddings of sentence-transformers PLMs to train an ML classifier for identifying at-risk women; the model achieved good performance (F1 score of 0.76). However, more research is required to characterize how word useage in birth narratives indicates maternal mental health, and understanding and analyzing traumatic narratives remains a research area ripe for exploration.

This paper explores the capabilities of ChatGPT, examining its efficacy in analyzing childbirth narratives to identify potential markers of CB-PTSD. Through the lens of ChatGPT and associated models, we aim to bridge the gap between traumatic narratives and early detection of psychiatric dysfunction, offering a novel approach to identifying women at risk for CB-PTSD.

## 2 Materials and Methods 

### 2.1 Study Design

- The study is part of a larger research project focusing on the impact of childbirth experiences on maternal mental health.
- Women over 18 years old who had given birth to a living infant within the past six months participated anonymously in a web survey, providing information about their mental health and childbirth experiences.
- At the end of the survey, participants had the opportunity to narrate their childbirth story, and these narratives were collected on average 2.73 ± 1.82 months postpartum.

![img](https://github.com/Sameta-cani/papers/blob/main/imgs/Pasted%20image%2020240111185246.png)

### 2.2 Measures

Among investigators, the PCL-5, a highly reliable survey, was used to identify PTSD, with a score of 31 as the standard.

### 2.3 Narrative Analysis 

![img](https://github.com/Sameta-cani/papers/blob/main/imgs/Pasted%20image%2020240111191553.png)

gpt-3.5-turbo-16k: Natural language understanding and generation + capable of processing 4x longer narratives of up to 16,384 tokens

![img](https://github.com/Sameta-cani/papers/blob/main/imgs/Pasted%20image%2020240111192905.png)

**Model #1 - Zero-shot classification** with no previous examples given to the model. The category associated with the model's highest-confidence response was '1' (Class 1: CB-PTSD) or '0' (Class 0: No CB-PTSD) as the predicted class for the narrative. The 'temperature' variable was set to 0, to make the model deterministic, i.e., always choosing the most likely next token in the sequence.

**Model #2 - Few-shot classification**. We provided two narratives and their associated labels in a conversation format to guide the model towards the classification task (Table 2). The gpt-3.5-turbo-16k model with ‘temperature’= 0 then used these examples to classify the expected output for the subsequent narrative. Increasing the number of examples up to 4 provided similar model performances.

**Model #3 - Training an ML classifier**. We trained a neural network (NN) ML model using the vector representation (embeddings) of narratives generated by the text-embedding-ada-002 OpenAI’s model, to classify narratives as markers of endorsement (Class 1), or no-endorsement (Class 0), of CB-PTSD. 

In Step #1 of Aapendix A, we label narratives associated with PCL-5 $\geq$ 31 as 'CB-PTSD' (Class 1; 190 subjects), else 'no CB-PTSD' (Class 0; 1,105 subjects).

In Step #2, we discarded narratives with 30 words and balanced the dataset using down-sampling by randomly sampling the majority Class 0 to fit the size of the minority Class 1, resulting in 190 narratives in each class. We constructed the Train and Test dataset as described in Step #2, resulting in 170 narratives in each class.

To identify similar or contextually relevant narratives, in Step #3 we adopted the approach used in our previous work. This approach analyzes pairs of narratives as training examples, thus substantially increasing the number of training examples.

We created three sets of sentence-pairs using the Train set: 
- Set #1: All possible pairs of sentences (C(n, r) = C(170, 2) = 14365) in Class 1 (CB-PTSD).
- Set #2: All possible pairs of sentences (14365) in Class 0. 
- Set #3: Pairs of sentences (28730), one randomly selected from Class 1 and another randomly selected from Class 0.

We labeled sets #1 and #2 as positive examples as they contained semantically or contextually similar pairs of sentences (i.e., either a pair of narratives of individuals with, or without, CB-PTSD). We labeled set #3 as negative examples as it contained pairs of non-semantically or non-contextually similar pairs of sentences. This data augmentation process produced 57460 training examples in the Train set.\

Next, we mapped each narrative using the text-embedding-ada-002 model into a 1536-dimensional vector. Lastly, we computed the Hadamard product ($\circ$) among each of the 57460 embedding (emb) vectors of pairs of sentences $(u, v)$  in sets #1 to #3 of the Train set (Step 3.1), such that $z = (emb(u) \circ emb(b))$ ([[#Appendix A Steps to Build and Test Model 3 of This Study|Appendix A]]).

Finally, using the 57640 vectors, following the modeling approach in, we trained a deep feedforward neural network (DFNN) model to classify pairs of sentences in sets #1 to #3 as semantically similar or not. 

**DFNN**

![img](https://github.com/Sameta-cani/papers/blob/main/imgs/Pasted%20image%2020240112013220.png)


- Input layer: 1,536 neurons
- hidden layer: 400 neurons, 50 neurons
- output layer: 1 neurons
- activation function: ReLU(hidden), Sigmoid(output)
- epochs: 50
- optimizer: Adam
- learning rate: $1e^{-4}$
- batch size: 32
- loss function: binary cross-entropy

Steps #1 to #3 of Model #3 ([[#Appendix A Steps to Build and Test Model 3 of This Study|Appendix A]]) were repeated 10 times to capture different narratives for creating an accurate representation of Classes 0 and 1.

**Model evaluation**. Model#1 and Model#2 were performed with the entire dataset, and Model#3 was performed with test data, and were compared after each repeated evaluation 10 times. The measurement indices were set to F1 score and AUC.

## 3 Results

![img](https://github.com/Sameta-cani/papers/blob/main/imgs/Pasted%20image%2020240112014409.png)

Model #3 outperformed all other models in terms of AUC, F1 score, sensitivity, and specificity.

ChatGPT Model #1 and Model #2 are untrained and therefore have difficulty classifying narratives from specific specialties. Model#3 performed better than the other models by using a larger number of examples (57,460) and being trained on a specific classification task. This specialized training used embeddings to create a classification system designed to detect CB-PTSD. By training the model in this way, it was better suited for the specialized task of CB-PTSD detection.

![img](https://github.com/Sameta-cani/papers/blob/main/imgs/Pasted%20image%2020240112020415.png)

The models that we compared (Table 4) were evaluated using two Evaluation Methods on the dataset.

Evaluation Method 1: We fine-tuned each of the 6 evaluated PLMs on a down-stream task of classifying narratives as CB-PTSD (Class 1) or not (Class 0). We used 30%-70% of the data for the Test and Train split, respectively.

Evaluation Method 2: We used the developed Model #3 with embeddings of the 6 evaluated PLMs. Following Step 2 of [[#Appendix A Steps to Build and Test Model 3 of This Study|Appendix A]], we split the Train and Test sets 10 times (similar to a 10-fold cross-validation process).

## 4 Discussion

This study sought to explore the performance of different variations of ChatGPT for the purpose of identifying probable cases of childbirth related post-traumatic stress disorder (CB-PTSD) using the childbirth narratives of postpartum women.

By assessing several model variations of ChatGPT, we systematically studied the capabilities and shortcomings of pre-trained large language models (PLMs) to assess maternal mental health using childbirth narratives. While Models #1 (zero-shot learning) and #2 (few-shot learning) that utilize the pre-trained ChatGPT model exhibited limitations, Model #3, drawing from OpenAI’s GPT-3 embeddings (text-embedding-ada-002), demonstrated superior performance in identifying CB-PTSD. 

Notably, Model #3’s performance surpasses both the basic implementations of ChatGPT and other PLMs trained in clinical and mental health domains, supporting its potential to offer richer insights into maternal mental health following traumatic childbirth.

### 4.1 Limitations 

- The potential enhancement of our model with data from additional sources remains unexplored.
- While we assessed the presence of CB-PTSD using the PCL-5 questionnaire, clinician evaluations were not performed.
- A more diverse subject population beyond middle-class American women is needed in future work to facilitate the development of a universally applicable tool for CB-PTSD assessment.

### 4.2 Improvements

- Specific fine-tuning of ChatGPT for CB-PTSD narrative language, optimizing embedding vector representation.
- The integration of additional data types, including electronic medical records.

## 5 Conclusions

We find that a ChatGPT model untrained on a specific clinical task shows inadequate performance in the task of identifying childbirth-related post-traumatic stress disorder (CB-PTSD), while our model trained on the embeddings of OpenAI’s model yields the best performance to date for this task, representing good performance.

## Appendix A Steps to Build and Test Model #3 of This Study

**Step 1. Define a PCL-5 cutoff score.**  and **Step 2. Data preparation.**

![img](https://github.com/Sameta-cani/papers/blob/main/imgs/Pasted%20image%2020240112180708.png)

**Step 3. Develop a Machine Learning (ML) classifier that utilizes Natural Language Processing (NLP) features.

![img](https://github.com/Sameta-cani/papers/blob/main/imgs/Pasted%20image%2020240112182334.png)

This approach allowed us to generate multiple training examples since there are $\frac{n(n-1)}{2}$ possible combinations for $n$ sentences, thus addressing the challenge of training an ML model with a low number of examples, as in Class 1.

More specifically, the following three substeps describe the model development.

1. Each pair of sentences in Class 1, and each pair of sentences in Class 0, were labeled as positive examples, indicating semantically or contextually similar sentences of individuals with (Set #1) or without (Set #2) CB-PTSD, respectively. Next, negative examples (Set #3) of the same size as the positive examples sets (||Set #1|| + ||Set #2||) were created by randomly selecting pairs of sentences, one from Class 1, and the other from Class 0, indicating semantically or contextually nonsimilar sentences.
2. Using the text-embedding-ada-002 model, each sentence was mapped into a dense vector space. Then, for each Set #1 to #3, we computed a vector z of the embedding (emb) of each pair of sentences (u, v), selected in Substep 1, z = (emb(u) ◦ emb(v)).
3. A densely connected feedforward neural network (DFNN) was trained to classify pairs of sentences (by processing vector z) as semantically similar or not.

**Step4. Test model performance.**

We compared the performance of Model #3 with the model in, as well as with Model#1 and Model #2. We report the area under the receiver operating characteristic curve (AUC), F1 score, Sensitivity, and Specificity measures on the Test set. 

1. To test Model #3 on a newly unseen narrative $S$ in the Test set, we first compute its embeddings.
2. Calculate the average embedding vector ($\bar{v}_n$) of all Train narratives in Class 0, and the average embedding vector ($\bar{v}_p$) of all Train narratives in Class 1. 
3. To decide the class of S, we compute $z_n = (emb(S) ◦ \bar{v}_n)$, and $z_p = (emb(S) ◦ \bar{v}_p)$. 
4. Apply Model #3 (denoted as $f(x)$) to $z_n$ and $z_p$, and compare its output, i.e., compare the likelihood of similarity of emb($S$) to $v_p$ with the likelihood of similarity of emb($S$) to $v_n$. If $f(z_p) > f(z_n)$, we say that $S ∈$ Class 1, else $S ∈$ Class 0. 

Intuitively, our model should assign a higher likelihood of similarity between an embedded narrative of a woman with CB-PTSD to the vector $\bar{v}_p$ than to the vector $\bar{v}_n$.

![img](https://github.com/Sameta-cani/papers/blob/main/imgs/Pasted%20image%2020240112184518.png)
