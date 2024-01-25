# Publicly Available Clinical BERT Embeddings

## Authors

**Emily Alsentzer**: Harvard-MIT Cambridge, MA

**John R. Murphy**: MIT CSAIL Cambridge, MA

**Willie Boag**: MIT CSAIL Cambridge, MA

**Wei-Hung Weng**: MIT CSAIL Cambridge, MA

**Di Jin**: MIT CSAIL Cambridge, MA

**Tristan Naumann**: Microsoft Research Redmond, WA

**Matthew B. A. McDermott**: MIT CSAIL Cambridge, MA

<hr>

## Article

**Posted Date**: Apr 6th, 2019

**DOI**: https://arxiv.org/abs/1904.03323

<hr>

## Abstract

To date, there are no publicly available pre-trained BERT models yet in the clinical domain. In this work, we explored and released two BERT models for clinical text:
1. generic clinical text
2. discharge summaries

Using domain-specific models improves performance on three common clinical NLP tasks compared to non-specific embeddings.

## 1 Introduction

Clinical narratives (e.g., physician notes) have known differences in linguistic characteristics from both general text and non-clinical biomedical text, motivating the need for specialized clinical BERT models.

In particular, we make the following contributions:
1. We train and publicly release BERT-Base and BioBERT-finetuned models trained on both all clinical notes and only discharge summaries.
2. We demonstrate that using clinical-specific contextual embeddings improves both upon general domain results and BioBERT results across well-established clinical NER tasks and one medical natural language inference task (i2b2 2010, i2b2 2012, and MedNLI). 

## 2 Related Work

**BERT**
- Can create a context-sensitive embedding for each word in a given sentence
- Deeper and contains much more parameters, thus possessing greater representation power
- Can be incorporated into a downstream task and gets fine-tuned as an integrated task-specific architecture.
- In general, BERT has been found to be superior to ELMo and far superior to non-contextual embeddings on a variety of tasks, including those in the clinical domain.

## 3 Methods

### 3.1 Data

![img](https://github.com/Sameta-cani/papers/blob/main/imgs/Pasted%20image%2020240118174135.png)

Many note types are semi-structured, with section headers separating free-text paragraphs. To process these notes, we split all notes into sections, then used Scispacy(specifically, the en core sci md tokenizer) to perform sentence extraction.

Clinical text from the approximately 2 million notes in the MIMIC-III v1.4 database

We train two varieties of BERT on MIMIC notes: 
- **Clinical BERT**: uses text from all note types 
- **Discharge Summary BERT**: uses only discharge summaries in an effort to tailor the corpus to downstream tasks (which often largely use discharge summaries).

### 3.2 BERT Training

In this work, we aim to provide the pre-trained embeddings as a community resource, rather than demonstrate technical novelty in the training procedure.

We trained two BERT models on clinical text: 
1. **Clinical BERT**(initialized from BERTBase)
2. **Clinical BioBERT**(initialized from BioBERT)

For all downstream tasks, BERT models were allowed to be fine-tuned, then the output BERT embedding was passed through a single linear layer for classification, either at a per-token level for NER or de-ID tasks or applied to the sentinel “begin sentence” token for MedNLI.

For all pre-training experiments, we leverage the tensorflow implementation of BERT.
### Pre-training
- batch size: 32
- maximum sequence length: 128
- learning rate: $5 \cdot 10^{-5}$ 
- training step: 150,000 steps (Iinitially 300,000 steps)
- dup factor: 5
- masked language model probability: 0.15
- max predictions per sequence: 20
### Fine-tuning
- learning rate $\in \{2 \cdot 10^{-5}, 3 \cdot 10^{-5}, 5 \cdot 10^{-5}\}$
- batch size $\in \{16, 32\}$
- epochs $\in \{3, 4\}$
For the NER tasks, wer also tried epoch $\in \{2\}$. The maximum sequence length was 150 across all tasks.

### Computational Cost

- Entire embedding model procedure took roughly 17 - 18 days of computational runtime 
- Using a single GeForce GTX TITAN X 12 GB GPU (and significant CPU power and memory for pre-processing tasks).

### 3.3 Tasks 

![img](https://github.com/Sameta-cani/papers/blob/main/imgs/Pasted%20image%2020240118172608.png)

## 4 Results & Discussions

### Clinical NLP Tasks 

![img](https://github.com/Sameta-cani/papers/blob/main/imgs/Pasted%20image%2020240118172902.png)

- Overall, we feel our results demonstrates the utility of using domain-specific contextual embeddings for non de-ID clinical NLP tasks.
- Additionally, on one task Discharge Summary BERT offers performance improvements over Clinical BERT, so it may be that adding greater specificity to the underlying corpus is helpful in some cases.

### Qualitative Embedding Comparisons

![img](https://github.com/Sameta-cani/papers/blob/main/imgs/Pasted%20image%2020240118173432.png)

These lists suggest that Clinical BERT retains greater cohesion around medical or clinicoperations relevant terms than does BioBERT.

### Limitations & Future Work

**Limitations**
- we do not experiment with any more advanced model architectures atop our embeddings. This likely hurts our performance.
- MIMIC only contains notes from the intensive care unit of a single healthcare institution (BIDMC).
- our model shows no improvements for either de-ID task we explored.

**Future Work**
- Differences in care practices across institutions are significant, and using notes from multiple institutions could offer significant gains.

## 5 Conclusion

We find robust evidence that our clinical embeddings are superior to general domain or BioBERT specific embeddings for non de-ID tasks, and that using note-type specific corpora can induce further selective performance benefits.
