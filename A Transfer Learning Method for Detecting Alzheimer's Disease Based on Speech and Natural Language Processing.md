# Summary

## 1. Problem Definition

- 알츠하이머병(AD)은 노령 인구에서의 유병률이 증가함에 따라 주요한 공중 보건 문제가 되고 있습니다. 이 논문은 현재의 방법들인 MRI 및 PET 스캔이 비싸고 침습적이라는 점을 고려할 때, 조기 AD 진단을 위한 비침습적이고 비용 효율적인 방법이 필요하다고 지적합니다.

## 2. Existing Research

- 이 논문은 유전자 발현 데이터셋에 딥러닝을 결합한 분자 바이오마커를 포함한 여러 AD 탐지 기술들을 검토합니다. 전통적인 머신 러닝 알고리즘의 한계점을 논의하는데, 이들은 광범위한 전문 지식을 필요로 하며 종종 이동성과 정확성이 부족합니다.

## 3. Experiment

### 3-1. Data Source

- 이 연구는 Alzheimer병 환자와 정상 개체의 오디오 녹음과 그에 상응하는 필기본을 포함한 ADReSS 데이터셋을 활용합니다.

### 3-2. Model Development

- 전이 학습 방법을 사용하며, 텍스트 임베딩을 위해 미리 학습된 BERT 모델 (구체적으로 distilBert)을 활용합니다. 이 과정은 문장이나 필기본을 768차원 벡터로 변환하는 것을 포함합니다.
- BERT 모델에서 추출한 특징들은 최종 분류 작업에 로지스틱 회귀 분류기에 사용됩니다.

### 3-3. Performance Evaluation

- 모델의 성능은 정확도, 정밀도, 재현율 및 F1 점수를 포함한 여러 지표를 기반으로 평가됩니다. 이러한 지표는 모델이 음성 및 텍스트 데이터에서 Alzheimer병 사례를 올바르게 식별하는 데 얼마나 효과적인지를 판단하는 데 도움이 됩니다.
- 이 연구는 제안된 모델의 성능을 CNN, 랜덤 포레스트, SVM 및 AdaBoost와 같은 다른 분류기와 비교하여 상대적인 효과를 평가합니다.

### 3-4. Experimental Focus

- 실험은 주로 필기본에서의 음성 패턴 및 언어 사용을 분석하여 Alzheimer병과 관련된 인지 장애의 지표를 찾습니다.
- 목표는 음성 및 언어 처리를 비침습적이고 경제적인 도구로 사용하여 Alzheimer병의 조기 감지를 증명하는 것입니다.

## 4. Proposed Solution

- 논문은 딥러닝과 기계 학습을 결합한 혁신적인 모델 아키텍처를 제안합니다. distilBert 모델을 사용하여 심층 의미적 특징을 추출하고, 이 특징들은 분류를 위해 로지스틱 회귀 모델로 전달됩니다.
- 이 접근 방식은 전이 학습과 BERT의 주의 메커니즘의 장점을 활용하여 AD와 관련된 특정 지표에 집중함으로써 AD 스크리닝에 대한 신뢰성 있고 저비용 및 편리한 솔루션을 제공하는 것을 목표로 합니다.

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

- AD 조기 진단을 위해 음성 및 자연어 처리(NLP) 기술을 기반으로 한 전이 학습 모델을 개발했습니다.
- 대규모 데이터셋의 부족으로 인해 특징 엔지니어링 없이 복잡한 신경망 모델을 사용하는 것이 제한되는데, 전이 학습은 이 문제를 효과적으로 해결할 수 있습니다.
- 구체적으로, 간결화된 양방향 인코더 표현 (distilBert) 임베딩과 로지스틱 회귀 분류기를 결합하여 AD와 정상 대조군을 구분하는 데 사용됩니다.
- 제안된 모델의 정확도는 0.88로, 이는 도전 과제에서의 최고 점수와 거의 동등합니다.
- 결과적으로, 이 연구에서 사용한 전이 학습 방법은 AD 예측을 개선하며, 특징 엔지니어링 필요성을 줄이는 것뿐만 아니라 충분히 큰 데이터셋의 부족도 해결합니다.

## Introduction

- 이 분야의 이전 연구들과 대조적으로, 이 연구는 수동 전문가식 특징 추출을 사용하지 않고 음성에서 의심스러운 AD 증상 특징을 자동으로 찾기 위해 신뢰할 수 있는 딥러닝 모델을 사용했습니다. 구체적으로는 미리 학습된 distilBert 언어 모델을 특징 추출기로 사용하여 입력 문장 또는 문서의 특징을 얻었으며, 이진 분류에 효과적인 간단한 로지스틱 회귀 분류기를 사용하여 AD와 정상 대조군을 분류했습니다.

게다가, 모델의 최적 매개변수를 얻기 위해 그리드 서치 전략을 사용했습니다. 결과는 이 방법이 2020년 ADRess 데이터셋에 대해 높은 정확도인 0.88을 달성한 것을 보여줍니다.

이 연구의 주요 기여는 다음과 같습니다:

1. 복잡한 전문 지식 없이도 효과적으로 구현된 필기본을 기반으로 한 Alzheimer병 진단의 간단하고 효과적인 모델이 설계되었습니다.
2. 딥러닝과 기계 학습을 결합한 혁신적인 모델 아키텍처가 제안되었으며, ADRess 데이터셋에서 최상의 성능을 얻었습니다.
3. 우리의 제안된 접근 방식은 신뢰성, 저비용 및 편리함의 장점을 가지고 있으며 Alzheimer병 스크리닝을 위한 실행 가능한 솔루션을 제공할 수 있습니다.

## Related Works

언어 기능은 다른 단계에서의 인지 장애 감지에 중요한 역할을 하기 때문에 NLP 기술과 딥러닝의 결합은 Alzheimer병 및 경도 인지 장애(MCI)의 감지에 정확하고 편리한 솔루션을 제공합니다.

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

게다가, 많은 연구에서 multimodal 데이터셋을 사용하여 Alzheimer병 및 경도 인지 장애(MCI)를 감지하였으며, 다른 모델로부터 더 정확하고 차별화된 정보를 얻을 수 있습니다.

전반적으로 강력한 표현 학습 능력과 구분력 있는 분류기, multimodal 정보 및 전이 학습은 Alzheimer병 및 경도 인지 장애(MCI)의 정확한 진단에 효과적인 요인입니다.

## Methods

### Transfer Learning

미리 학습된 모델을 사용하여 분류하는 일반적인 흐름은 다음 단계로 구성됩니다:

1. 대규모 데이터셋에서 일반적인 언어 모델을 훈련시키기.
2. 미리 학습된 언어 모델을 대상 데이터셋에 맞게 세부 조정하기.
3. 대상 특정 미리 학습된 언어 모델을 분류에 사용하기.

이 논문에서는 어텐션 메커니즘이 모델이 판단을 내릴 때 필기본의 일부에 집중할 수 있게 해주어 AD 진단에 적합하다는 주장을 합니다. 우리는 텍스트 임베딩을 위해 미리 학습된 BERT 모델을 사용했는데, 이 모델은 원래 문장이나 필기본을 768차원 벡터로 변환합니다.

### Overall Classification Framework

이 연구의 전체 모델 아키텍처는 주로 두 부분으로 구성됩니다: distilBert 모델과 로지스틱 회귀 분류기입니다.

DistilBert 모델은 BERT base 모델을 12개 층에서 6개 층으로 압축하고 토큰 유형 임베딩과 풀러(pooler)를 제거합니다. 이로써 속도는 60% 향상되고 아키텍처 크기는 40% 축소되지만 BERT 모델의 언어 이해 능력은 97% 보존됩니다.

구체적으로, 미리 학습된 distilBert 모델이 특징 추출기로 사용되며, 이 모델의 출력 레이어는 이진 분류를 위해 로지스틱 회귀 분류기로 대체됩니다.

![img](https://github.com/Sameta-cani/papers/blob/main/imgs/Pasted%20image%2020240115180117.png)

전체 과정의 알고리즘 설명은 아래에 제시되어 있습니다.

![img](https://github.com/Sameta-cani/papers/blob/main/imgs/Pasted%20image%2020240115180815.png)

DistilBert는 multi-head self attention과 location code와 같은 몇 가지 메커니즘을 가지고 있기 때문에 입력 텍스트의 전체적인 의미적 메시지를 철저하게 학습하여 멀리 떨어진 의존성을 포착할 수 있습니다.

이 프로세스는 여섯 번 반복되며 768차원의 의미적 특징 벡터가 얻어집니다. 본 연구에서 사용된 필기본은 그림 설명의 일부분으로, 최대 길이는 500을 넘지 않으므로 속도와 의미적 완성을 고려하여 단어 임베딩의 길이가 500으로 설정됩니다.

### Grid Search 

이 연구에서는 그리드 서치와 교차 검증을 포함하는 scikit-learn 도구의 GridSearchCV 함수를 사용하여 로지스틱 회귀 모델의 최적 매개변수를 찾습니다. 속도와 정확도를 고려하여 GridSearchCV 함수의 검색 범위는 0.0001에서 100까지이며, 간격은 20으로 설정되었습니다.

## Experiments

### ADReSS Datasets

![img](https://github.com/Sameta-cani/papers/blob/main/imgs/Pasted%20image%2020240115181958.png)

이 연구는 진단 어페지아 검사(Diagnostic Aphasia Examination)의 그림 설명 과제로, 참가자들은 그림을 가능한 한 자세히 설명하도록 요청됩니다.

데이터셋에서의 필기본 예시는 아래에 나와 있습니다.

*A boy and a girl are in a kitchen with their mothers. The little boy is getting a cookie for the little girl, but he is on a stool and is about to fall. The mother is washing dishes. She is obviously thinking of something else because the water pours out over the sink. She finished with some dishes. It seems to be summer because there are bushes. The window is open. There seems to be some kind of breeze because the curtains on the sill there blow. It must be fairly hot. The mother is in a sleeveless dress. The children are in short sleeve tops and have sandals. The little boy has tennis shoes. The mother obviously is unaware of what the children are doing. She will be aware of this shortly. How much more do you want to do?*

![img](https://github.com/Sameta-cani/papers/blob/main/imgs/Pasted%20image%2020240115182449.png)

다른 구간에서 두 그룹의 연령 분포는 **표 1**에 제시되었습니다.

![img](https://github.com/Sameta-cani/papers/blob/main/imgs/Pasted%20image%2020240115182529.png)

평균 연령 및 미니-정신 상태 검사(MMSE) 점수의 평균 값과 표준 편차는 **표 2**에 나와 있습니다.

### Experiment Results 

**Experiment environment**

- Windows 10
- Intel(R) Core I i5-6500 CPU @3.20 GHz, 3.19 GHz CPU, 44. GB RAM
- scikit-learn(logistic regression), NumPy, Pandas
- Python 3.6.13

이 실험에서는 모델의 성능을 평가하기 위해 정확도, 정밀도, 재현율 및 F1 점수를 지표로 사용하였습니다. (**==정밀도 표현이 잘못되었다고 생각합니다. TN -> TP로 수정해야 하지 않을까요?==**)

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

![img](https://github.com/Sameta-cani/papers/blob/main/imgs/Pasted%20image%2020240115184839.png)

챔피언은 음향 및 텍스트 두 모델을 사용하여 ERNIE 모델과 차별화된 표지자를 결합하여 표현 학습을 개선했으며(정확도 0.896), 우리는 distilBert 모델의 모델 아키텍처를 수정하여 텍스트만을 사용하여 강력한 분류 성능을 달성했습니다(정확도 0.88).

![img](https://github.com/Sameta-cani/papers/blob/main/imgs/Pasted%20image%2020240115185748.png)

**표 5**는 다른 분류기의 영향을 확인하기 위한 비교 결과를 보여줍니다.

## Discussion

최상의 성능은 우리 모델이 분류를 위한 유용한 특징을 학습했음을 나타내며, 이로써 전문가가 정의한 언어적 특징의 필요성을 줄이는데 도움이 되었을 뿐만 아니라 정확하고 복잡하며 포괄적인 특징을 데이터셋에서 추출할 수 있게 되었습니다.

우리 모델이 MMSE 점수 평가에 적합한지 여부는 추가로 검증해야 합니다.

자동 음성 인식(ASR) 생성 필기본을 직접 사용하면 추가 주석이 필요하지 않아 수동 특징 추출 방법보다 더 많은 이점이 있습니다.

우리 연구의 가장 큰 제한은 많은 매개변수를 가진 모델의 성능을 해석하기 어렵다는 점입니다. 즉, 우리 모델은 잘못된 판정의 이유를 이해할 수 없지만 올바른 예측의 경우 네트워크가 더 많은 주의를 기울인 단어를 식별할 수 있습니다. 이러한 기능은 AD 환자의 중요한 언어적 특성을 드러낼 수 있어 언어 치료 및 AD 환자와의 의사 소통에 도움이 될 수 있습니다.

미리 학습된 모델 및 세부 조정 패러다임의 실무 적용은 다양한 하위 작업에서 우수한 성능을 달성했습니다. 그러나 대규모 모델에서 해결해야 할 문제들, 예를 들어 데이터셋 품질, 거대한 훈련 에너지 소비, 탄소 배출 문제, 일반적인 감각 및 추론 능력 부족 등 여전히 해결해야 할 문제들이 있습니다.

## Feature works 

앞으로, Alzheimer병 진단을 위해 다음 두 가지 방향에 초점을 맞출 것입니다.
1. 대조적 학습 방법을 사용하여 Alzheimer병 진단에 대한 내재적 감정 분석에 중점을 둡니다.
2. 다국어 BERT와 transformer 모델을 포함한 다국어 언어 간 전이 학습을 활용하여 다국어 Alzheimer병 인식을 개선하기 위해 노력할 것입니다.
