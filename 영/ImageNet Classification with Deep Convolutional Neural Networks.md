# ImageNet Classification with Deep Convolutional Neural Networks

## Author

**Alex Krizhevsky**: University of Toronto
**Ilya Sutskever**: University of Tornoto
**Geoffrey E. Hinton**: University of Toronto

<hr>

## Article

**Keywords**: ImageNet, Convolutional Neural Networks(CNNs), AlexNet, GPU Acceleration, Dropout

**Posted Date**: 03 December 2012

**DOI**: https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html

<hr>

## Abstract

**제안한 모델**
- 6천만 개의 파라미터
- 650,000개의 뉴런
- 5개의 CNN Layer
- MaxPooling
- Softmax(class; 1000)
- Dropout(오버피팅을 방지하기 위해)

훈련 속도를 향상시키기 위해 비포화 뉴런을 사용하고, CNN에 효율적인 GPU를 구현했다. 이 모델은 ImageNet LSVRC-2010 대회의 테스트 데이터셋에서 기존 최고 성능 모델(SOTA)을 크게 상회하는 결과를 보였다. 또한, ILSVRC-2012 대회에서는 이 모델의 변형이 상위 5개의 가장 낮은 테스트 오류율을 기록했다.

## 1 Introduction

Object Recognition의 성능 향상을 위해서는 큰 데이터셋, 강력한 학습 모델, 그리고 오버피팅 방지를 위한 진보된 기술이 필요하다.

현실 세계의 객체는 다양성이 크기 때문에 더 많은 데이터셋이 필요하다. 최근에는 LabelMe, ImageNet 등 수백만 개의 이미지와 레이블을 포함하는 대규모 데이터셋을 수집할 수 있게 되어, 이를 통해 성능 향상을 기대할 수 있다.

**CNN의 장점**
- 깊이와 폭을 변경하여 수용성을 제어
- 이미지의 특성([stationarity of statistics and locality of pixel dependencies](https://seongkyun.github.io/study/2019/10/27/cnn_stationarity/))에 대해 강력
- 우수한 추론 능력
- feedforward 신경망과 비교했을 때, 연결과 매개변수가 훨씬 적기 때문에 학습이 더 쉽지만, 이론적으로 최고의 성능은 약간 더 낮을 가능성이 있음

CNN의 여러 장점에도 불구하고 고해상도 이미지를 대규모로 처리할 때는 상당한 비용이 발생한다. 하지만, 다음과 같은 방법으로 이 문제를 어느 정도 해결할 수 있다:

1. 고도로 최적화된 2D 컨볼루션과 고성능 GPU를 결합하여 구현함으로써 처리 효율을 높임
2. 질 좋은 데이터셋을 활용.

제안된 네트워크는 5개의 Convolution layer와 3개의 Fully-connected layer로 구성되어 있으며, 모든 레이어가 중요하여 어느 하나가 빠질 경우 성능이 저하된다. 네트워크는 2개의 GTX 580 3GB GPU를 사용하여 5~6일 동안 학습을 진행했으며, 향후 더 빠른 GPU와 더 큰 데이터셋을 사용할 경우 성능 개선이 기대된다.

## 2 Dataset

ILSVRC(ImageNet Large-Scale Visual Recognition Challenge)
-  1,000개의 category(각각 1,000개의 이미지)
-  ImageNet의 하위 집합 사용
-  train: 약 120만 개
-  valid: 약 5만 개
-  test: 약 15만 개
  
ILSVRC에서 제공하는 다양한 크기의 이미지를 먼저 짧은 쪽을 256픽셀로 조정하고, 중앙을 기준으로 크롭하여 256x256 픽셀의 고정 크기로 다운샘플링했다.

## 3 The Architecture

![그림 1](https://github.com/Sameta-cani/jwork/assets/83288284/a3b64741-620f-491e-a928-1b4ae6307941)

**Figure 1**: CNN 아키텍처에서 두 GPU 간의 책임을 명확하게 나타내고 있다.


### 3.1 ReLU Nonlinearity

본 논문에서는 기존의 포화 비선형성 활성화 함수인 $\text{tanh}(x)$ 및 $\text{sigmoid}(x)$ 대신, 비포화 비선형성 활성화 함수인 Rectified Linear Units (ReLUs), $f(x) = \text{max}(0, x)$를 사용할 경우 학습 속도를 훨씬 더 빠르게 향상시킬 수 있음을 제안한다.

![그림 2](https://github.com/Sameta-cani/jwork/assets/83288284/e0b25012-6c69-461f-ab9f-721e181d8ca8)

**Figure 2**: ReLU(실선)를 사용하는 4층 컨볼루션 신경망은 tanh 뉴런(점선)을 사용하는 동등한 네트워크에 비해 CIFAR-10에서 훈련 오류율 25%를 달성하는 속도가 6배 빠르다.



### 3.2 Training on Multiple GPUs

단일 GTX 580 GPU는 메모리가 3GB로 제한되어 있어, 학습 가능한 네트워크의 크기가 한정된다. 이에 두 개의 GPU를 병렬 연결하여 네트워크를 분산시켰다. 이 방식은 cross-validation에서는 문제가 될 수 있지만, 계산량을 효과적으로 조절할 수 있다. 결과적으로, 이 구성은 1개의 GPU를 사용했을 때보다 top-1 error rate를 1.7%, top-5 error rate를 1.2% 감소시켰으며, 학습 시간도 단축되었다.

### 3.3 Local Response Normalization

ReLU 활성화 함수는 다른 활성화 함수와 달리 입력 특성의 정규화가 필요 없다는 장점이 있다. 그러나 Local Response Normalization을 적용함으로써 일반화 성능을 향상시킬 수 있다.

$$
 b^i_{x, y} = a^{i}_{x, y}/(k + \alpha \sum_{j=\text{max}(0, i-n/2)}^{\text{min}(N - 1, i+n/2)}(a_{x, y}^j)^2)^\beta
$$

위 식에서 합계는 동일한 공간 위치에 있는 $n$개의 "인접" 커널 맵에 걸쳐 실행되며 $N$은 레이어의 총 커널 수를 의미한다.

상수 $k, n, \alpha$ 및 $\beta$는 검증셋을 사용하여 값이 결정되는 하이퍼 파라미터로, $k = 2, n = 5, \alpha = 10^{-4}, \beta = 0.75$를 사용했다.

이 방법으로 top-1 error rate와 top-5 error rate를 각각 1.4%, 1.2% 줄일 수 있었다.

### 3.4 Overlapping Pooling

![그림 2](https://github.com/Sameta-cani/jwork/assets/83288284/fa741a35-0c86-4f4a-9c21-88253bdfd106)

**Figure 3**: Overlapping Pooling

CNN의 PoolingLayer는 동일한 kernel 맵에서 인접한 뉴런 그룹의 출력을 요약하고, feature map에서의 크기를 줄여 연산량을 줄이는 역할을 한다.

여기서 stride의 크기를 $s$, kernel size를 $z$라고 가정한다. 만약 $s = z$로 설정하면 CNN에서 일반적으로 사용되는 전통적인 로컬 풀링을 얻을 수 있다.

$s < z$라면, Overlapping Pooling을 얻으며, 이때 top-1 error rate와 top-5 error rate를 각각 0.4%, 0.3% 줄일 수 있었다.

### 3.5 Overall Architecture

|Layer|# filters / neurons|Filter size|Stride|Padding|Size of feature map|Activation function|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Input|-|-|-|-|227 x 227 x 3|-|
|Conv1|96|11 x 11|4|-|55 x 55 x 96|ReLU|
|Local Response Normalization|-|-|-|-|-|-|
|Max Pool 1|-|3 x 3|2|-|27 x 27 x 96|-|
|Conv2|256|5 x 5|1|2|27 x 27 256|ReLU|
|Local Response Normalization|-|-|-|-|-|-|
|Max Pool 2|-|3 x 3|2|-|13 x 13 x 256|-|
|Conv3|384|3 x 3|1|1|13 x 13 x 384|ReLU|
|Conv4|384|3 x 3|1|1|13 x 13 x 384|ReLU|
|Conv5|256|3 x 3|1|1|13 x 13 x 256|ReLU|
|Max Pool 3|-|3 x 3|2|-|6 x 6 x 256|-|
|Dropout 1|rate = 0.5|-|-|-|6 x 6 x 256|-|
|Fully Connected 1|-|-|-|-|4096|ReLU|
|Dropout 2|rate = 0.5|-|-|-|4096|-|
|Fully Connected 2|-|-|-|-|4096|ReLU|
|Fully Connected 3|-|-|-|-|1000|Softmax|

**Table 1**: AlexNet Full Layer


**주요 사항**
- 세 번째 convolution layer의 kernel만 두 번째 레이어의 모든 kernel 맵에 연결된다(다른 layer는 동일한 GPU에 올라와 있는 것만 받음).
- fully-connected layer는 모두 이전 layer의 뉴런과 연결되어 있다.
- Local Response Normalization layer는 첫 번째와 두 번째 convolution layer 다음에 있다.
- Max Pooling layer는 Local Response Normalization layer와 5번째 convolution layer 다음에 있다.
- ReLU 함수는 모든 layer의 출력에 적용됐다.

## 4 Reducing Overfitting

큰 네트워크의 사이즈(6천 만개의 파라미터)로 인해 오버피팅이 발생할 것을 예상하여 다음과 같은 방식들을 취했다.

### 4.1 Data Augmentaion

256x256 이미지를 좌우 반전한 후 임의의 224x224 패치를 crop하여 이를 학습에 사용(테스트 시에는 아래와 같은 과정을 따름)
1. 5개(네 모서리와 중앙)의 224x224 이미지 패치를 crop한다.
2. 1에서 얻은 5개의 패치를 좌우 반전한다.
3. 1과 2 총 10개의 패치를 모델에 넣고 결괏값을 평균내어 최종 예측으로 사용한다.

![그림 4](https://github.com/Sameta-cani/jwork/assets/83288284/29959dac-afbb-40e6-8b32-644baa7124e6)

**Figure 4**: Data Augmentaion Ex


훈련 이미지의 RGB 채널 강도를 변경하는 것이다. RGB 이미지의 픽셀을 아래와 같다고 하자. 
$$
I_{xy} = [I^R_{xy}, I^G_{xy}, I^B_{xy}]^T
$$

위 식에 아래 식을 더해준다.

$$
[\mathbf{p}_1, \mathbf{p}_2, \mathbf{p}_3][\alpha_1\lambda_1, \alpha_2\lambda_2, \alpha_3\lambda_3]^T
$$

위 식은 ImageNet 훈련 세트 전체의 RGB 픽셀 값 세트에 대해 PCA를 수행하고 각 훈련 이미지에 발견된 주성분의 배수를 추가한 값이다($\alpha$는 평균 0과 표준편차 0.1을 갖는 표준 가우시안 분포에서 추출한 무작위 변수).

Data Augmentaion을 위한 2가지 방식은 CPU의 Python코드로 생성되며 계산을 거의 하지 않는다. 이 방식은 top-1 error rate을 1%이상 줄여주었다.

### 4.2 Dropout

Dropout 기법을 적용하여 정해진 확률로 각 은닉층 뉴런의 출력을 0으로 설정했다. 이 방식으로 "탈락"된 뉴런은 순방향과 역방향 과정에서 기여하지 않으며, 이는 매 입력마다 다른 아키텍처의 신경망을 샘플링하는 것과 유사하다(여기서 모든 아키텍처는 가중치를 공유).이러한 접근은 네트워크가 특정 뉴런에 과도하게 의존하는 것을 방지하여 일반화 성능을 향상시킬 수 있다.

## Details of learning

**학습 관련 하이퍼 파라미터**
- batch size: 128
- optimizer: stochastic gradient descent(SGD)
- momentum: 0.9
- weight decay: 0.0005
- weight init: zero-mean Gaussian distribution with standard deviation 0.01
- learning rate: 0.001

$$
v_{i+1} := 0.9 \cdot v_i - 0.0005\cdot \epsilon \cdot w_i - \epsilon \cdot <\frac{\partial L}{\partial w}|_{w_i}>_{D_i} \\
w_{i+1} := w_i + v_{i+1}
$$

120만 개의 이미지 훈련셋을 2개의 NVIDIA GTX 580 3GB GPU로 5~6일 동안 90cycle 돌렸다.

## 6 Results

|Model|Top-1|Top-5|
|---|:---:|:---:|
|Sparse coding|47.1%|28.2%|
|SIFT + FVs|45.7%|25.7%|
|CNN|**37.5%**|**17.0%**|

**Table 2**: ILSVRC-2010 테스트셋 결과 비교

|Model|Top-1(val)|Top-5(val)|Top-5(test)|
|---|:---:|:---:|:---:|
|SIFT + FVs|-|-|26.2%|
|1 CNN|40.7%|18.2%|-|
|5 CNNs|38.1%|16.4%|**16.4%**|
|1 CNN*|39.0%|16.6%|-|
|7 CNNs*|36.7%|15.4%|**15.3%**|

**Table 3**: ILSVRC-2012 검증셋, 테스트셋 결과 비교

5 CNN과 7 CNN은 여러 유사 모델들을 학습시킨 후에 평균을 낸 것이다.

### 6.1 Qualitative Evaluations

![그림 5](https://github.com/Sameta-cani/jwork/assets/83288284/34a9c457-14af-49a9-8c45-55315663fd70)

**Figure 5**: 96 convolutional kernels of size 11 x 11 x 3 learned by the first convolutional layer on the 227 x 227 x 3 input images.

Figure 5에서 상단의 48개 이미지는 GPU 1에 의해 학습되었으며, 색에 영향을 받지 않는 모습을 보여준다. 하단의 48개 이미지는 GPU 2에 의해 학습되었고, 색상에 따라 차이가 나타난다.

![그림 6](https://github.com/Sameta-cani/jwork/assets/83288284/2cfb35f6-a468-4f74-97d1-6c5cab54863b)

**Figure 6**: 8개의 ILSVRC-2010 테스트 이미지와 우리 모델이 가장 가능성 높다고 판단한 5개의 라벨

![그림 7](https://github.com/Sameta-cani/jwork/assets/83288284/8ff6254b-8d2e-4ef0-ba80-05853bfc16c7)

**Figure 7**: 첫 번째 열에는 5개의 ILSVRC-2010 테스트 이미지가 표시되어 있다. 나머지 열에서는 각 테스트 이미지의 특징 벡터와 유클리드 거리가 가장 작은 마지막 히든 레이어의 특징 벡터를 가진 6개의 훈련 이미지가 나열되어 있다.

## 7 Discussion

이 연구에서는 크고 깊은 컨볼루션 네트워크를 사용하여 이미지 분류 작업에서 우수한 성능을 달성했으며, 네트워크의 깊이가 성능에 매우 중요하다는 점을 강조했다.

향후 비디오 시퀀스에서 우수한 성능을 달성할 수 있는 컨볼루션 네트워크 연구를 진행하고자 한다.
