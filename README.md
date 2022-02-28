# skin
## Problems
* 사람의 피부 유형은 오일, 민감, 색소, 주름 4가지 지표에 따라 16가지로 나눌 수 있음
* 사람의 피부 유형을 구분할 수 있는 피부 유형 검사는 문항 수가 많은 문제가 있음
* 이에 참가자는 이용에 불편함을 느낌

## Goal
* 사람 얼굴 사진을 받아 피부 검사 문항의 답을 예측함
* 이를 통해 잘 예측 할 수 있는 문항은 삭제하여 문항 수 최소화

## Requirements
* ubuntu 18.04
* python 3.8.3
* CUDA 11.3.0
* pytorch
* torchvision

## Datasets
* 사람 얼굴 이미지
* MTCNN을 통해 얼굴 부분만 탐지하여 활용

## Arguments
|Args|Type|Description|Default|
|----|----|----|----|
|arch|[str]| ResNet50, EfficientNet b7, ViT Large, Mixer Large | ResNet50|
|optim |[str]| SGD, Adam| Adam |
|lr  |[float] | learning rate|1e-4(decay=0.1)|
|weight-decay| [float] | weight decay |5e-4|
|epoch     |[int] |epochs|50|
|train-batch |[int]|batch size|128 |
|save_dir |[str]| save files dir| - |

## Run
```
python train.py --arch 'model_you_want' --epoch 50 --lr 0.0001 --save_dir 'your_save_path'
```

## Results
### graph
![image](https://user-images.githubusercontent.com/86753195/143201237-39daac65-09e9-41ff-a2bd-af54a82e6aa3.png)
* ResNet50 아키텍처를 통해 학습하였을 때, 기존 성능에 버금가는 모습을 보여줌
* 특기할 점은 ResNet과 EfficientNet의 성능이 최근 비전에서 각광받는 transformer를 접목한 모델 ViT보다 좋았음
* 이는 얼굴 사진은 항상 유사한 형태를 띄는데, 이러한 상황에서 inductive bias를 가지는 CNN이 더 좋은 성능을 발휘 할 수 있었다고 생각해볼 수 있음

### table
![image](https://user-images.githubusercontent.com/86753195/143365352-bd9c984c-7b5f-4681-b323-6c0afb67edf9.png)
* 문항별 편차를 살펴보았을 때, 전체적으로 기존 연구와 비슷한 양상을 띄는 것을 확인할 수 있음
* 색으로 칠한 문항은 평균 오차가 0.6 미만인 문항들로 충분히 잘 예측하여 삭제할 수 있는 문항
* 기존 연구 대비 새로 제시한 모델은 삭제할 수 있는 문항이 더 많아짐을 알 수 있음