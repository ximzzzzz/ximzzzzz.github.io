# Norm 겉핥기



여러 예측모델을 공부하다보면 자주 마주치는 단어가 있다.



### 'regularization'



낱개로 이해하려면 어렵지만  overfitting 과 함께 생각하면 이해가 쉽다



![overfitting](https://i.stack.imgur.com/t0zit.png)



위 그림에서 overfit 과 underfit 의 차이점은 뭘까?

바로 변수x가 overfit에 비해 덜 사용되고 있다는 점이다. 

또한 x 세제곱, 네제곱과 같이 복잡하게 사용되지 않는다는 점이다.



그렇다면, overfit의 모델을 좀 더 일반화하기 위해서 우리는 어떻게 해야할까?



##### 첫번째는 단순하게 진짜 변수를 덜 사용하는 것이다. 

데이터에 여러변수가 있을때 모두 사용하는 것이 아닌 중요한 것만 뽑아서 사용한다.

하지만 만약 지금 사용하는 변수를 모두 사용해야 할때는 어떻게 해야할까?



##### 두번째는 다항식의 계수를 0에 가깝게 만듦으로써, 다항식을 의미없게(약하게) 만드는 것이다.



예시를 통해 확인해보자

| Graph                                                        | formula                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src='https://user-images.githubusercontent.com/44566113/52961724-f3598f80-33de-11e9-95ca-afce9604d3b2.png' width=350> | <img src='https://user-images.githubusercontent.com/44566113/52962293-38ca8c80-33e0-11e9-9b90-65b4d328f791.PNG' width='500' > |



위 그래프에서 볼 수 있듯이 5차 다항식으로 이루어진 빨간 그래프는 매우 난해하다

반면, 1차로 이루어진 선형그래프 보라색은 매우 단순하다

마지막으로, 5차 다항식으로 이루어졌지만 계수를 0.001과 같이 0에 가깝게 만들었더니

선형에 가까울 정도로 단순해진다는 걸 확인할 수 있다.



그렇다면 우리는 어떻게 계수(weight)를0에 가깝게 만들 수 있을까?



정답은 **'penalty'** 를 주는 것이다.  

복잡한 모델일 수록 모델의 복잡도에 비례하여)큰 penalty를 줌으로써 계수를 0에 가깝게 만드는 것이다.

이때 penalty를 주기위해 모델의 복잡도를 측정하는 방식이 바로 `L2 norm` 이라 할 수 있다.





































출처  

https://www.geeksforgeeks.org/underfitting-and-overfitting-in-machine-learning/

https://www.desmos.com/calculator







