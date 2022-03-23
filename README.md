# kaggle_store_sales

Kaggle Store Sales의 일자별 점포별 품목별로 다양한 feature와 모델을 실험하여 최고 스코어를 산출하는 모듈입니다.


##### 폴더구조
|폴더|설명|
|:---:|:---|
|src|프로그램|
|input|Store Sales 제공 데이터. input.zip 압축해제요  |
|output|시나리오별 predict 데이터|

##### 변경사항
소스내 경로수정 path = "d:/lge/pycharm-projects/kaggle_store_sales/input/"

#### 실험 결과
![image](https://user-images.githubusercontent.com/20777148/159754298-d54ee31d-1e05-4b07-a392-36ed05fef266.png)

#### scenario_id
x001d001y001m001 : x001(feature) d001(feature 날짜구간) y001(sales) m001(모델종류)  
예시>  
x001d001y001m001 : scenario_model_linearregression.py  
x001d001y001m002 : scenario_model_ridge.py  
x001d001y001m003 : scenario_model_customeregressor.py  

#### s10 ~ s90 : 상점별 품목별 스코어(1-mse) 

#### 챠트  

plot_store('x001d001y001m001', 2) 시나리오별 상점별 챠트 출력

![image](https://user-images.githubusercontent.com/20777148/159756750-c776fdbd-ddf4-4481-a946-580181128de1.png)

## submition 전략
아래와 그림과 같이 시나리오별 상점별 품목별로 스코어가 산출되므로 스코어가 낮은 상점 및 품목을 대상으로 최적화된 시나리오 추가시
높은 스코어 달성이 가능할것으로 예상

![image](https://user-images.githubusercontent.com/20777148/159757168-ca0c6dbe-1b1b-4527-a43b-448e2f5416ec.png)

##### 상점별 품목별 최고 스코어 산출 쿼리
select max(a.scenario_id) as scenario_id, a.store_nbr, a.family2, a.score  
from   results a,  
	  ( select store_nbr, family2, max(score) score  
		from results  
		group by store_nbr, family2  
	  ) b  
where a.store_nbr = b.store_nbr  
and   a.family2 = b.family2  
and   a.score = b.score  
group by a.store_nbr, a.family2, a.score  
order by a.store_nbr, a.family2, a.score  





