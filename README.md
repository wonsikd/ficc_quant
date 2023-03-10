# ficc_quant

## inverse covariance clustering의 국면을 활용한 동적 포트폴리오
## 하나증권 퀀트 챌린지
##  본 프로젝트는 하나증권에서 주최한 FICC QUANT 부문의 프로젝트를 위한 자료입니다.

### 다음으로는 코드의 과정을 설명 드리겠습니다. 모두 class화를 하지 않고 데이터 정제와 자산군 속 세부 비중설정은 주피터 노트북으로 과정을 담았습니다. 이는 데이터 전처리 과정이 복잡하여 이해가 안되실 수 있는 점을 감안하여 파트를 총 3개로 나누어 코딩을 진행하였습니다.
전체적인 코드 실행 과정은 다음과 같습니다.
### 1)	Data_preprocessing_step1
### 해당 주피터노트북 파일은 국면모델에 넣어줄 인풋 데이터를 정제해주는 파일입니다. 아웃풋 값으로는 all_ETF_data과 all_index_data 데이터를 내보냅니다. all_ETF_data는 비중 최적화를 해주기 위해 필요한 ETF 과거 데이터입니다. all_index_data는 국면 모델에 필요한 금융위험지표와 기타 인덱스 지표들이 담겨있습니다.
### 2)	ficc_quant/TICC/ ICC_regime_step2.py
### 해당 파일은 Data_preprocessing_step1에서 나온 데이터들을 활용하여 국면을 날짜마다 예측하고, 국면을 활용해 MVO 최적화를 해주는 코드입니다. 날짜별 자산군 투자 비중이 담긴 데이터프레임이 output 값으로 산출되며, 이를 활용하여 자산 군집 내에서 step3코드에서 risk parity 전략을 실행될 수 있게끔 해줍니다.
### 3)	weight_riskparity_step3
### 해당 파일은 3번째 단계로 step2에서 나온 비중 값을 가지고 자산군 내에서 risk parity 전략을 시행하여 최종 비중을 산출되게끔 해줍니다. Output 파일은 ‘last_portfolio_weight.xlsx’파일이며, 날짜별 ETF의 비중이 담겨있습니다. 또한, 백테스팅을 하기 위한 ‘all_ETF_data.xlsx’ 파일 역시 내보냅니다. 해당 파일은 백테스팅에 필요한 ETF의 과거 데이터를 변화율 형태로 변환해줍니다.
### 4)	Backtesting_step4
### 해당 파일은 최종적으로 나온 비중을 가지고 백테스팅을 하는 코드입니다. 백테스팅의 비교 벤치마크는 동일비중 투자 포트폴리오입니다.
