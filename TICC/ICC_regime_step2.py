from TICC_solver import TICC
import numpy as np
import sys
import warnings
import pandas as pd
from scipy.optimize import minimize
from typing import List

#warnings.filterwarnings(action='ignore')

def obj_sharpe(weights, returns, covmat):
    ret = np.dot(weights.T, returns)
    vol = np.sqrt(np.dot(weights, np.dot(covmat, weights)))
    return 1 / ((ret) / np.sqrt(vol))


TOLERANCE = 1e-20


def get_weights(num_of_asset: int):
    # Equal weight example.
    weight = 1 / num_of_asset
    weights = np.repeat(weight, num_of_asset)
    return weights


def get_covariances(data):
    # Calculate covariance matrix.
    covariance = data.cov()
    return covariance


def _get_risk_contribution(weights, covariance):
    # Optional: convert dataframe to numpy array.
    covariance = covariance.to_numpy()
    # Convert 1d array to 2d array (n x 1).
    weights = weights.reshape(-1, 1)
    # Calculate portfolio variance.
    variance = weights.T @ covariance @ weights
    # Calculate portfolio sigma.
    sigma = np.sqrt(variance)
    # Calculate mrc.
    mrc = 1 / sigma * (covariance @ weights)
    # Calculate rc.
    rc = np.multiply(weights, mrc)
    # Normalize.
    rc = rc / sum(rc)
    return rc


def _get_error(weights, covariance):
    # Get rc.
    rc_matrix = _get_risk_contribution(weights, covariance)
    # Calculate error.
    error = np.sum(np.square(rc_matrix - rc_matrix.T))
    return error


def _get_opt_weights(covariance: List[float]):
    # Constraints: sum of weights is 1, weight is more than 0.
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}, {'type': 'ineq', 'fun': lambda x: x})
    # Setting max iteration number.
    options = {'maxiter': 800}
    # Get initial weights (Equal weights).
    initial_weights = get_weights(len(covariance))
    # Optimisation process in scipy.
    optimize_result = minimize(fun=_get_error,
                               x0=initial_weights,
                               args=covariance,
                               method='SLSQP',
                               constraints=constraints,
                               tol=TOLERANCE,
                               options=options)
    # Recover the weights from the optimised object.
    weights = optimize_result.x
    # It returns the optimised weights.
    return weights




def generate_weight(rebalance_day_list,all_index_data,profit_data,window_size = 5,regime_split = 3):

    for rebalance_day_index, rebalance_day in enumerate(rebalance_day_list):
        print('현재 리밸런싱 날짜는 '+str(rebalance_day)[:10]+' 입니다')

        # 사후 편향을 없애기 위해 리밸런싱 날짜 이전 데이터만 가져옵니다.
        train_data = all_index_data.loc[:rebalance_day, :]
        past_profit_data = profit_data.loc[:rebalance_day, :]

        #### 과거 데이터를 이용해 ICC 알고리즘을 실행하는 코드입니다.###
        ticc = TICC(window_size=window_size, number_of_clusters=regime_split, lambda_parameter=11e-2, beta=300,
                    maxIters=100,
                    threshold=2e-5,
                    write_out_file=False, prefix_string="output_folder/", num_proc=11, compute_BIC=True)
        (cluster_assignment, cluster_MRFs, BIC) = ticc.fit(input_file=train_data)

        #### 과거 데이터를 이용해 ICC 알고리즘을 실행하는 코드입니다.###
        # 이후 해당 데이터를 cluster_data로 저장해줍니다.
        cluster_data = pd.DataFrame(cluster_assignment)
        cluster_data.index = train_data.index[window_size - 1:]
        cluster_data.columns = ['regime']

        #저장해준 데이터를 과거 past_profit_data 데이터프레임에 붙혀줍니다. 이는 최근 국면과 같은 국면일때의 데이터를 추출하기 위함입니다.
        past_profit_data = past_profit_data.join(cluster_data)
        past_profit_data_merge = past_profit_data.iloc[window_size - 1:]

        #가장 최근 국면은 과거 5주 중에 가장 많은 국면을 보인 국면으로 판단합니다.
        recent_regime_data = past_profit_data_merge.regime.iloc[-5:]
        now_regime = max(recent_regime_data, key=list(recent_regime_data).count)

        # 현재 국면과 동일한 과거 국면들만 모아서 따로 past_profit_data_merge라는 데이터프레임에 저장해줍니다.
        past_profit_data_merge = past_profit_data_merge.loc[past_profit_data_merge.regime == now_regime]

        ###### 이제부터는 해당 국면의 과거 자료를 통해 MVO(risk parity) 최적화를 해주는 과정입니다.

        #수익률 데이터프레임에서 주요 자산군의 수익률자료만 빼와 따로 저장해줍니다.

        data = past_profit_data_merge.loc[:, ['dollar', 'highyield', 'TIPS', 'US_bond', 'gold']].fillna(0)
        data_columns = data.columns  # 자산군 칼럼을 미리 저장해줍니다.


        covmat = data.cov() #MVO에 쓰일 covariance
        ret_annual = data.mean() # MVO에 쓰일 평균

        n_assets = len(data_columns) # 자산개수는 주요 자산군의 개수와 동일합니다.

        # 초기 투자 비중은 종목 개수만큼 균등하게 정합니다.
        weights = np.ones([n_assets]) / n_assets

        # 투자 비중의 범위는 0~50%이고, 이를 종목 개수만큼 튜플로 만들어줍니다. 상한선을 정해준 이유는 과거 추정 값이 정확하지 않아 자산의 쏠림이 발생할 수 있기에 인위적으로 설정해주었습니다.
        bnds = tuple((0, 0.5) for i in range(n_assets))

        # 제약식은 람다(lambda) 함수로 만들어줍니다.
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

        # minimize( ) 함수에 목적함수, 초기 투자 비중, 투자 비중을 제외한 목적함수에 전달할 나머지 매개변수, 최적화 알고리즘, 범위, 제약조건을 전달해줍니다.
        res = minimize(obj_sharpe, weights, (ret_annual, covmat), method='SLSQP', bounds=bnds, constraints=cons)

        #최종적으로 저장된 weight값을 weight_pd에 저장해줍니다.
        weight_mvo = res['x']

        #### 밑 코드는 risk parity 코드입니다.
        # Get covariance.
        covariance_matrix = get_covariances(data)
        # Get optimum weights.
        weight = _get_opt_weights(covariance_matrix)
        weight = list(np.array(weight))
        ###########


        if rebalance_day_index == 0:
            weight_pd = pd.DataFrame(weight).T
            weight_pd_mvo = pd.DataFrame(weight_mvo).T
        else:
            weight_pd = pd.concat([weight_pd, pd.DataFrame(weight).T], axis=0)
            weight_pd_mvo = pd.concat([weight_pd_mvo, pd.DataFrame(weight_mvo).T], axis=0)
        print(weight_pd)
        print(weight_pd_mvo)

    weight_pd.columns = data_columns
    weight_pd_mvo.columns = data_columns
    weight_pd.index = rebalance_day_list
    weight_pd_mvo.index = rebalance_day_list
    weight_pd.to_excel('weight_pd_risk.xlsx') #리스크패리티
    weight_pd_mvo.to_excel('weight_pd_mvo.xlsx') #mvo

    return weight_pd,weight_pd_mvo








if __name__ == '__main__':

    directory = 'C:/Users/User/ficc_quant/'
    all_index_data = pd.read_excel(directory+'all_index_data.xlsx', index_col=0,engine='openpyxl') # 국면 데이터 input으로 쓰이는 값
    profit_data=pd.read_excel(directory+'price_change.xlsx', index_col=0,engine='openpyxl') # 비중 최적화에 쓰일 데이터
    print(profit_data)

    print(all_index_data.columns)


    rebalance_day_list=pd.date_range('2013-12-31', '2022-12-01', freq='2m') # 리밸런싱을 하는 날짜입니다. 매달 리밸런싱을 진행합니다.


    generate_weight(rebalance_day_list, all_index_data, profit_data,window_size=7, regime_split=3)









