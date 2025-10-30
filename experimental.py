import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from docplex.mp.model import Model
from hw1 import make_random_predicate, execute_subsetsums_exact, execute_subsetsums_round ,execute_subsetsums_noise, execute_subsetsums_sample, reconstruction_attack

data: pd.DataFrame = pd.read_csv("https://raw.githubusercontent.com/opendp/cs208/refs/heads/main/spring2025/data/fake_healthcare_dataset_sample100.csv")
pub = ["age", "sex", "blood", "admission"]
target = "result"

# calc

def calc_rsme(real, expected):
    """Calculate RSME between real and expected answers.
    :param real: answers attack
    :param expected: the true answers
    :returns rsme"""
    ans = np.asarray(real).ravel()
    exp = np.asarray(expected).ravel()
    dif = ans - exp
    return float(np.sqrt(np.mean((dif)**2)))

def seccess_rate(data_pub, predicates, answers):
    """Calculate success fraction.
    :param data_pub:
    :param predicates: 
    :param answers: 
    :returns Success fraction"""
        
    recon = reconstruction_attack(data_pub, predicates, answers)
    true = data[target].values
    rate = np.mean(recon == true)
    return rate

# Evaluating

def eval_round(R, predicates, data):
    """compares round vs actual answers
    :param R:
    :param predicates: 
    :param data: 
    :returns rsme and success fraction"""
    exact_answers = execute_subsetsums_exact(predicates)
    atk_answers = execute_subsetsums_round(R, predicates)

    rmse = calc_rsme(exact_answers, atk_answers)
    success = seccess_rate(data, predicates, atk_answers)
    return rmse, success

def eval_noise(sigma, predicates, data):
    """compares noise vs actual answers
    :param sigma:
    :param predicates: 
    :param data: 
    :returns rsme and success fraction"""
    exact_answers = execute_subsetsums_exact(predicates)
    atk_answers = execute_subsetsums_noise(sigma, predicates)

    rmse = calc_rsme(exact_answers, atk_answers)
    success = seccess_rate(data, predicates, atk_answers)
    return rmse, success

def eval_sample(t, predicates, data ):
    """compares noise vs actual answers
    :param t:
    :param predicates: 
    :param data: 
    :returns rsme and success fraction"""
    exact_answers = execute_subsetsums_exact(predicates)
    atk_answers = execute_subsetsums_sample(t, predicates)

    rmse = calc_rsme(exact_answers, atk_answers)
    success = seccess_rate(data, predicates, atk_answers)
    return rmse, success


if __name__ == "__main__":
    majority_baseline = max(data["result"].mean(),1.0-data["result"].mean())
    n=len(data)
    k=2*n
    data_pub=data[pub]
    rr_list, rn_list, rs_list = [], [], []
    sr_list, sn_list, ss_list = [], [], []

    rr_final, rn_final, rs_final = [], [], []
    sr_final, sn_final, ss_final = [], [], []

    for row in range(1, 101):
        print(row)
        for teniterations in range (0,10): 
            predicates = [make_random_predicate() for i in range(k)]
            
            rmse_r, succ_r = eval_round(R=row, predicates=predicates, data=data_pub)
            rmse_n, succ_n = eval_noise(sigma=row, predicates=predicates, data=data_pub)
            rmse_s, succ_s = eval_sample(t=row, predicates=predicates, data=data_pub)


            rr_list.append(rmse_r)
            sr_list.append(succ_r)

            rn_list.append(rmse_n) 
            sn_list.append(succ_n)

            rs_list.append(rmse_s)
            ss_list.append(succ_s)

        rr_ave = sum(rr_list)/10
        sr_ave = sum(sr_list)/10
        rn_ave = sum(rn_list)/10
        sn_ave = sum(sn_list)/10
        rs_ave = sum(rs_list)/10
        ss_ave = sum(ss_list)/10

        rr_list.clear()
        sr_list.clear()
        rn_list.clear()
        sn_list.clear()
        rs_list.clear()
        ss_list.clear()

        rr_final.append(rr_ave)
        sr_final.append(sr_ave)
        rn_final.append(rn_ave)
        sn_final.append(sn_ave)
        rs_final.append(rs_ave)
        ss_final.append(ss_ave)

##### CSV #####
    csv = pd.DataFrame(list(zip(*[rr_final,sr_final,rn_final,sn_final,rs_final,ss_final]))).add_prefix('Col')
    csv.to_csv('RSME_SUCCESS_DATA')

##### PLOTS #####   
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(rr_final)), rr_final, label="Round")
    plt.xlabel("Parameter = R")
    plt.ylabel("RMSE")
    plt.title("RMSE vs R")
    plt.legend()
    plt.grid(True)


    plt.figure(figsize=(10, 5))
    plt.plot(range(len(sr_final)), sr_final, label="Round")
    plt.axhline(y= majority_baseline, linestyle = ':', label= "Majority Value")
    plt.xlabel("Parameter = R")
    plt.ylabel("Success")
    plt.title("Success vs R")
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(rn_final)), rn_final, label="Noise")
    plt.xlabel("Parameter = Sigma")
    plt.ylabel("RMSE")
    plt.title("RMSE vs Sigma")
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(sn_final)), sn_final, label="Noise")
    plt.axhline(y= majority_baseline, linestyle = ':', label= "Majority Value")
    plt.xlabel("Parameter = Sigma")
    plt.ylabel("Success")
    plt.title("Success vs Sigma")
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(rs_final)), rs_final, label="Sample")
    plt.xlabel("Parameter = Sigma")
    plt.ylabel("RMSE")
    plt.title("RMSE vs Sigma")
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(ss_final)), ss_final, label="Sample")
    plt.axhline(y= majority_baseline, linestyle = ':', label= "Majority Value")
    plt.xlabel("Parameter = t")
    plt.ylabel("Success")
    plt.title("Success vs Sigma")
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 5))
    plt.plot(rs_final, ss_final, label="Sample")
    plt.xlabel("RMSE")
    plt.ylabel("Success")
    plt.title("Sample RMSE vs Success")
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 5))
    plt.plot(rr_final, sr_final, label="Round")
    plt.xlabel("RMSE")
    plt.ylabel("Success")
    plt.title("Round RMSE vs Success")
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 5))
    plt.plot(rn_final, sn_final, label="Noise")
    plt.xlabel("RMSE")
    plt.ylabel("Success")
    plt.title("Noise RMSE vs Success")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()