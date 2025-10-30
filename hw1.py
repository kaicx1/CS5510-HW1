# Starter code for Homework 2.
import numpy as np
import pandas as pd

# Problem setup

# Update to point to the dataset on your machine
data: pd.DataFrame = pd.read_csv("https://raw.githubusercontent.com/opendp/cs208/refs/heads/main/spring2025/data/fake_healthcare_dataset_sample100.csv")

# names of public identifier columns
pub = ["age", "sex", "blood", "admission"]

# variable to reconstruct
target = "result"

def make_random_predicate():
    """Returns a (pseudo)random predicate function by hashing public identifiers."""
    prime = 2003
    desc = np.random.randint(prime, size=len(pub))
    # this predicate maps data into a 1-d ndarray of booleans
    #   (where `@` is the dot product and `%` modulus)
    return lambda data: ((data[pub].values @ desc) % prime % 2).astype(bool)

def execute_subsetsums_exact(predicates):
    """Count the number of patients that satisfy each predicate.
    Resembles a public query interface on a sequestered dataset.
    Computed as in equation (1).

    :param predicates: a list of predicates on the public variables
    :returns a 1-d np.ndarray of exact answers the subset sum queries"""
    return data[target].values @ np.stack([pred(data) for pred in predicates], axis=1)

# TODO: write exexcte_subsetsums_round(R,predicates), exexcte_subsetsums_noise(sigma,predicates), exexcte_subsetsums_sample(t,predicates)
def execute_subsetsums_round(R, predicates):
    """Count the number of patients that satisfy each predicate result rounded to R.
    :param R: positive integer 
    :param predicates: a list of predicates on the public variables
    :returns a 1-d np.ndarray of exact answers the subset sum queries"""
    # geting exact to edit
    temp = np.asarray(execute_subsetsums_exact(predicates),dtype=float).ravel()
    #rounding
    round = np.round(temp / float(R)) * float(R)
    return round

def execute_subsetsums_noise(sigma, predicates):
    """Count the number of patients that satisfy each predicate with additional noise.
    :param sigma: noise 
    :param predicates: a list of predicates on the public variables
    :returns a 1-d np.ndarray of exact answers the subset sum queries"""
    # geting exact to edit
    temp = np.asarray(execute_subsetsums_exact(predicates),dtype=float).ravel()
    # adding noise
    noise = temp + np.random.normal(0,sigma,temp.shape)
    return noise

def execute_subsetsums_sample(t, predicates):
    """Count the number of patients that satisfy each predicate with additional noise.
    :param sigma: noise 
    :param predicates: a list of predicates on the public variables
    :returns a 1-d np.ndarray of exact answers the subset sum queries"""
    # geting exact to edit
    temp = data[target].values
    n= len(data)
    # getting sample 
    sample  = np.random.choice(n, size=t, replace=False)
    sample = temp[sample] @ np.stack([pred(data)[sample] for pred in predicates], axis=1)

    return (n/t) * sample


# TODO: Write the reconstruction function!
def reconstruction_attack(data_pub, predicates, answers):
    """Reconstructs a target column based on the `answers` to queries about `data`.

    :param data_pub: data of length n consisting of public identifiers
    :param predicates: a list of k predicate functions
    :param answers: a list of k answers to a query on data filtered by the k predicates
    :return 1-dimensional boolean ndarray"""

    # make masks for each RANDOM predicate
    masks = [np.asarray(p(data_pub)).astype(float).ravel() for p in predicates]
    # convert from 1D to 2D array
    predMatrix = np.stack([m for m in masks], axis= 0)
    # create 1D matrix of answers( answers = array[hiddenbits] * array[predicatemask])
    bMatrix = np.asarray(answers, dtype=float).ravel()

    # calculating least squares
    results,*_ = np.linalg.lstsq(predMatrix,bMatrix,rcond=None)

    return np.round(results)



if __name__ == "__main__":
    # EXAMPLE: writing and using predicates
    num_female_patients, num_emergency_admits , num_sixtythree= execute_subsetsums_exact([
        lambda data: data['sex'] == 1,      # "is-female" predicate
        lambda data: data['admission'] == 2,  # "had emergency admission" predicate
        lambda data: data['age'] == 63,
    ])

    print(num_female_patients)
    print(num_emergency_admits)
    print(num_sixtythree)
    # EXAMPLE: making and using a random predicate
    example_predicate = make_random_predicate()
    num_patients_that_matched_random_predicate = execute_subsetsums_exact([example_predicate])
    print(num_patients_that_matched_random_predicate)

    # The boolean mask from applying the example predicate to the data:
    example_predicate_mask = example_predicate(data)
    
# TODO: Write the main
    n=len(data)
    k=2*n
    data_pub=data[pub]

    predicates = [make_random_predicate() for i in range(k)]

    answers = execute_subsetsums_exact(predicates)
    answersRound = execute_subsetsums_round(5, predicates)
    answersNoise = execute_subsetsums_noise(5.0,predicates)
    answersSample = execute_subsetsums_sample(100, predicates)
    
    recon0 = reconstruction_attack(data_pub, predicates, answers)
    recon1 = reconstruction_attack(data_pub, predicates, answersRound)
    recon2 = reconstruction_attack(data_pub, predicates, answersNoise)
    recon3 = reconstruction_attack(data_pub, predicates, answersSample)

    true_bits = data[target].astype(bool).values

    accuracy = (recon0 == true_bits).mean()
    mismatch = np.flatnonzero(recon0 != true_bits)

    print("results")
    print(f"n = {n}, k = {k}")
    print(f"Accuracy: {accuracy*100:.4f}%")
    if len(mismatch) > 0:
        print(mismatch[:20])


    accuracy = (recon1 == true_bits).mean()
    mismatch = np.flatnonzero(recon1 != true_bits)

    print("results")
    print(f"n = {n}, k = {k}")
    print(f"Accuracy: {accuracy*100:.4f}%")
    if len(mismatch) > 0:
        print(mismatch[:20])

    accuracy = (recon2 == true_bits).mean()
    mismatch = np.flatnonzero(recon2 != true_bits)

    print("results")
    print(f"n = {n}, k = {k}")
    print(f"Accuracy: {accuracy*100:.4f}%")
    if len(mismatch) > 0:
        print(mismatch[:20])

    
    accuracy = (recon3 == true_bits).mean()
    mismatch = np.flatnonzero(recon3 != true_bits)

    print("results")
    print(f"n = {n}, k = {k}")
    print(f"Accuracy: {accuracy*100:.4f}%")
    if len(mismatch) > 0:
        print(mismatch[:20])


###############################################################################################################################
#CHECKS#
#correct masking check.
    print(answersNoise) 
'''
    data_pub=data[pub]
    predicates = [
            lambda data: data['sex'] == 1,      # "is-female" predicate
            lambda data: data['admission'] == 2,  # "had emergency admission" predicate
    ]
    masks = [np.asarray(p(data_pub)).ravel() for p in predicates]
    print(masks[0])
    print(masks[1])

    print(answers)
    print(answersRound)
    print(answersNoise)
    print(answersSample)
'''
##################################################################################################################################