# CS5510-HW1
Authors: Kai Xue
Professor: Mingyan Xiao
Course: CS5510 Data Privacy and Security
Date: 29 October 2025

---
## Question One: Reconstruction Attack

### Part a
```
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
```
### Part b
####Round
```
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
```
####Noise
```
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
```
####Sample
```
def execute_subsetsums_sample(t, predicates):
    """Count the number of patients that satisfy each predicate using t length subsets.
    :param t: noise 
    :param predicates: a list of predicates on the public variables
    :returns a 1-d np.ndarray of exact answers the subset sum queries"""
    # geting exact to edit
    temp = data[target].values
    n= len(data)
    # getting sample 
    sample  = np.random.choice(n, size=t, replace=False)
    sample = temp[sample] @ np.stack([pred(data)[sample] for pred in predicates], axis=1)

    return (n/t) * sample
```

### Part c

---
## Quetion Two: A Bayesian Interpretation of MIAs

###Part a

###Part b
