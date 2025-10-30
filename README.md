# CS5510-HW1
Authors: Kai Xue

Professor: Mingyan Xiao

Course: CS5510 Data Privacy and Security

Date: 29 October 2025

---
## Question One: Reconstruction Attack
### PART A
**Implementation/ Method:** Use Least-Squares to solve Reconstruction attack
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

### PART B
**i) Round - ound each result to the nearest multiple of R.**
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

**ii) Noise - add independent Gaussian noise of mean zero and variance σ2 to each result.**
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

**iii) Sample - given a parameter t ∈ {1, . . . , n}, randomly subsample a set T consisting of t out of the n rows and calculate all of the answers using only the rows in T (scaling up answers by a factor of n/t).**
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

### PART C
**Compute Accuracy**
```
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
```

**Success Fraction vs Parameter AND RMSE vs Parameter AND RMSE vs Success Fraction**
---
**ROUNDING**

Transition: At R=10 (.64799999)

Results: Sucess Fraction went down as R increased, but started to rise back to Majority Value at 40 and reached Majority value consistantly past 60. Rounding was the stronggest around R=20 through R=40. Rounding was not a successful defensive measure from R=1 through R=10.

![alt text](https://github.com/kaicx1/CS5510-HW1/blob/main/roundgraphs.png)
![alt_text](https://github.com/kaicx1/CS5510-HW1/blob/main/roundNvsS.png)
---
NOISE

Transition: At R=4 (.517)

Results: At low noise levels (Sigma) reconstruction was nearly perfect. The addition of Gaussain noise performs well when Sigma is close to 5 as recontruction success is .385 while still retaining accuracy. The graph shows an inverse logrithmic curve showing that after Sigma passes ~20 there is diminishing returns. Smaller Sigma provided low privacy with higher accuracy. Larger Sigma provides higher privacy with lower accuracy. The Success vs RSME graph shows a reverse logmarithic graph.

![alt_text](https://github.com/kaicx1/CS5510-HW1/blob/main/noisegraphs.png)
![alt_text](https://github.com/kaicx1/CS5510-HW1/blob/main/noiseNvsS.png)
---
SAMPLING

Transition: N/A when t=1 (.6500000000001)

Results: When t is small we see a lower reconstruction rate that never crosses Majority Value, but as t grows the rate of reconstruction is hightend. Larger Samples yield a higher reconstruction rate. When the Samples sizes t reach around 70 the reconstruction rate jumps to around .90. Sampling seems to best used around t = 65 as accuracy is relativly high without the trade off of higher reconstruction attack success rate. The Success vs RSME graph shows an inverse logmarithic curve indicating that t = ~65 is a area that supports both accuracy and privacy. 

![alt_text](https://github.com/kaicx1/CS5510-HW1/blob/main/samplegraphs.png)
![alt_text](https://github.com/kaicx1/CS5510-HW1/blob/main/sampleNvsS.png)
---
## Quetion Two: A Bayesian Interpretation of MIAs

### PART A
**Suppose an attacker carries out a Membership Inference Attack on Alice and receives an “In” result. Let Opost be the odds corresponding to Alice’s belief conditioned on “In” resultfrom the MIA. Write a formula for Opost in terms of Oprior and the TPR and FPR of theMIA (on the same data distribution).**

$O_{\text{prior}} = \frac {p}{1-p}$

TPR = P(MIA SAYS 'in' | ALICE IS 'in')

FPR = P(MIA SAYS 'in' | ALICE NOT 'in')

$O_{\text{post}} = \frac {P(member|'in')}{P(non-member|'in')}$

Posterior odds:

$O_{\text{post}} = O_{\text{prior}} * \frac {TPR}{FPR}$

### PART B
TPR = Porportion of correctly classified positive cases
FPR = Porportion of 
TPR checks if the amount of times positives are correct, but does not check if false positives are in the dataset. If FPR is large, even if TPR is large or TPR = 1, the attack can not be confident as many false positives ("in" results) would be false. 
