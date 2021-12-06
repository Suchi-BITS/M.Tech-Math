#Q1. (i) Find the approximate time your computer takes for a single addition by adding first 106 positive integers using a for loop and dividing the time taken by 106 Similarly find the approximate time taken for a single multiplication and division. Report the result obtained in the form of a table.
Sol: 
#Q1)i: for Addition
import time

# starting time
start1 = time.time()
sum = 0
for num in range(0,1000001):
    sum = sum+num
end1 = time.time()
time_taken1=(end1 - start1)/1000000
print(f"Runtime of the program is for addition is in seconds", time_taken1)

div=1
start2 = time.time()
for v in range(1, 1000001):
  div=div/v
end2=time.time()
time_taken2=(end2-start2)/1000000
print("Runtime of the program is for division is ", time_taken2)



start3 = time.time()
product = 1
for n in range(1, 1000001):
    product = product*n
end3 = time.time()
time_taken3=(end3-start3)/1000000
# total time taken
print("Runtime of the program is for multiplication is ", time_taken3)

#!/usr/bin/env python
# coding: utf-8

# In[32]:


import time
import random
import pandas as pd
from tqdm import tqdm
from copy import copy, deepcopy
from pprint import pprint
from functools import partial
import math
from matplotlib import pyplot as plt


# In[7]:


def roundto_d_significant(value, sig_digits):
    if value == 0:
        return 0
    return round(value, sig_digits - int(math.floor(math.log10(abs(value)))) - 1)


def rank_of_matrix(matrix):
    rank = 0
    for row in matrix:
        if sum(row) != 0:
            rank += 1
    return rank
    

def gauss_elimination(A: list, b: list, partial_pivoting: bool = False, d: int = 3):
    """
    A: A square matrix n*n
    b: A vecor of size n
    partial_pivoting: Whether to perform partial pivoting or not
    d: number of significant digits to be rounded to
    """
    assert len(A) == len(b) # No of equations is equal to length of vector
    for row in A:
        assert len(row) == len(A) # Assert square matrix
        
    rounding = partial(roundto_d_significant, sig_digits = d)
    
    # Create augmented matrix
    for i in range(len(A)):
        A[i].append(b[i])
    
    # Bringing the matrix to reduced echelon form (REF)
    for index_row in range(0, len(A)-1):
        
        # Partial pivoting
        if partial_pivoting:
            cur_max = abs(A[index_row][index_row])
            cur_max_row = index_row
            # Check if pivoting need to be done
            for i in range(index_row + 1, len(A)):
                if abs(A[i][index_row]) > cur_max:
                    cur_max = abs(A[i][index_row]) # Absolute value considered for pivoting
                    cur_max_row = i
            # If pivoting need to be done
            if cur_max_row != index_row:
                temp = A[index_row]
                A[index_row] = A[cur_max_row]
                A[cur_max_row] = temp
        
        for i in range(index_row + 1, len(A)):
            # Skip the rwo transform if the value is already zero
            if A[i][index_row] == 0:
                continue
            
            # Calculate the coefficient to mutiply with the index row
            scaler = rounding(A[i][index_row] / A[index_row][index_row])
            if (A[i][index_row] < 0 and A[index_row][index_row] < 0) or ((A[i][index_row] > 0 and A[index_row][index_row] > 0)):
                scaler = -scaler
            
            # Assign all the prior values to zero
            for j in range(0, index_row + 1):
                A[i][j] = 0

            # Compute rest of the values in the row
            for j in range(index_row + 1, len(A[i])):
                A[i][j] = rounding(A[i][j] + (scaler * A[index_row][j]))
                
                    
    # Back substitution

    # Initialize None for all variables
    variable_values = {}
    for i in range(len(A)):
        variable_values[f'x{i}'] = None
    
    # Assign arbitrary values if needed
    no_aug_A = [[A[i][j] for j in range(len(A[i]) - 1)] for i in range(len(A))]
    if rank_of_matrix(no_aug_A) != rank_of_matrix(A):
        print("The system is inconsistent!!")
        return
    else:
        if rank_of_matrix(A) < len(A):
            for i in range(len(A)-1, rank_of_matrix(A)-1, -1):
                variable_values[f'x{i}'] = rounding(random.random())
        
        # Solve different equations for different variables
        for i in range(rank_of_matrix(no_aug_A) - 1, -1, -1):
            known_coeffs = [A[i][j] * variable_values[f'x{j}'] for j in range(i+1, len(A[i]) - 1)]
            known_coeffs = [rounding(v) for v in known_coeffs]
            rhs = rounding((A[i][-1] - rounding(sum(known_coeffs))))
            variable_values[f'x{i}'] =  rounding(rhs / A[i][i])

    return variable_values


# In[12]:


gauss_elimination(A=[[2, 1, 1], [2, 4, 1], [5, 7, 4]], b=[3, 8, 22], partial_pivoting=False, d=5)


# In[13]:


gauss_elimination(A=[[2, 1, 1], [2, 4, 1], [5,7, 4]], b=[3, 8, 22], partial_pivoting=True, d=5)


# In[15]:


def guass_operation_count(n: int):
    """
    n: Number of row / columns in a square matrix
    """
    # Operation count for addition
    ref_addition = (n * (n+1) * (2 * n + 1)) / 6 # Number of addition for REF
    backsub_addition = (n * (n - 1)) / 2
    total_addition = ref_addition + backsub_addition
    
    # Operation count for multiplication
    total_multiplication = total_addition # Same as addition
    
    # Operation count for division
    ref_division = (n * (n-1)) / 2
    backsub_division = n
    total_division = ref_division + backsub_division

    # Summarize
    operation_count = {
        'addition': total_addition,
        'multiplication': total_multiplication,
        'division': total_division
    }
    
    return operation_count


# In[16]:


guass_operation_count(3)


# In[17]:


def generate_test_data():
    test_cases = {}
    for case in range(100, 1100, 100):
        A = [[roundto_d_significant(random.random(), sig_digits=5) for _ in range(case)]for _ in range(case)]
        b = [roundto_d_significant(random.random(), sig_digits=5) for _ in range(case)]
        test_cases[case] = {'A': A, 'b': b}

    return test_cases


# In[19]:


test_suit = generate_test_data()
for each_case in test_suit.items():
    print(each_case[0])
    print(f"A: {len(each_case[1]['A'])} * {len(each_case[1]['A'][0])}")
    print(f"b: {len(each_case[1]['b'])}")
    print("-"*20)
    


# In[20]:


#Perform Guassian elimination for each of the test cas
#With pivoting
for case_key, each_case in test_suit.items():
    print(case_key)
    st_time = time.time()
    solution = guass_elimination(A = deepcopy(each_case['A']), b = deepcopy(each_case['b']), partial_pivoting=True, d=5)
    time_taken = time.time() - st_time
    each_case['with_pivoting_solution'] = solution
    each_case['with_pivoting_time_taken'] = time_taken
    print(each_case['with_pivoting_time_taken'])
    print("+"*30, "\n\n")
    


# In[21]:


#without Pivoting
time_taken = {}
for case_key, each_case in test_suit.items():
    print(case_key)
    st_time = time.time()
    solution = guass_elimination(A = deepcopy(each_case['A']), b = deepcopy(each_case['b']), partial_pivoting=False, d=5)
    time_taken = time.time() - st_time
    each_case['without_pivoting_solution'] = solution
    each_case['without_pivoting_time_taken'] = time_taken
    print(each_case['without_pivoting_time_taken'])
    print("+"*30, "\n\n")
    


# In[22]:


#No of Operations
for case_id, each_case in test_suit.items():
    print(case_id)
    print(guass_operation_count(n=case_id))
    print("="*30, "\n")


# In[23]:


efficiency = {'test_case': [], 'with_pivoting_time_taken': [], 'without_pivoting_time_taken': [], 
              'theoritical_time': []}
for k, v in test_suit.items():
    efficiency['test_case'].append(k)
    efficiency['with_pivoting_time_taken'].append(v['with_pivoting_time_taken'])
    efficiency['without_pivoting_time_taken'].append(v['without_pivoting_time_taken'])
    theoritical_time = guass_operation_count(n=k)
    efficiency['theoritical_time'].append(theoritical_time['addition'] + theoritical_time['multiplication'] + theoritical_time['division'])
    

efficiency_df = pd.DataFrame(efficiency)
efficiency_df


# In[37]:


#Plotting log(T(n)) vs long(n) for all 10 cases and fit a straight line and report the slope of the line
efficiency_df['log_n'] = efficiency_df['test_case'].map(math.log)
efficiency_df


# In[38]:


def calculate_slope(x: list, y: list):
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    
    numerator = sum([(i-mean_x) * (j-mean_y) for i,j in zip(x, y)])
    dinominator = sum([(i-mean_x)**2 for i in x])
    
    return numerator / dinominator


def calculate_y_intercept(mean_x, mean_y, m):
    return mean_y - (m * mean_x)


# In[39]:


efficiency_df['log_with_pivoting_time_taken'] = efficiency_df['with_pivoting_time_taken'].map(math.log)

efficiency_df


# In[40]:


min_scale = min(efficiency_df['log_n'].tolist() + efficiency_df['log_with_pivoting_time_taken'].tolist())
max_scale = max(efficiency_df['log_n'].tolist() + efficiency_df['log_with_pivoting_time_taken'].tolist())

efficiency_df.plot.scatter(x = 'log_n', y = 'log_with_pivoting_time_taken', 
                           grid = True,xlim = (min_scale, max_scale), ylim = (min_scale, max_scale));


# In[41]:


# Calculating slope and y-intercept
m = calculate_slope(x=efficiency_df['log_n'], y=efficiency_df['log_with_pivoting_time_taken'])
y_intercept = calculate_y_intercept(mean_x=efficiency_df['log_n'].mean(), 
                                    mean_y=efficiency_df['log_with_pivoting_time_taken'].mean(),
                                    m=m)
# Rounding off
m = roundto_d_significant(m, sig_digits=5)
y_intercept = roundto_d_significant(y_intercept, sig_digits=5)

# Fit the line and plot
efficiency_df['log_with_pivoting_fitted_y'] = efficiency_df['log_n'].map(lambda x: m * x + y_intercept)
ax = efficiency_df.plot.scatter(x = 'log_n', y = 'log_with_pivoting_time_taken', grid = True,
                                xlim = (min_scale, max_scale), ylim = (min_scale, max_scale), 
                                title = f"Slope: {m}")
efficiency_df.plot(x = 'log_n', y = 'log_with_pivoting_fitted_y', color='red' ,legend=False, ax=ax);


# In[42]:


#Without pivoting
efficiency_df['log_without_pivoting_time_taken'] = efficiency_df['without_pivoting_time_taken'].map(math.log)

efficiency_df


# In[43]:


min_scale = min(efficiency_df['log_n'].tolist() + efficiency_df['log_without_pivoting_time_taken'].tolist())
max_scale = max(efficiency_df['log_n'].tolist() + efficiency_df['log_without_pivoting_time_taken'].tolist())

efficiency_df.plot.scatter(x = 'log_n', y = 'log_without_pivoting_time_taken', grid = True,
                           xlim = (min_scale, max_scale), ylim = (min_scale, max_scale));


# In[44]:


# Calculating slope and y-intercept
m = calculate_slope(x=efficiency_df['log_n'], y=efficiency_df['log_without_pivoting_time_taken'])
y_intercept = calculate_y_intercept(mean_x=efficiency_df['log_n'].mean(), 
                                    mean_y=efficiency_df['log_without_pivoting_time_taken'].mean(),
                                    m=m)

# Rounding off
m = roundto_d_significant(m, sig_digits=5)
y_intercept = roundto_d_significant(y_intercept, sig_digits=5)

# Fit the line and plot
efficiency_df['log_without_pivoting_fitted_y'] = efficiency_df['log_n'].map(lambda x: m * x + y_intercept)

ax = efficiency_df.plot.scatter(x = 'log_n', y = 'log_without_pivoting_time_taken', grid = True,
                                xlim = (min_scale, max_scale), ylim = (min_scale, max_scale), 
                                title = f"Slope: {m}")
efficiency_df.plot(x = 'log_n', y = 'log_without_pivoting_fitted_y', color='red' ,legend=False, ax=ax);


# In[ ]:




