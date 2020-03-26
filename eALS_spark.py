from __future__ import print_function

import sys

import numpy as np
from numpy.random import rand
from numpy import matrix
np.random.seed(42)
from pyspark.sql import SparkSession
import pyspark
import pandas as pd
import json
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, vstack, hstack
from pyspark.ml.linalg import SparseVector
# from pyspark.mllib.linalg import SparceMatrix
import time


def loadSimplified(file_path, ratingFile):
    # Reading from CSV
    df_reviews_paper = pd.read_csv(file_path+ratingFile, delimiter='\t', header=None)

    # Removing duplicate ratings !
    l = list(set(zip(df_reviews_paper[0], df_reviews_paper[1])))
    # Sorting and getting two lists
    userList, itemList = list(zip(*sorted(l, key=lambda x: x[0])))

    # Generating sparse matrix
    trainMatrix = coo_matrix(([1]*len(userList), (userList, itemList)), 
                             shape=(max(df_reviews_paper[0])+1, max(df_reviews_paper[1])+1), 
                             dtype=np.int64)
    return trainMatrix


def loadYelp(file_path, ratingFile):
    '''
    Note: This function selects a few features from the review dataset, and omits the text feature because of memory concerns
    '''
    review_features = ["review_id", "user_id", "business_id", "stars", "date", "useful", "funny", "cool"]
    reviewFile = []
    print("Generating matrix R from dataset file. May take up to a minute...")
    with open(file_path+ratingFile, encoding="utf8") as infile:
        for line in infile: # For loop takes about 60 sec
            json_line = json.loads(line)
            reviewFile.append([json_line[feature] for feature in review_features])
            
    df_reviews = pd.DataFrame(reviewFile, columns=review_features)
    
    # Sorting by date
    df_reviews['date'] = pd.to_datetime(df_reviews['date'])
    df_reviews.set_index('date', inplace=True)
    df_reviews.sort_index(inplace=True)
    
    # Getting lists of unique users and businesses
    user_id_list = list(df_reviews['user_id'].value_counts().keys())
    business_id_list = list(df_reviews['business_id'].value_counts().keys())
    # review_id_list = list(df_reviews['review_id'].value_counts().keys())
    print("user_id length: ", len(user_id_list))
    print("business_id length: ", len(business_id_list))
    
    # Mapping users and businesses to numbers
    user_mapping = dict(list(enumerate(user_id_list)))
    business_mapping = dict(list(enumerate(business_id_list)))

    # Reversing the key and values for mapping
    user_mapping = {v: k for k, v in user_mapping.items()}
    business_mapping = {v: k for k, v in business_mapping.items()}

    # Mapping
    df_reviews['user_coord'] = list(df_reviews['user_id'].map(user_mapping))
    df_reviews['business_coord'] = list(df_reviews['business_id'].map(business_mapping))
    
    trainMatrix = coo_matrix(([1]*len(df_reviews['stars']), (df_reviews['user_coord'], df_reviews['business_coord'])), 
                         shape=(len(user_id_list), len(business_id_list)), dtype=np.int64)
    
    return trainMatrix


def init():
    ### Generating weight matrix p, Wi, W

    ### Set the Wi as a decay function w0 * pi ^ alpha
    sum_train = trainMatrix.getnnz()
    p = np.array(trainMatrix.sum(0))[0] # Getting vector of sums of each columns

    # Convert p[i] to probability 
    p = np.divide(p, np.array([sum_train]*itemCount))
    p = np.power(p, alpha)
    Z = sum(p)

    # Assign weight
    Wi = w0*p/Z

    # Init caches
    U = np.random.normal(init_mean, init_stdev, (userCount, factors))
    V = np.random.normal(init_mean, init_stdev, (itemCount, factors))
    
    return U, V, Wi


def rmseDense(R, U, V):
    return np.sqrt(np.sum(np.power(np.array(R - U.dot(V.T)), 2))/(U.shape[0]*U.shape[0]))


# def predict(u, i):
#     return np.inner(U[u, :], V[i, :])

def update_user(user_iterable_chunk, Wi, V, SV):
    u, itemRow, weightItemRow, URow = user_iterable_chunk
    
    itemIndexes = list(np.nonzero(itemRow.toarray())[1])
    
    if len(itemIndexes) == 0: 
        return u, URow
    
    prediction_items = [0]*itemCount
    for i in itemIndexes:
        prediction_items[i] = np.inner(URow, V[i, :]) # May improve for loop !
    
    # Getting large, dense vectors
    rating_items = list(itemRow.toarray())[0]
    w_items = list(weightItemRow.toarray())[0]
    
    for f in range(factors):
        numer = denom = 0
        for k in range(factors):
            if k != f:
                numer -= URow[k] * SV[f, k]

        # O(Nu) complexity for the positive part
        for i in itemIndexes:
            prediction_items[i] -= URow[f] * V[i, f]
            numer +=  (w_items[i]*rating_items[i] - (w_items[i]-Wi[i]) * prediction_items[i]) * V[i, f]
            denom += (w_items[i]-Wi[i]) * V[i, f] * V[i, f]
        denom += SV[f, f] + LAMBDA
        
        
        # Parameter Update
        URow[f] = numer/denom
        # Update the prediction cache
        for i in itemIndexes:
            prediction_items[i] += URow[f] * V[i, f]

    return u, csr_matrix(URow)



def update_item(item_iterable_chunk, Wi, U, SU):
    i, userRow, weightUserRow, VRow = item_iterable_chunk
    
    userIndexes = list(np.nonzero(userRow.toarray())[0])
    
    if len(userIndexes) == 0: 
        return i, VRow
    
    prediction_users = [0]*userCount
    for u in userIndexes:
        prediction_users[u] = np.inner(U[u, :], VRow) # We may try to improve the for loop !
    
    
    # Getting large, dense vectors
    rating_users = list(userRow.toarray().T)[0]
    w_users = list(weightUserRow.toarray().T)[0]
    
    for f in range(factors):
        numer = denom = 0
        for k in range(factors):
            if k != f:
                numer -= VRow[k] * SU[f, k]
        
        numer *= Wi[i]
        
        # O(Ni) complexity for the positive part
        for u in userIndexes:
            prediction_users[u] -= U[u, f] * VRow[f]
            numer +=  (w_users[u]*rating_users[u] - (w_users[u]-Wi[i]) * prediction_users[u]) * U[u, f]   
            denom += (w_users[u]-Wi[i]) * U[u, f] * U[u, f]
        denom += Wi[i] * SU[f, f] + LAMBDA
        
        # Parameter Update
        VRow[f] = numer/denom
        
        
        # Update the prediction cache
        for u in userIndexes:
            prediction_users[u] += U[u, f] * VRow[f]
        
    return i, csr_matrix(VRow)


def lossSpark(user_iterable_chunk, Wi, V, SV):
    u, itemRow, weightItemRow, URow = user_iterable_chunk
    
    itemIndexes = list(np.nonzero(itemRow.toarray())[1])

    rating_items = list(itemRow.toarray())[0]
    w_items = list(weightItemRow.toarray())[0]
    
    l = 0.0
    for i in itemIndexes:
        pred = np.inner(URow, V[i, :])
        l += w_items[i] * np.power(rating_items[i]-pred, 2)
        l -= Wi[i]* np.power(pred, 2)
        
    l += np.inner(np.dot(SV, URow), URow)
    
    return l
    


if __name__ == "__main__":
    
    ### Loading Datasets
    file_path = "data/" # Unix path
    
    # 3 datasets:
    
    # First: Pretty large, tested, works
    trainMatrix = loadSimplified(file_path, 'yelp_rating.csv')
    
    # Second: Full Yelp. Larger. untested
#     trainMatrix = loadYelp(file_path, 'review.json')

    # Third: Very simple, just for testing

#     M = 50
#     U = 200
#     F = 10
#     trainMatrix = coo_matrix(np.matrix(rand(M, F)) * np.matrix(rand(U, F).T))


    # Using same notation as author's code
    userCount = trainMatrix.shape[0] # M
    itemCount = trainMatrix.shape[1] # N
    alpha = 0.75
    w0 = 10.0
    init_mean = 0
    init_stdev = 0.1
    
    LAMBDA = 0.01   # regularization
    ITERATIONS = 10
    partitions = 5000
    factors = 20 # K
    
    W = trainMatrix.copy() # Unsure if W and train matrix should be the same
    
    # Variable and cache initialization
    U, V, Wi = init()
    
    # For efficient line and column slicing !
    trainMatrix_csr = trainMatrix.tocsr()
    trainMatrix_csc = trainMatrix.tocsc()
    W_csr = W.tocsr()
    W_csc = W.tocsc()
    
    # Slicing R matrix for later parallelization
    userRowList = [trainMatrix_csr[u] for u in range(userCount)]
    weightUserRowList = [W_csr[u] for u in range(userCount)]
    
    itemRowList = [trainMatrix_csc[:, i] for i in range(itemCount)]
    weightItemRowList = [W_csc[:, i] for i in range(itemCount)]

    
    # SPARK SESSION
    conf = pyspark.SparkConf().setAll([
                                        ('spark.executor.memory', '8g'), 
                                        ('spark.driver.maxResultSize', '8g'),
                                        ('spark.driver.memory','12g')])
    spark = SparkSession.builder.config(conf=conf).appName("Python_eALS").getOrCreate()
    sc = spark.sparkContext

    print("Running eALS with M=%d, N=%d, K=%d, iters=%d, partitions=%d\n" %
          (userCount, itemCount, factors, ITERATIONS, partitions))
    
    # Broadcast variable
    Wi_b = sc.broadcast(Wi)

    ### Building model ###
    for i in range(ITERATIONS):
        print("Iteration: ", i)

        # Update user iterable
        URowList = [U[u, :] for u in range(userCount)]
        user_iterable = zip(range(userCount), userRowList, weightUserRowList, URowList)

        # Update SV cache
        SV = np.dot(np.transpose(V), (np.array(V) * np.array(Wi.reshape((Wi.size, 1))))) # Better than the paper's nested for loops !
        
        # Update broadcasted variables
        SV_b = sc.broadcast(SV)
        V_b = sc.broadcast(V)
        
        # Spark update users task
        res = sc.parallelize(user_iterable, partitions) \
                        .map(lambda u: update_user(
                            (u[0], u[1], u[2], u[3]), 
                            Wi_b.value, 
                            V_b.value, 
                            SV_b.value)) \
                        .collect()  

        # Collect updated U
        U = vstack(np.array(res)[:, 1]).toarray()
        
        # Update item iterable
        VRowList = [V[i, :] for i in range(itemCount)]
        item_iterable = zip(range(itemCount), itemRowList, weightItemRowList, VRowList)

        # Update SU cache
        SU = np.dot(np.transpose(U), U)
        
        # Update broadcasted variables
        SU_b = sc.broadcast(SU)
        U_b = sc.broadcast(U)
        
        # Spark update items task
        res = sc.parallelize(item_iterable, partitions) \
                        .map(lambda i: update_item(
                            (i[0], i[1], i[2], i[3]),   
                            Wi_b.value, 
                            U_b.value, 
                            SU_b.value)) \
                        .collect()  

        # Collect updated V
        V = vstack(np.array(res)[:, 1]).toarray()

        
        # Spark loss calculation
        L = LAMBDA * (np.sum(U**2) + np.sum(V**2))
        LOSS = sc.parallelize(user_iterable, partitions) \
                        .map(lambda u: lossSpark(
                            (u[0], u[1], u[2], u[3]), 
                            Wi_b.value, 
                            V_b.value, 
                            SV_b.value)) \
                        .sum()

        print("LOSS: ", L+LOSS)
        
        # Dense LOSS calculation
        
        # Needs lots of RAM on driver as it is performing a dense multiplication (more than 20GB). 
        # Works on my computer but may not work on a standard machine
#         print("LOSS: ", rmseDense(trainMatrix.toarray(), U, V))
        
    np.save("U_trained", U)
    np.save("V_trained", V)
        
    spark.stop()