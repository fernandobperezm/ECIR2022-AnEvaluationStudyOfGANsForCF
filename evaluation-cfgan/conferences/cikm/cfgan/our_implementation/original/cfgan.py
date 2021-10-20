"""
Created on Mar. 11, 2019.
tensorflow implementation of the paper:
Dong-Kyu Chae et al. "CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks," In Proc. of ACM CIKM, 2018.
@author: Dong-Kyu Chae (kyu899@agape.hanyang.ac.kr)

usage: python model.py
environment: python3.5xx, tensorflow_gpu
IMPORTANT: make sure that (1) the user & item indices start from 0, and (2) the index should be continuous, without any empy index.

This file has been modified by "Fernando B. Pérez Maurera" to run this code "as-is" from an
outside python script. Specifically, I added constraints about the tensorflow version,
reorganized and renamed imports, and wrapped the code in a function so it is easier to call from
outside. Original code can still be seen (and run) inside the CFGAN_original folder.
"""
# cd Dropbox && cd Projects && cd CFGAN_for_open && python model.py
from typing import Tuple, List, Dict, Any

import numpy as np
import conferences.cikm.cfgan.our_implementation.original.trainer as trainer
import conferences.cikm.cfgan.our_implementation.original.data as data
import conferences.cikm.cfgan.our_implementation.original.parameters as parameters


######## prepare data ########
# IMPORTANT: make sure that (1) the user & item indices start from 0, and (2) there is NO EMPTY index (i.e., users or items having no ratings should be removed).
def run_original(benchmark: str) -> Tuple[
    Dict[str, Any],
    Dict[int, Dict[str, float]],
    np.ndarray,
    np.ndarray,
    Any,
    Any,
    Any,
    Any,
    Dict[int, List[int]],
    Dict[int, List[int]],
]:
    path = "conferences/cikm/cfgan/original_source_code/datasets"
    # benchmark = "Ciao"  # "Ciao", "ML100K", "ML1M"

    ######## load hyper-parameters ########
    hyperParams = parameters.getHyperParams(benchmark)

    mode = hyperParams["mode"]  # 'userBased', 'itemBased'

    # load purchase matrix on memory
    trainSet, userCount, itemCount = data.loadTrainingData(benchmark, path)
    testSet, GroundTruth = data.loadTestData(benchmark, path)
    userList_test = list(testSet.keys())

    # load all item (or user) purchase vectors on memory (thus, fast but consuming more memory)
    trainVector, testMaskVector, batchCount = data.to_Vectors(
        trainSet, userCount, itemCount, userList_test, mode
    )

    # since we deal with implicit feedback
    maskingVector = trainVector

    # prepare for the random negative sampling (it is also memory-inefficient)
    unobserved = []
    for batchId in range(batchCount):
        unobserved.append(list(np.where(trainVector[batchId] == 0)[0]))

    ######## train CFGAN #########
    topN = [5, 20]
    useGPU = False
    results, item_scores, recommendations = trainer.trainCFGAN(
        userCount,
        itemCount,
        batchCount,
        trainVector,
        maskingVector,
        testMaskVector,
        userList_test,
        topN,
        unobserved,
        GroundTruth,
        hyperParams,
        mode,
        useGPU,
    )

    ######## visualize learning curves (optional) #########

    #plotPath = "plots/" + benchmark + "/"
    #trainer.visualize(plotPath, precision, "precision")
    #trainer.visualize(plotPath, recall, "recall")
    #trainer.visualize(plotPath, ndcg, "ndcg")
    #trainer.visualize(plotPath, mrr, "mrr")

    # Note: Change with respect to the original implementation, we return the hyper-parameters, results dictionary,
    # the item-preferences for all users, top-N recommendations for them, the number of users, items, number of users
    # in the test set and training/testing splits.
    return (hyperParams, results, item_scores, recommendations,
            trainVector, userCount, itemCount, userList_test, trainSet, testSet)


