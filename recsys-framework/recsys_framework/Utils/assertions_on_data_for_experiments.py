#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/12/2018

@author: Maurizio Ferrari Dacrema
"""
from typing import List, Optional

import numpy as np
import scipy.sparse as sp

from recsys_framework.Utils.conf_logging import get_logger

logger = get_logger(__name__)


def assert_implicit_data(urm_list: List[sp.spmatrix]) -> None:
    """
    Checks whether the URM in the list only contain implicit data in the form 1 or 0
    :param urm_list:
    :return:
    """

    for urm in urm_list:
        assert np.all(urm.data == np.ones_like(urm.data)), "assert_implicit_data: URM is not implicit as it contains data other than 1.0"

    print("Assertion assert_implicit_data: Passed")
    logger.info("Assertion assert_implicit_data: Passed")


def assert_disjoint_matrices(urm_list: List[sp.spmatrix]) -> None:
    """
    Checks whether the URM in the list have an empty intersection, therefore there is no data point contained in more than one
    URM at a time
    :param urm_list:
    :return:
    """

    urm_implicit_global: Optional[sp.spmatrix] = None

    cumulative_nnz = 0

    for urm in urm_list:

        cumulative_nnz += urm.nnz
        urm_implicit = urm.copy()
        urm_implicit.data = np.ones_like(urm_implicit.data)

        if urm_implicit_global is None:
            urm_implicit_global = urm_implicit

        else:
            urm_implicit_global += urm_implicit

    assert cumulative_nnz == urm_implicit_global.nnz, \
        "assert_disjoint_matrices: URM in list are not disjoint, {} data points are in more than one URM".format(cumulative_nnz-urm_implicit_global.nnz)


    print("Assertion assert_disjoint_matrices: Passed")
    logger.info("Assertion assert_disjoint_matrices: Passed")
