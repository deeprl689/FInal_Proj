#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 12:12:44 2019

@author: geoffreylucas
"""

# Original Sentence from Wikipedia:
# Georgetown University is a private research university in the Georgetown neighborhood of Washington, D.C.

import pickle

# Code for saving to pickle
# pickle.dump(sub_result, open("sub-result.pkl", "wb"))

# To build the original list again.
sub_result = pickle.load(open("sub-result.pkl", "rb"))
