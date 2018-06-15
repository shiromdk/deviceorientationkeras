# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
    
dataset = pd.read_csv("jsonformatter.csv")
X = dataset.iloc[:, 1:4].values
y = dataset.iloc[:, 0].values

