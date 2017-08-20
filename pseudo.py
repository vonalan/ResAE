#!/usr/bin/env python
# coding=utf-8

import numpy as np 

threshold = 0
err = inf

while(err > threshold): 
    nn = NN([n_in, n_hidden, n_out])
    nn.fit(x_train, y_train)
    out_train = nn.predict(x_train)

    err = ((out_train - y_train) * 2).sum(axis = 1).mean()
