# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 09:28:29 2023

@author: Dell
"""

from scipy .stats import norm
nd = norm(36,6) 
z1 = 1- nd.cdf(36)
