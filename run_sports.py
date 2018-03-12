# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 15:19:50 2018

@author: 317005
"""

from probfoil.probfoil import *

targets= ['athleteplaysinleague',
'teamplaysinleague', 
'athleteplaysforteam',
'teamalsoknownas', 
'athleteledsportsteam', 
'teamplaysagainstteam', 
'teamplayssport',
'athleteplayssport']

for fold in range(1,6):
    for target in targets:
        files = ['../rule_learning_experiment/probfoil/'+target+'/sports.settings', '../rule_learning_experiment/probfoil/'+target+'/sports_fold_'+str(fold)+'.data']
        probfoil(files=files, l=5, verbose=2, symmetry_breaking=True)
        
print('Finished.')