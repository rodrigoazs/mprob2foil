# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 18:30:31 2018

@author: 317005
"""
from probfoil.probfoil import *

#probfoil(files=['../rule_learning_experiment/probfoil/teamalsoknownas/sports.settings', '../rule_learning_experiment/probfoil/teamalsoknownas/sports_fold_1.data'], l=3, verbose=2)

#probfoil(settings=settings, train=train, test=test, l=3, verbose=1)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 22:47:12 2017

@author: rodrigo
"""

# Importing the libraries
import pandas as pd
import random
import re
import time

def create_folds(data, size):
    length = int(len(data)/size) #length of each fold
    folds = []
    for i in range((size-1)):
        folds += [data[i*length:(i+1)*length]]
    folds += [data[(size-1)*length:len(data)]]
    return folds


def get_data(dataset, n_folds, target, target_parity, allow_negation=True, example_mode='balance'):
    #relations_accepted = ['athleteplayssport','teamalsoknownas','athleteplaysforteam','teamplayssport']
    settings = ''
    base = ''
    folds = [''] * n_folds
    
    relations = {}
    for data in dataset.values:
        entity_type = (data[1].split(':'))[1]
        entity = (data[1].split(':'))[2]
        probability = data[3]
        relation = (data[4].split(':'))[1]
        value_type = (data[5].split(':'))[1]
        value = (data[5].split(':'))[2]
           
        entity = entity.lower() #.replace('_', '')
        value = value.lower() #.replace('_', '')
        #re.sub('[^a-zA-Z]', '', title[j])
        entity = re.sub('[^a-z_]', '', entity)
        value = re.sub('[^a-z_]', '', value)
        
        #entity and value cannot start with '_', otherwise it is considered variable (?)
        entity = entity[1:] if entity[0] == '_' else entity
        value = value[1:] if value[0] == '_' else value
                  
        if relation in relations:
            relations[relation].append([entity, relation, value, probability, entity_type, value_type])
        else:
            relations[relation] = [[entity, relation, value, probability, entity_type, value_type]]

    for relation in relations:
        if relation != target:
            settings += 'mode('+str(relation)+'(+,+)).\n'
            settings += 'mode('+str(relation)+'(+,-)).\n'
            settings += 'mode('+str(relation)+'(-,+)).\n'
    settings += '\n'
    for key, value in relations.items():
        first = value[0]
        settings += 'base('+str(first[1])+'('+str(first[4])+','+str(first[5])+')).\n'
    settings += '\n'
    if allow_negation == False:
        settings += 'option(negation,off).\n'
    settings += '\n'
    settings += 'learn('+target+'/'+str(target_parity)+').\n'
    settings += '\n'
    settings += 'example_mode('+ example_mode +').\n'
    settings += '\n'
    
    for key, value in relations.items():
        for d in value:
            if d[1] != target:
                base += str(d[3])[:6]+'::' +str(d[1]) + '(' +str(d[0])+ ','+str(d[2])+ ').\n'

    tar = relations[target]
    random.shuffle(tar)
    tar = create_folds(tar, n_folds)         
    
    for i in range(n_folds):
        tuples = {}
        all_objects = set()
        for d in tar[i]:
            s = str(d[0])
            o = str(d[2])
            if s not in tuples:
                tuples[s] = set([o])
            else:
                tuples[s].add(o)
            all_objects.add(o)

        # print positive and negative targets
        for d in tar[i]:
            folds[i] += str(d[3])[:6]+'::' +str(d[1]) + '(' +str(d[0])+ ','+str(d[2])+ ').\n'
            #rand_objects = all_objects.difference(tuples[str(d[0])])
            #folds[i] += '0.0::'+str(d[1]) + '(' +str(d[0])+ ','+str(random.choice(list(rand_objects)))+ ').\n'
            
    return {'settings': settings, 'base': base, 'folds': folds}

dataset = pd.read_csv('../rule_learning_experiment/NELL.sports.08m.850.small.csv')
n_folds = 3
# ============================================

targets = [
('athleteplaysinleague', 2),
('teamplaysinleague', 2),
('athleteplaysforteam', 2),
('teamalsoknownas', 2),
('athleteledsportsteam', 2),
('teamplaysagainstteam', 2),
('teamplayssport', 2),
('athleteplayssport', 2)
]

data = get_data(dataset, 3, 'teamalsoknownas', 2)

time_start = time.time()
# cross-validation
for i in range(n_folds):
    settings = data['settings']
    train = data['base']
    test = data['base']
    for j in range(n_folds):
        if j == i:
            test += data['folds'][j]
        else:
            train += data['folds'][j]
    probfoil(settings=settings, train=train, test=test, l=5, symmetry_braking=False, verbose=1, allow_negation=True)
time_total = time.time() - time_start
print ('\nCross-validation finished')
print ('Total time:\t%.4fs' % time_total)            
                      