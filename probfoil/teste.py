# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 15:54:43 2018

@author: 317005
"""
g = EnStructure()

g.add_base('parent', ['person','person'])
g.add_base('grandmother', ['person','person'])
g.add_base('male',['person'])

g.add_tuple('parent',['bart','stijn'])
g.add_tuple('parent',['bart','pieter'])
g.add_tuple('parent',['luc','soetkin'])
g.add_tuple('parent',['willem','lieve'])
g.add_tuple('parent',['willem','katleen'])
g.add_tuple('parent',['rene','willem'])
g.add_tuple('parent',['rene','lucy'])
g.add_tuple('parent',['leon','rose'])
g.add_tuple('parent',['etienne','luc'])
g.add_tuple('parent',['etienne','an'])
g.add_tuple('parent',['prudent','esther'])

g.add_tuple('parent',['katleen','stijn'])
g.add_tuple('parent',['katleen','pieter'])
g.add_tuple('parent',['lieve','soetkin'])
g.add_tuple('parent',['esther','lieve'])
g.add_tuple('parent',['esther','katleen'])
g.add_tuple('parent',['yvonne','willem'])
g.add_tuple('parent',['yvonne','lucy'])
g.add_tuple('parent',['alice','rose'])
g.add_tuple('parent',['rose','luc'])
g.add_tuple('parent',['rose','an'])
g.add_tuple('parent',['laura','esther'])

g.add_tuple('male',['bart'])
g.add_tuple('male',['etienne'])
g.add_tuple('male',['leon'])
g.add_tuple('male',['luc'])
g.add_tuple('male',['pieter'])
g.add_tuple('male',['prudent'])
g.add_tuple('male',['rene'])
g.add_tuple('male',['stijn'])
g.add_tuple('male',['willem'])

g.add_tuple('grandmother',['esther','soetkin'])
g.add_tuple('grandmother',['esther','stijn'])
g.add_tuple('grandmother',['esther','pieter'])
g.add_tuple('grandmother',['yvonne','lieve'])
g.add_tuple('grandmother',['yvonne','katleen'])
g.add_tuple('grandmother',['alice','luc'])
g.add_tuple('grandmother',['alice','an'])
g.add_tuple('grandmother',['rose','soetkin'])
g.add_tuple('grandmother',['laura','lieve'])
g.add_tuple('grandmother',['laura','katleen'])


rule = [[EnPredicate('parent'), EnVariable('C'), EnVariable('B')], [EnPredicate('parent'), EnVariable('D'), EnVariable('A')]]
variables = {EnVariable('A'): EnAtom('rene'), EnVariable('B'): EnAtom('rose')}

g.satisfy_clause(rule, variables)