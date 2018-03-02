# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 15:54:43 2018

@author: 317005
"""

def satisfy_clause(self, clause, variables={}):
    clause_pos = 0
    atom_pos = 0
    variables = {}
    pos = None
    results = [1.0] * len(clause)
    clause_end = len(clause)

    while(clause_pos < clause_end):
        if atom_pos == 0:
            predicate = clause[clause_pos][0]
            if type(predicate) == EnNot:
                pos = self.tuples.data[predicate.child]
            else:
                pos = self.tuples.data[predicate]
        else:
            atom = clause[clause_pos][atom_pos] # obtem argumento
            if type(j) == EnVariable and atom in variables:
                atom = variables[atom]
            if type(j) == EnAtom:
                if atom not in pos:
                    if type(clause[clause_pos][0]) == EnNot:
                            results[clause_pos] = 1.0
                            clause_pos += 1
                            atom_pos = 0
                    else:
                        return 0.0
                else:
                    if atom_pos+1 == len(clause[clause_pos]):
                        if type(clause[clause_pos][0]) == EnNot:
                            return 0.0
                        else:
                            results[clause_pos] = 1.0
                            clause_pos += 1
                            atom_pos = 0
                    else:
                        atom_pos += 1
                        pos = pos[atom]
            if type(j) == EnVariable: # nova variavel
                   typ = self.bases[str(clause[clause_pos][0].child if type(clause[clause_pos][0]) == EnNot else clause[clause_pos][0])][atom_pos-1]
                   list_atoms = list(self.atoms[typ])
    return results