# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 14:35:52 2018

@author: 317005
"""

""" Adicionado
"""
class EnPredicate(object):
    def __init__(self, name):
        self.name = name
           
    def __str__(self):
        return self.name
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        return str(self) == str(other)
        
class EnNot(object):
    def __init__(self, child):
        self.child = child
           
    def __str__(self):
        return 'not('+str(self.child)+')'
    
#    def __hash__(self):
#        return hash(self.name)
    
#    def __eq__(self, other):
#        return str(self) == str(other)

class EnAtom(object):
    def __init__(self, name):
        self.name = name
        
    def __str__(self):
        return self.name
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        return str(self) == str(other)
    
class EnVariable(object):
    def __init__(self, name):
        self.name = name
        
    def __str__(self):
        return self.name
    
    def __int__(self):
        return ord(self.name) - 65
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        return str(self) == str(other)
    
class EnTuples(object):
    def __init__(self):
        self.data = {}
        
    def add(self, predicate, atoms):
        if predicate not in self.data:
            self.data[predicate] = {}
        root = self.data[predicate]
        for i in atoms:
            if i in root:
                root = root[i]
            else:
                root[i] = {}
                root = root[i]
       
class EnStructure(object):
    def __init__(self):
        self.tuples = EnTuples()
        self.predicates = {}
        self.atoms = {}
        self.bases = {}
    
    def add_base(self, relation, args):
        if relation not in self.predicates:
            self.bases[relation] = args
            for arg in args:
                if arg not in self.atoms:
                    self.atoms[arg] = {}
        
    def add_tuple(self, relation, args):
        atoms = []
        bases = self.bases[relation]
        for i in range(len(args)):
            if args[i] not in self.atoms[bases[i]]:
                self.atoms[bases[i]][args[i]] = EnAtom(args[i])
            atoms.append(self.atoms[bases[i]][args[i]])
        if relation not in self.predicates:
            self.predicates[relation] = EnPredicate(relation)
        self.tuples.add(self.predicates[relation], atoms)
        
    def count_tuples(self, target):
        def recursive_count(root):
            if len(root) == 0:
                return 1
            else:
                s = 0
                for tupl in root:
                    s += recursive_count(root[tupl])
                return s
        tuples = self.tuples.data[target]
        return recursive_count(tuples)

    def satisfy_clause(self, clause, variables={}):
        clause_pos = 0
        atom_pos = 0
        pos = None
        clause_end = len(clause)
        new_variables = {}
        variables_order = []

        while(clause_pos < clause_end):
            if atom_pos == 0:
                predicate = clause[clause_pos][0]
                if type(predicate) == EnNot:
                    pos = self.tuples.data[predicate.child]
                else:
                    pos = self.tuples.data[predicate]
                atom_pos += 1
            atom = clause[clause_pos][atom_pos] # get arg
            if type(atom) == EnVariable and atom in variables:
                    atom = variables[atom]
            if type(atom) == EnVariable and atom not in variables: # new variable
                   if atom not in new_variables:
                       pred = clause[clause_pos][0] if type(clause[clause_pos][0]) != EnNot else clause[clause_pos][0].child
                       arg_type = self.bases[pred][atom_pos-1]
                       #list_atoms = list(self.atoms[arg_type]) if atom_pos == 1 else list(pos)
                       list_atoms = list(self.atoms[arg_type])
                       new_variables[atom] = [clause_pos, atom_pos, 0, pos, list_atoms, len(list_atoms)] # clause_pos, atom_pos, atom_list_pos, tree_position, list, list_size
                       variables_order.append(atom)
#                   print(str(atom))
#                   for key, value in new_variables.items():
#                       print(str(key) + ':\n')
#                       print(str(value[0]) + ' ' +str(value[1]) + ' ' +str(value[2]) + '='+str(value[4][value[2]])+' \n')
#                   print(str(clause_pos) + ' -- ' +str(atom_pos))
#                   for i in range(len(clause)):
#                       for j in range(len(clause[i])):
#                           print(clause[i][j])
#                   for key, value in variables.items():
#                       print(str(key) + '='+str(value))
#                   print('====================')
                   atom = EnAtom(new_variables[atom][4][new_variables[atom][2]])
            if type(atom) == EnAtom:
                if atom not in pos:
                    if type(clause[clause_pos][0]) == EnNot:
                            #results[clause_pos] = 1.0
                            clause_pos += 1
                            atom_pos = 0
                    else:
                        if len(variables_order):
                            # go back
                            pos = new_variables[variables_order[-1]][3]
                            clause_pos = new_variables[variables_order[-1]][0]
                            atom_pos = new_variables[variables_order[-1]][1]
                            new_variables[variables_order[-1]][2] += 1
                            # if reached end of list, restart
                            while new_variables[variables_order[-1]][2] == new_variables[variables_order[-1]][5]:
                                if len(variables_order) == 1:
                                    return 0.0
                                del new_variables[variables_order[-1]]
                                variables_order = variables_order[:-1]
                                pos = new_variables[variables_order[-1]][3]
                                clause_pos = new_variables[variables_order[-1]][0]
                                atom_pos = new_variables[variables_order[-1]][1]
                                new_variables[variables_order[-1]][2] += 1
                        else:
                            return 0.0
                else:
                    if atom_pos+1 == len(clause[clause_pos]):
                        if type(clause[clause_pos][0]) == EnNot:
                            if len(variables_order):
                                # go back
                                pos = new_variables[variables_order[-1]][3]
                                clause_pos = new_variables[variables_order[-1]][0]
                                atom_pos = new_variables[variables_order[-1]][1]
                                new_variables[variables_order[-1]][2] += 1
                                # if reached end of list, restart
                                while new_variables[variables_order[-1]][2] == new_variables[variables_order[-1]][5]:
                                    if len(variables_order) == 1:
                                        return 0.0
                                    del new_variables[variables_order[-1]]
                                    variables_order = variables_order[:-1]
                                    pos = new_variables[variables_order[-1]][3]
                                    clause_pos = new_variables[variables_order[-1]][0]
                                    atom_pos = new_variables[variables_order[-1]][1]
                                    new_variables[variables_order[-1]][2] += 1
                            else:
                                return 0.0
                        else:
                            #results[clause_pos] = 1.0
                            clause_pos += 1
                            atom_pos = 0
                    else:
                        atom_pos += 1
                        pos = pos[atom]
        return 1.0

    def satisfy_clause_recursive(self, clause, variables={}):
        def recursive(clause_pos, atom_pos, variables, root, values):
            if atom_pos == 0: #it is a predicate
                predicate = clause[clause_pos][0]
                if type(predicate) == EnNot:
                    return recursive(clause_pos, atom_pos+1, dict(variables), self.tuples.data[predicate.child], values)
                else:
                    return recursive(clause_pos, atom_pos+1, dict(variables), self.tuples.data[predicate], values)
            else:
                j = clause[clause_pos][atom_pos]
                if type(j) == EnVariable and j in variables:
                    j = variables[j]
                if type(j) == EnAtom:
                    if j not in root:
                        if type(clause[clause_pos][0]) == EnNot:
                            v = list(values)
                            v.append(1.0)
                            if clause_pos+1 == len(clause):
                                s = 1.0
                                for i in v:
                                    s *= i
                                return s
                            return recursive(clause_pos+1, 0, dict(variables), root, v)
                        else:
                            return 0.0
                    else:
                        if atom_pos+1 == len(clause[clause_pos]):
                            v = list(values)
                            if type(clause[clause_pos][0]) == EnNot:
                                return 0.0
                            else:
                                v.append(1.0)
                            if clause_pos+1 == len(clause):
                                s = 1.0
                                for i in v:
                                    s *= i
                                return s
                            else:
                                return recursive(clause_pos+1, 0, dict(variables), root, v)
                        else:
                            return recursive(clause_pos, atom_pos+1, dict(variables), root[j], values)
                elif type(j) == EnVariable: # new variable
                    typ = self.bases[str(clause[clause_pos][0].child if type(clause[clause_pos][0]) == EnNot else clause[clause_pos][0])][atom_pos-1]
                    found = 0.0
                    for key, new_atom in self.atoms[typ].items():
                        new_dict = dict(variables)
                        new_dict[j] = new_atom
                        found = recursive(clause_pos, atom_pos, new_dict, root, values)
                        if found == 1.0:
                            #with open('log.txt', 'a+') as file:
                            #    file.write('teste\n')
                            #    file.write(str(new_dict))
                            break
                    return found
        return recursive(0, 0, variables, 0, [])