
from problog.engine import DefaultEngine
from problog.logic import Term, Var, Constant, Not
from problog import get_evaluatable


class DataFile(object):
    """Represents a data file. This is a wrapper around a ProbLog file that offers direct
    querying and evaluation functionality.

    :param source: ProbLog logic program
    :type source: LogicProgram
    """

    def __init__(self, *sources):
        self._database = DefaultEngine().prepare(sources[0])
        self._databaseEn = EnStructure()
        for source in sources[1:]:
            for clause in source:
                self._database += clause
        for source in sources:
            for clause in source:
                predicate = str(clause.functor)
                args = [str(x) for x in clause.args]
                if predicate == 'base':
                    base = str(clause.args[0].functor)
                    types = [str(x) for x in clause.args[0].args]
                    self._databaseEn.add_base(base, types)
                elif predicate != 'target' and predicate != 'mode' and predicate != 'learn' and predicate != 'example_mode':
                    self._databaseEn.add_tuple(predicate, args)

    def query(self, functor, arity=None, arguments=None):
        """Perform a query on the data.
        Either arity or arguments have to be provided.

        :param functor: functor of the query
        :param arity: number of arguments
        :type arity: int | None
        :param arguments: arguments
        :type arguments: list[Term]
        :return: list of argument tuples
        :rtype: list[tuple[Term]]
        """
        if arguments is None:
            assert arity is not None
            arguments = [None] * arity

        query = Term(functor, *arguments)
        return self._database.engine.query(self._database, query)

    def ground(self, rule, functor=None, arguments=None):
        """Generate ground program for the given rule.

        :param rule: rule to evaluate
        :type rule: Rule
        :param functor: override rule functor
        :type functor: str
        :param arguments: query arguments (None if non-ground)
        :type arguments: list[tuple[Term]] | None
        :return: ground program
        :rtype: LogicFormula
        """
        if rule is None:
            db = self._database
            target = Term(functor)
        else:
            db = self._database.extend()
            target = None
            for clause in rule.to_clauses(functor):
                target = clause.head
                db += clause

        if arguments is not None:
            queries = [target.with_args(*args) for args in arguments]
        else:
            queries = [target]

        return self._database.engine.ground_all(db, queries=queries)

    def evaluate(self, rule, functor=None, arguments=None, ground_program=None):
        """Evaluate the given rule.

        :param rule: rule to evaluate
        :type rule: Rule
        :param functor: override rule functor
        :type functor: str
        :param arguments: query arguments (None if non-ground)
        :type arguments: list[tuple[Term]] | None
        :param ground_program: use pre-existing ground program (perform ground if None)
        :type ground_program: LogicFormula | None
        :return: dictionary of results
        :rtype: dict[Term, float]
        """
        if ground_program is None:
            ground_program = self.ground(rule, functor, arguments)

        knowledge = get_evaluatable().create_from(ground_program)
        return knowledge.evaluate()
    
    def evaluateEn(self, clause):
        n_clause = []
        for i in range(len(clause)):
            a = [EnPredicate(clause[i][0])]
            for j in range(1, len(clause[i])):
                a.append(EnAtom(str(clause[i][j])))
            n_clause.append(a)
        return self._databaseEn.satisfy_clause(n_clause)
        
    def evaluateRuleEn(self, rule, example):
        with open('log.txt', 'a+') as file:
            file.write(str(rule) + '('+str(example)+')')
        variables={}
        head = rule.get_literals()[0]
        #head_predicate = str(head.functor)
        #head_args = []
        for i in range(len(head.args)):
            if type(head.args[i]) == Var:
                v = EnVariable(str(head.args[i]))
                variables[v] = EnAtom(str(example[i]))
                #head_args.append(v)
            #elif type(head.args[i]) == Constant:
                #head_args.append(EnAtom(str(head.args[i])))
        #head_get = [EnPredicate(head_predicate)]
        #head_get.extend(head_args)
        body = rule.get_literals()[1:]
        #rule_get = [head_get]
        rule_get = []
        for clause in body:
            cl = clause.child if type(clause) == Not else clause
            clause_predicate = str(cl.functor)
            clause_args = []
            for i in range(len(cl.args)):
                if type(cl.args[i]) == Var:
                    v = EnVariable(str(cl.args[i]))
                    clause_args.append(v)
                elif type(cl.args[i]) == Constant:
                    clause_args.append(EnAtom(str(cl.args[i])))
            clause_get = [EnNot(EnPredicate(clause_predicate)) if type(clause) == Not else EnPredicate(clause_predicate)]
            #clause_get = [EnPredicate(clause_predicate)]
            clause_get.extend(clause_args)
            rule_get.append(clause_get)
        return self._databaseEn.satisfy_clause(rule_get, variables)

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
    
#    def satisfy_clause(self, clause, variables={}):
#        def recursive(clause_pos, atom_pos, variables, root):
#            if atom_pos == 0: #it is a predicate
#                predicate = clause[clause_pos][0]
#                if type(predicate) == EnNot:
#                    it = 1.0 - recursive(clause_pos, atom_pos+1, dict(variables), self.tuples.data[predicate.child])
#                else:
#                    it = recursive(clause_pos, atom_pos+1, dict(variables), self.tuples.data[predicate])
#                if clause_pos+1 == len(clause):
#                    return it
#                else:
#                    return it * recursive(clause_pos+1, 0, dict(variables), root)
#            else:
#                j = clause[clause_pos][atom_pos]
#                if type(j) == EnVariable and j in variables:
#                    j = variables[j]
#                if type(j) == EnAtom:
#                    if j not in root:
#                        return 0.0
#                    else:
#                        if atom_pos+1 == len(clause[clause_pos]):
#                            #if clause_pos+1 == len(clause):
#                            #    return 1.0
#                            #else:
#                            #    return recursive(clause_pos+1, 0, dict(variables), root)
#                            return 1.0
#                        else:
#                            return recursive(clause_pos, atom_pos+1, dict(variables), root[j])
#                elif type(j) == EnVariable: # new variable
#                    typ = self.bases[str(clause[clause_pos][0].child if type(clause[clause_pos][0]) == EnNot else clause[clause_pos][0])][atom_pos-1]
#                    found = 0.0
#                    for key, new_atom in self.atoms[typ].items():
#                        new_dict = dict(variables)
#                        new_dict[j] = new_atom
#                        found = recursive(clause_pos, atom_pos, new_dict, root)
#                        if found == 1.0:
#                            #with open('log.txt', 'a+') as file:
#                            #    file.write('teste\n')
#                            #    file.write(str(new_dict))
#                            break
#                    return found
#        return recursive(0, 0, variables, 0)

    def count_satisfy_rule(self, head, body):
        self.trues = 0
        def recursive(variables, root):
            if len(root) == 0:
                start(variables)
            else:
                for tupl in root:
                    new_variables = list(variables)
                    new_variables.append(tupl)
                    recursive(new_variables, root[tupl])
        def start(variables):
            #lista2 = [str(x) for x in variables]
            #print(lista2)
            dict_variables = {head[x]:variables[x-1] for x in range(1, len(head))}
            st = self.satisfy_clause(body, dict_variables)
            if st == True:
                self.trues += 1
        tuples = self.tuples.data[head[0]]
        recursive([], tuples)
        return self.trues