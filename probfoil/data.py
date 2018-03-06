
from problog.engine import DefaultEngine
from problog.logic import Term, Var, Constant, Not
from problog import get_evaluatable
from .satisfy import *

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
        return self._databaseEn.satisfy_clause_recursive(n_clause)

    def evaluateRuleEn(self, rule, example):
        #with open('log.txt', 'a+') as file:
        #    file.write(str(rule) + '('+str(example)+')')
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
        return self._databaseEn.satisfy_clause_recursive(rule_get, variables)