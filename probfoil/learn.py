from __future__ import print_function

from problog.logic import Var, Term, Constant
from itertools import product
from problog.util import Timer
import logging


class KnownError(Exception):
    pass


class LearnEntail(object):

    def __init__(self, data, language, target=None, logger=None, **kwargs):
        self._language = language

        if target is not None:
            try:
                t_func, t_arity = target.split('/')
                target = Term(t_func, int(t_arity))
            except Exception:
                raise KnownError('Invalid target specification \'%s\'' % target)

        self._target = target
        self._examples = None
        self._logger = logger

        self._data = data
        self._scores_correct = None

    @property
    def target(self):
        """The target predicate of the learning problem."""
        return self._target

    @property
    def examples(self):
        """The examples (tuple of target arguments) of the learning problem."""
        return self._examples

    @property
    def language(self):
        """Language specification of the learning problem."""
        return self._language

    def load(self, data):
        """Load the settings from a data file.

        Initializes language, target and examples.

        :param data: data file
        :type data: DataFile
        """
        self.language.load(data)  # for types and modes
        
        if self._target is None:
            try:
                target = data.query('learn', 1)[0]
                target_functor, target_arity = target[0].args
            except IndexError:
                raise KnownError('No target specification found!')
        else:
            target_functor, target_arity = self._target.functor, self._target.args[0]
        target_arguments = [Var(chr(65 + i)) for i in range(0, int(target_arity))]
        self._target = Term(str(target_functor), *target_arguments)

        # Find examples:
        #  if example_mode is closed, we will only use examples that are defined in the data
        #      this includes facts labeled with probability 0.0 (i.e. negative example)
        #  otherwise, the examples will consist of all combinations of values appearing in the data
        #      (taking into account type information)
        example_mode = data.query(Term('example_mode'), 1)
        if example_mode and str(example_mode[0][0]) == 'auto':
            types = self.language.get_argument_types(self._target.functor, self._target.arity)
            values = [self.language.get_type_values(t) for t in types]
            self._examples = list(product(*values))
        elif example_mode and str(example_mode[0][0]) == 'balance':
            # Balancing based on count only
            pos_examples = [r for r in data.query(self._target.functor, self._target.arity)]
            pos_count = len(pos_examples)
            types = self.language.get_argument_types(self._target.functor, self._target.arity)
            values = [self.language.get_type_values(t) for t in types]
            from random import shuffle
            neg_examples = list(product(*values))
            shuffle(neg_examples)
            logger = logging.getLogger(self._logger)
            logger.debug('Generated negative examples:')
            for ex in neg_examples[:pos_count]:
                logger.debug(Term(self._target(*ex).with_probability(0.0)))

            self._examples = pos_examples + neg_examples[:pos_count]
        else:
            self._examples = [r for r in data.query(self._target.functor, self._target.arity)]

        with Timer('Computing scores', logger=self._logger):
            self._scores_correct = self._compute_scores_correct()

    def _compute_scores_correct(self):
        """Computes the score for each example."""
        #result = self._data.evaluate(rule=None, functor=self.target.functor, arguments=self.examples)
        
        scores_correct = []
#        with open('log.txt', 'a+') as file:
#            file.write('_compute_scores_correct\n')
        for example in self.examples:
            #scores_correct.append(result.get(Term(self.target.functor, *example), 0.0))
            #a = result.get(Term(self.target.functor, *example), 0.0)
            #scores_correct.append(a)
            clause = [self.target.functor]
            clause.extend(example)         
            scores_correct.append(self._data.evaluateEn([clause]))
            #b = self._data.evaluateEn([clause])
#            with open('log.txt', 'a+') as file:
#                if a == b:
#                    file.write('CORRECT ' + str(a) +' = ' +str(b)+'\n')
#                else:
#                    file.write('INCORRECT ' + str(a) +' != ' +str(b)+'\n')
#                    file.write(str(example)+'\n')
        return scores_correct

    def _compute_scores_predict(self, rule):
        return self._compute_scores_predict_ground(rule)
    
    def _compute_scores_predict_test(self, rule):
        functor = 'eval_rule'

        to_eval = set(range(0, len(self.examples)))
        examples = [self.examples[i] for i in to_eval]
        # print (len(set_zero), len(set_one))

        # message = ''
        # for ex in examples:
        #     message += str(Term('query', Term(functor, *ex))) + '.\n'
        # message += '\n'.join(map(str, rule.to_clauses(functor))) + '\n'

        # Call ProbLog
        #result = self._data.evaluate(rule, functor=functor, arguments=examples)
        
        # Extract results
#        with open('log.txt', 'a+') as file:
#            file.write('_compute_scores_predict_ground\n')
#            file.write('rule='+str(rule)+'\n')
        scores_predict = [0.0] * len(self.examples)
        for i, example in zip(to_eval, examples):
            #scores_predict[i] = result.get(Term(functor, *example), 0.0)
            #b = self._data.evaluateRuleEn(rule, example)
#            with open('log.txt', 'a+') as file:
#                file.write(str(example)+'\n')
            scores_predict[i] = self._data.evaluateRuleEn(rule, example)
#            a = scores_predict[i]
#            with open('log.txt', 'a+') as file:
#                file.write(str(example)+'\n')
#                if a == b:
#                    file.write('CORRECT ' + str(a) +' = ' +str(b)+'\n')
#                else:
#                    file.write('INCORRECT ' + str(a) +' != ' +str(b)+'\n')

        return scores_predict
    
    def _compute_scores_predict_ground(self, rule):
        functor = 'eval_rule'

        # Don't evaluate examples that are guaranteed to evaluate to 0.0 or 1.0.
        set_one = []
        set_zero = []
        if rule.previous is not None:
            set_one = [i for i, s in enumerate(rule.previous.scores) if s > 1 - 1e-8]
        if rule.parent is not None:
            set_zero = [i for i, s in enumerate(rule.parent.scores) if s < 1e-8]

        to_eval = set(range(0, len(self.examples))) - set(set_one) - set(set_zero)
        examples = [self.examples[i] for i in to_eval]
        # print (len(set_zero), len(set_one))

        # message = ''
        # for ex in examples:
        #     message += str(Term('query', Term(functor, *ex))) + '.\n'
        # message += '\n'.join(map(str, rule.to_clauses(functor))) + '\n'

        # Call ProbLog
        #result = self._data.evaluate(rule, functor=functor, arguments=examples)
        
        # Extract results
#        with open('log.txt', 'a+') as file:
#            file.write('_compute_scores_predict_ground\n')
#            file.write('rule='+str(rule)+'\n')
        scores_predict = [0.0] * len(self.examples)
        for i, example in zip(to_eval, examples):
            #scores_predict[i] = result.get(Term(functor, *example), 0.0)
            #b = self._data.evaluateRuleEn(rule, example)
#            with open('log.txt', 'a+') as file:
#                file.write(str(example)+'\n')
            scores_predict[i] = self._data.evaluateRuleEn(rule, example)
#            a = scores_predict[i]
#            with open('log.txt', 'a+') as file:
#                file.write(str(example)+'\n')
#                if a == b:
#                    file.write('CORRECT ' + str(a) +' = ' +str(b)+'\n')
#                else:
#                    file.write('INCORRECT ' + str(a) +' != ' +str(b)+'\n')
        for i in set_one:
            scores_predict[i] = 1.0

        return scores_predict

    # def _compute_scores_predict_nonground(self, rule):
    #     """Evaluate the current rule using a non-ground query.
    #
    #       This is not possible because we can't properly distribute the weight of the
    #        non-ground query over the possible groundings.
    #       So this only works when all rules in the ruleset are range-restricted.
    #
    #     :param rule:
    #     :return:
    #     """
    #     functor = 'eval_rule'
    #     result = self._data.evaluate(rule, functor=functor, arguments=[self._target.args])
    #
    #     types = None
    #     values = None
    #
    #     from collections import defaultdict
    #     from problog.logic import is_variable
    #
    #     index = defaultdict(dict)
    #     for key, value in result.items():
    #         if not key.is_ground():
    #             if values is None:
    #                 types = self.language.get_argument_types(self._target.functor, self._target.arity)
    #                 values = [len(self.language.get_type_values(t)) for t in types]
    #             c = 1
    #
    #             gi = []
    #             gk = []
    #             for i, arg, vals in zip(range(0, len(key.arity)), key.args, values):
    #                 if is_variable(key.args):
    #                     c *= vals
    #                 else:
    #                     gi.append(i)
    #                     gk.append(arg)
    #             import math
    #             p = 1.0 - value ** (1.0 / c)
    #             p = 1 - math.exp(math.log(1 - value) / c)
    #             index[tuple(gi)][tuple(gk)] = [p, c]
    #
    #     scores_predict = [0.0]
    #     for i, arg in enumerate(self.examples):
    #         for gi, idx in index.items():
    #             gk = tuple(arg[j] for j in gi)
    #             res = idx.get(gk, [0.0, 0])
    #             scores_predict
    #
    #     print (rule, result)
    #     print (self.examples)


class CandidateSet(object):

    def __init__(self):
        pass

    def push(self, candidate):
        raise NotImplementedError('abstract method')

    def pop(self):
        raise NotImplementedError('abstract method')

    def __bool__(self):
        raise NotImplementedError('abstract method')


class BestCandidate(CandidateSet):

    def __init__(self, candidate=None):
        CandidateSet.__init__(self)
        self.candidate = candidate

    def push(self, candidate):
        if self.candidate is None or self.candidate.score_cmp < candidate.score_cmp:
            self.candidate = candidate

    def pop(self):
        if self.candidate is not None:
            return self.candidate
        else:
            raise IndexError('Candidate set is empty!')

    def __bool__(self):
        return not self.candidate is None


class CandidateBeam(CandidateSet):

    def __init__(self, size):
        CandidateSet.__init__(self)
        self._size = size
        self._candidates = []

    def _bottom_score(self):
        if self._candidates:
            return self._candidates[-1].score_cmp
        else:
            return -1e1000

    def _insert(self, candidate):
        for i, x in enumerate(self._candidates):
            if x.is_equivalent(candidate):
                raise ValueError('duplicate')
            elif x.score_cmp < candidate.score_cmp:
                self._candidates.insert(i, candidate)
                return False
        self._candidates.append(candidate)
        return True

    def push(self, candidate):
        """Adds a candidate to the beam.

        :param candidate: candidate to add
        :return: True if candidate was accepted, False otherwise
        """
        if len(self._candidates) < self._size or candidate.score_cmp > self._bottom_score():
            #  We should add it to the beam.
            try:
                is_last = self._insert(candidate)
                if len(self._candidates) > self._size:
                    self._candidates.pop(-1)
                    return not is_last
            except ValueError:
                return False
            return True
        return False

    def pop(self):
        return self._candidates.pop(0)

    def __bool__(self):
        return bool(self._candidates)

    def __nonzero__(self):
        return bool(self._candidates)

    def __str__(self):
        s = '==================================\n'
        for candidate in self._candidates:
            s += str(candidate) + ' ' + str(candidate.score) + '\n'
        s += '=================================='
        return s
