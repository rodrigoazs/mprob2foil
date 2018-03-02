"""Implementation of the Prob2FOIL algorithm.


"""

from __future__ import print_function

from problog.program import PrologFile
from .data import DataFile
from .language import TypeModeLanguage
from problog.util import init_logger
from problog.logic import Term
from .rule import FOILRule
from .learn import CandidateBeam, LearnEntail

from logging import getLogger

import time
import argparse
import sys
import random

from .score import rates, accuracy, m_estimate_relative, precision, recall, m_estimate_future_relative, significance, pvalue2chisquare


class ProbFOIL(LearnEntail):

    def __init__(self, data, beam_size=5, logger='probfoil', m=1, l=None, p=None, **kwargs):
        LearnEntail.__init__(self, data, TypeModeLanguage(**kwargs), logger=logger, **kwargs)

        self._beamsize = beam_size
        self._m_estimate = m
        if p is None:
            self._min_significance = None
        else:
            self._min_significance = pvalue2chisquare(p)

        self.load(data)   # for types and modes

        getLogger(self._logger).info('Number of examples (M): %d' % len(self._examples))
        getLogger(self._logger).info('Positive weight (P): %.4f' % sum(self._scores_correct))
        getLogger(self._logger).info('Negative weight (N): %.4f' %
                                     (len(self._scores_correct) - sum(self._scores_correct)))

        self.interrupted = False
        self._stats_evaluations = 0
        self._max_length = l

    def best_rule(self, current):
        """Find the best rule to extend the current rule set.

        :param current:
        :return:
        """

        current_rule = FOILRule(target=self.target, previous=current, correct=self._scores_correct)
        current_rule.scores = [1.0] * len(self._scores_correct)
        current_rule.score = self._compute_rule_score(current_rule)
        c_tp, c_fp, c_tn, c_fn = rates(current_rule)
        current_rule.score_cmp = (current_rule.score, c_tp)
        current_rule.processed = False
        current_rule.probation = False
        current_rule.avoid_literals = set()
        if current:
            prev_tp = rates(current)[0]
        else:
            prev_tp = 0.0

        best_rule = current_rule

        self.interrupted = False
        try:
            candidates = CandidateBeam(self._beamsize)
            candidates.push(current_rule)
            iteration = 1
            while candidates:
                next_candidates = CandidateBeam(self._beamsize)

                getLogger(self._logger).debug('Best rule so far: %s [%s]' % (best_rule, best_rule.score))
                getLogger(self._logger).debug('Candidates for iteration %s:' % iteration)
                getLogger(self._logger).debug(candidates)
                iteration += 1
                while candidates:
                    current_rule = candidates.pop()
                    current_rule_literal_avoid = set(current_rule.avoid_literals)
                    getLogger(self._logger).log(8, 'TO AVOID: %s => %s' % (current_rule, current_rule.avoid_literals))
                    c_tp, c_fp, c_tn, c_fn = rates(current_rule)
                    if self._max_length and len(current_rule) >= self._max_length:
                        pass
                    else:
                        for ref in self.language.refine(current_rule):
                            if ref in current_rule.avoid_literals:  # or ref.prototype in current_rule.avoid_literals:
                                getLogger(self._logger).log(8, 'SKIPPED literal %s for rule %s' % (ref, current_rule))
                                continue
                            rule = current_rule & ref
                            getLogger(self._logger).log(7, 'EVALUATING RULE %s' % rule)
                            rule.scores = self._compute_scores_predict(rule)
                            self._stats_evaluations += 1
                            rule.score = self._compute_rule_score(rule)
                            r_tp, r_fp, r_tn, r_fn = rates(rule)
                            rule.score_cmp = (rule.score, r_tp)
                            rule.score_future = self._compute_rule_future_score(rule)
                            rule.processed = False
                            rule.avoid_literals = current_rule_literal_avoid

                            if prev_tp > r_tp - 1e-8:       # new rule has no tp improvement over previous
                                getLogger(self._logger).log(8, '%s %s %s %s [REJECT coverage] %s' % (rule, rule.score, rates(rule), rule.score_future, prev_tp))
                                # remove this literal for all sibling rules
                                current_rule_literal_avoid.add(ref)
                                current_rule_literal_avoid.add(ref.prototype)
                            elif rule.score_future <= best_rule.score:
                                getLogger(self._logger).log(8, '%s %s %s %s [REJECT potential] %s' % (rule, rule.score, rates(rule), rule.score_future, best_rule.score))
                                # remove this literal for all sibling rules
                                current_rule_literal_avoid.add(ref)
                                current_rule_literal_avoid.add(ref.prototype)
                            elif r_fp > c_fp - 1e-8:  # and not rule.has_new_variables():
                                # no fp eliminated and no new variables
                                getLogger(self._logger).log(8, '%s %s %s %s [REJECT noimprov] %s' % (rule, rule.score, rates(rule), rule.score_future, best_rule.score))
                                # remove this literal for all sibling rules
                                # current_rule_literal_avoid.add(ref)
                                # current_rule_literal_avoid.add(ref.prototype)
                            elif r_fp > c_fp - 1e-8 and current_rule.probation:
                                getLogger(self._logger).log(8, '%s %s %s %s [REJECT probation] %s' % (rule, rule.score, rates(rule), rule.score_future, best_rule.score))
                            elif r_fp < 1e-8:
                                # This rule can not be improved by adding a literal.
                                # We reject it for future exploration,
                                #  but we do consider it for best rule.
                                getLogger(self._logger).log(9, '%s %s %s %s [REJECT* fp] %s' % (rule, rule.score, rates(rule), rule.score_future, prev_tp))
                                if rule.score_cmp > best_rule.score_cmp:
                                    getLogger(self._logger).log(9, 'BETTER RULE %s %s > %s' % (rule, rule.score_cmp, best_rule.score_cmp))
                                    best_rule = rule
                            else:
                                if r_fp > c_fp - 1e-8:
                                    rule.probation = True
                                else:
                                    rule.probation = False
                                if next_candidates.push(rule):
                                    getLogger(self._logger).log(9, '%s %s %s %s [ACCEPT]' % (rule, rule.score, rates(rule), rule.score_future))
                                else:
                                    getLogger(self._logger).log(8, '%s %s %s %s [REJECT beam]' % (rule, rule.score, rates(rule), rule.score_future))

                                if rule.score_cmp > best_rule.score_cmp:
                                    getLogger(self._logger).log(9, 'BETTER RULE %s %s > %s' % (rule, rule.score_cmp, best_rule.score_cmp))
                                    best_rule = rule
                candidates = next_candidates
        except KeyboardInterrupt:
            self.interrupted = True
            getLogger(self._logger).info('LEARNING INTERRUPTED BY USER')

        while best_rule.parent and best_rule.parent.score > best_rule.score - 1e-8:
            best_rule = best_rule.parent

        self._select_rule(best_rule)
        return best_rule

    def initial_hypothesis(self):
        initial = FOILRule(self.target, correct=self._scores_correct)
        initial = initial & Term('fail')
        initial.scores = [0.0] * len(self._scores_correct)
        initial.score = self._compute_rule_score(initial)
        initial.avoid_literals = set()
        return initial

    def learn(self):
        hypothesis = self.initial_hypothesis()
        current_score = 0.0

        while True:
            next_hypothesis = self.best_rule(hypothesis)
            new_score = accuracy(next_hypothesis)
            getLogger(self._logger).info('RULE LEARNED: %s %s' % (next_hypothesis, new_score))

            s = significance(next_hypothesis)
            if self._min_significance is not None and s < self._min_significance:
                break
            if new_score > current_score:
                hypothesis = next_hypothesis
                current_score = new_score
            else:
                break
            if self.interrupted:
                break
            if hypothesis.get_literal() and hypothesis.get_literal().functor == '_recursive':
                break   # can't extend after recursive

        return hypothesis

    def _compute_rule_score(self, rule):
        return m_estimate_relative(rule, self._m_estimate)

    def _compute_rule_future_score(self, rule):
        return m_estimate_future_relative(rule, self._m_estimate)

    def _select_rule(self, rule):
        pass

    def statistics(self):
        return [('Rule evaluations', self._stats_evaluations)]


class ProbFOIL2(ProbFOIL):

    def __init__(self, *args, **kwargs):
        ProbFOIL.__init__(self, *args, **kwargs)

    def _select_rule(self, rule):
        # set rule probability and update scores
        if hasattr(rule, 'max_x'):
            x = round(rule.max_x, 8)
        else:
            x = 1.0

        if x > 1 - 1e-8:
            rule.set_rule_probability(None)
        else:
            rule.set_rule_probability(x)
        if rule.previous is None:
            scores_previous = [0.0] * len(rule.scores)
        else:
            scores_previous = rule.previous.scores

        for i, lu in enumerate(zip(scores_previous, rule.scores)):
            l, u = lu
            s = u - l
            rule.scores[i] = l + x * s

    def _compute_rule_future_score(self, rule):
        return self._compute_rule_score(rule, future=True)

    def _compute_rule_score(self, rule, future=False):
        return self._compute_rule_score_slow(rule, future)

    def _compute_rule_score_slow(self, rule, future=False):
        if rule.previous is None:
            scores_previous = [0.0] * len(rule.scores)
        else:
            scores_previous = rule.previous.scores

        data = list(zip(rule.correct, scores_previous, rule.scores))

        max_x = 0.0
        max_score = 0.0
        max_tp = 0.0
        max_fp = 0.0

        def eval_x(x, data, future=False):
            pos = 0.0
            all = 0.0
            tp = 0.0
            fp = 0.0
            tp_p = 0.0
            fp_p = 0.0
            for p, l, u in data:
                pr = l + x * (u - l)
                tp += min(p, pr)
                fp += max(0, pr - p)
                tp_p += min(p, l)
                fp_p += max(0, l - p)
                pos += p
                all += 1

            if future:
                fp = fp_p
            m = self._m_estimate
            if pos - tp_p == 0 and all - tp_p - fp_p == 0:
                mpnp = 1
            else:
                mpnp = m * ((pos - tp_p) / (all - tp_p - fp_p))
            score = (tp - tp_p + mpnp) / (tp + fp - tp_p - fp_p + m)
            return tp, fp, score

        tp_x, fp_x, score_x = eval_x(1.0, data, future)
        if score_x > max_score:
            max_x = 1.0
            max_tp = tp_x
            max_fp = fp_x
            max_score = score_x
            if not future:
                getLogger('probfoil').log(6, '%s: x=%s (%s %s) -> %s' % (rule, 1.0, tp_x, fp_x, score_x))

        for p, l, u in data:
            if u - l < 1e-8:
                continue
            x = (p - l) / (u - l)

            if x >= 1.0 or x < 0.0:
                continue

            tp_x, fp_x, score_x = eval_x(x, data, future)
            if not future:
                getLogger('probfoil').log(6,
                                          '%s: x=%s (%s %s %s) (%s %s) -> %s' % (rule, x, p, l, u, tp_x, fp_x, score_x))
            if score_x > max_score:
                max_x = x
                max_tp = tp_x
                max_fp = fp_x
                max_score = score_x

        if not future:
            getLogger('probfoil').log(6, '%s: [BEST] x=%s (%s %s) -> %s' % (rule, max_x, max_tp, max_fp, max_score))
            rule.max_x = max_x
            rule.max_tp = max_tp
            rule.max_fp = max_fp

        if max_x < 1e-8:
            return 0.0

        return max_score

    def _compute_rule_score_fast(self, rule, future=False):
        if rule.previous is None:
            scores_previous = [0.0] * len(rule.scores)
        else:
            scores_previous = rule.previous.scores

        pos = 0.0
        all = 0.0

        tp_prev = 0.0
        fp_prev = 0.0
        fp_base = 0.0
        tp_base = 0.0
        ds_total = 0.0
        pl_total = 0.0

        if not future:
            getLogger('probfoil').log(5, '%s: %s' % (rule, list(zip(rule.correct, scores_previous, rule.scores))))

        values = []
        for p, l, u in zip(rule.correct, scores_previous, rule.scores):
            pos += p
            all += 1.0

            tp_prev += min(l, p)
            fp_prev += max(0, l - p)

            ds = u - l  # improvement on previous prediction (note: ds >= 0)
            if ds == 0:  # no improvement
                pass
            elif p < l:  # lower is overestimate
                fp_base += ds
            elif p > u:  # upper is underestimate
                tp_base += ds
            else:   # correct value still achievable
                ds_total += ds
                pl_total += p - l
                y = (p - l) / (u - l)   # for x equal to this value, prediction == correct
                values.append((y, p, l, u))

        neg = all - pos
        mpnp = self._m_estimate * (pos / all)

        def comp_m_estimate(tp, fp):
            score = (tp + mpnp) / (tp + fp + self._m_estimate)
            # print (self._m_estimate, mpnp, tp, fp, score)
            return score

        max_x = 1.0
        tp_x = pl_total + tp_base + tp_prev
        if future:
            fp_x = fp_prev + fp_base
        else:
            fp_x = ds_total - pl_total + fp_base + fp_prev
        score_x = comp_m_estimate(tp_x, fp_x)
        max_score = score_x
        max_tp = tp_x
        max_fp = fp_x

        if values:
            values = sorted(values)
            if not future:
                getLogger('probfoil').log(5, '%s: %s' % (rule, [map(lambda vv: round(vv, 3), vvv) for vvv in values]))

            tp_x, fp_x, tn_x, fn_x = 0.0, 0.0, 0.0, 0.0
            ds_running = 0.0
            pl_running = 0.0
            prev_y = None
            for y, p, l, u in values + [(None, 0.0, 0.0, 0.0)]:     # extra element forces compute at end
                if y is None or prev_y is not None and y > prev_y:
                    # There is a change in y-value.
                    x = prev_y  # set current value of x
                    tp_x = pl_running + x * (ds_total - ds_running) + x * tp_base + tp_prev
                    if future:
                        fp_x = fp_prev
                    else:
                        fp_x = x * ds_running - pl_running + x * fp_base + fp_prev

                    score_x = comp_m_estimate(tp_x, fp_x)

                    if not future:
                        getLogger('probfoil').log(6, '%s: x=%s (%s %s) -> %s' % (rule, x, tp_x, fp_x, score_x))
                    if max_score is None or score_x > max_score:
                        max_score = score_x
                        max_x = x
                        max_tp = tp_x
                        max_fp = fp_x

                        # if not future:
                        #     rts = rates(rule)
                        #     est = m_estimate(rule)
                        #     print(x, tp_x, fp_x, rts, score_x, est)
                        #     # assert abs(tp_x - rts[0]) < 1e-8
                        #     # assert abs(fp_x - rts[1]) < 1e-8
                        #     # assert abs(est - score_x) < 1e-8

                prev_y = y
                pl_running += p - l
                ds_running += u - l

            assert abs(ds_running - ds_total) < 1e-8
            assert abs(pl_running - pl_total) < 1e-8

        if not future:
            getLogger('probfoil').log(6, '%s: [BEST] x=%s (%s %s) -> %s' % (rule, max_x, tp_x, fp_x, score_x))
            rule.max_x = max_x
            rule.max_tp = max_tp
            rule.max_fp = max_fp
        return max_score


def main(argv=sys.argv[1:]):
    args = argparser().parse_args(argv)

    if args.seed:
        seed = args.seed
    else:
        seed = str(random.random())
    random.seed(seed)

    logger = 'probfoil'

    if args.log is None:
        logfile = None
    else:
        logfile = open(args.log, 'w')

    log = init_logger(verbose=args.verbose, name=logger, out=logfile)

    log.info('Random seed: %s' % seed)

    # Load input files
    data = DataFile(*(PrologFile(source) for source in args.files))

    if args.probfoil1:
        learn_class = ProbFOIL
    else:
        learn_class = ProbFOIL2

    time_start = time.time()
    learn = learn_class(data, logger=logger, **vars(args))

    hypothesis = learn.learn()
    time_total = time.time() - time_start

    print ('================ SETTINGS ================')
    for kv in vars(args).items():
        print('%20s:\t%s' % kv)

    if learn.interrupted:
        print('================ PARTIAL THEORY ================')
    else:
        print('================= FINAL THEORY =================')
    rule = hypothesis
    rules = rule.to_clauses(rule.target.functor)

    # First rule is failing rule: don't print it if there are other rules.
    if len(rules) > 1:
        for rule in rules[1:]:
            print (rule)
    else:
        print (rules[0])
    print ('==================== SCORES ====================')
    print ('            Accuracy:\t', accuracy(hypothesis))
    print ('           Precision:\t', precision(hypothesis))
    print ('              Recall:\t', recall(hypothesis))
    print ('================== STATISTICS ==================')
    for name, value in learn.statistics():
        print ('%20s:\t%s' % (name, value))
    print ('          Total time:\t%.4fs' % time_total)

    if logfile:
        logfile.close()


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    parser.add_argument('-1', '--det-rules', action='store_true', dest='probfoil1',
                        help='learn deterministic rules')
    parser.add_argument('-m', help='parameter m for m-estimate', type=float,
                        default=argparse.SUPPRESS)
    parser.add_argument('-b', '--beam-size', type=int, default=5,
                        help='size of beam for beam search')
    parser.add_argument('-p', '--significance', type=float, default=None,
                        help='rule significance threshold', dest='p')
    parser.add_argument('-l', '--length', dest='l', type=int, default=None,
                        help='maximum rule length')
    parser.add_argument('-v', action='count', dest='verbose', default=None,
                        help='increase verbosity (repeat for more)')
    parser.add_argument('--symmetry-breaking', action='store_true',
                        help='avoid symmetries in refinement operator')
    parser.add_argument('--target', '-t', type=str,
                        help='specify predicate/arity to learn (overrides settings file)')
    parser.add_argument('-s', '--seed', help='random seed', default=None)
    parser.add_argument('--log', help='write log to file', default=None)
    return parser


if __name__ == '__main__':
    main()