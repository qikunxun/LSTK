import numpy as np
import os
import sys
import pickle
import argparse
import time
from collections import Counter, defaultdict


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds', default="", type=str)
    parser.add_argument('--truths', default=None, type=str)
    parser.add_argument('--top_k', default=10, type=int)
    parser.add_argument('--raw', default=False, action="store_true")
    parser.add_argument('--v', default=False, action="store_true")
    option = parser.parse_args()
    print(option)    
    start = time.time()

    if not option.raw:
        truths = pickle.load(open(option.truths, "rb"))
        query_heads, query_tails = truths.values()
    
    hits_1 = 0
    hits_3 = 0
    hits_10 = 0
    hits_by_q = defaultdict(list)
    ranks = 0
    ranks_by_q = defaultdict(list)
    rranks = 0.
    line_cnt = 0

    # lines = [l.strip().split(",") for l in open(option.preds).readlines()]
    # line_cnt = len(lines)

    with open(option.preds) as fd:
        for l in fd:
            l = l.strip().split('\t')
            assert(len(l) > 3)
            # print(l)
            q, h, t = l[0:3]
            index = l.index('#')
            this_preds = l[3:]
            this_preds_g = l[3:index]
            this_preds_e = l[index + 1:]
            assert ('#' not in this_preds_g)
            assert ('#' not in this_preds_e)
            assert(h == this_preds[-1])
            hitted = 0.
            line_cnt += 1
            if not option.raw:
                if q.startswith("inv_"):
                    q_ = q[len("inv_"):]
                    also_correct = query_heads[(q_, t)]
                else:
                    also_correct = query_tails[(q, t)]
                also_correct = set(also_correct)
                assert(h in also_correct)
                #this_preds_filtered = [j for j in this_preds[:-1] if not j in also_correct] + this_preds[-1:]
                this_preds_filtered_g = set(this_preds_g) - also_correct
                this_preds_filtered_e = set(this_preds_e[:-1]) - also_correct
                this_preds_filtered_e.add(this_preds[-1])
                # if len(this_preds_filtered) <= option.top_k:
                #     hitted = 1.
                m = len(this_preds_filtered_g)
                n = len(this_preds_filtered_e)
                rank = m + (n + 1) / 2
            else:
                # if len(this_preds) <= option.top_k:
                #     hitted = 1.
                m = len(this_preds_g)
                n = len(this_preds_e)
                rank = m + (n + 1) / 2
            # print(m, n, rank)
            hits_1 += 1 if rank == 1 else 0
            hits_3 += 1 if rank <= 3 else 0
            hits_10 += 1 if rank <= 10 else 0
            ranks += rank
            rranks += 1. / rank
            hits_by_q[q].append(hitted)
            ranks_by_q[q].append(rank)

    print("Hits at %d is %0.4f" % (1, hits_1 / line_cnt))
    print("Hits at %d is %0.4f" % (3, hits_3 / line_cnt))
    print("Hits at %d is %0.4f" % (10, hits_10 / line_cnt))
    print("Mean rank %0.2f" % (1. * ranks / line_cnt))
    print("Mean Reciprocal Rank %0.4f" % (1. * rranks / line_cnt))

    if option.v:
        hits_by_q_mean = sorted([[k, np.mean(v), len(v)] for k, v in hits_by_q.items()], key=lambda xs: xs[1], reverse=True)
        for xs in hits_by_q_mean:
          xs += [np.mean(ranks_by_q[xs[0]]), np.std(ranks_by_q[xs[0]])]
          print(", ".join([str(x) for x in xs]))

    print("Time %0.3f mins" % ((time.time() - start) / 60.))
    print("="*36 + "Finish" + "="*36)
    
if __name__ == "__main__":
    evaluate()

