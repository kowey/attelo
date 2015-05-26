'''
Scoring decoding results
'''

from collections import (defaultdict, namedtuple)

import numpy
from sklearn.metrics import confusion_matrix

from .table import (UNRELATED,
                    attached_only,
                    get_label_string)

# pylint: disable=too-few-public-methods


class CountPair(namedtuple('CountPair',
                           ['undirected',
                            'directed'])):
    """
    Number of correct edges etc, both ignoring and taking
    directionality into account
    """
    @classmethod
    def sum(cls, counts):
        """
        Count made of the total of all counts
        """
        counts = list(counts)  # accept iterable
        return cls(Count.sum(x.undirected for x in counts),
                   Count.sum(x.directed for x in counts))


class Count(namedtuple('Count',
                       ['tpos_attach',
                        'tpos_label',
                        'tpos_fpos',
                        'tpos_fneg'])):
    """
    Things we would count during the scoring process
    """
    @classmethod
    def sum(cls, counts):
        """
        Count made of the total of all counts
        """
        counts = list(counts)  # accept iterable
        return cls(sum(x.tpos_attach for x in counts),
                   sum(x.tpos_label for x in counts),
                   sum(x.tpos_fpos for x in counts),
                   sum(x.tpos_fneg for x in counts))


class EduCount(namedtuple('EduCount',
                          ['correct_attach',
                           'correct_label',
                           'total'])):
    """
    Things we would count during the scoring process
    """
    @classmethod
    def sum(cls, counts):
        """
        Count made of the total of all counts
        """
        return cls(sum(x.correct_attach for x in counts),
                   sum(x.correct_label for x in counts),
                   sum(x.total for x in counts))

    def __add__(self, other):
        return EduCount(self.correct_attach + other.correct_attach,
                        self.correct_label + other.correct_label,
                        self.total + other.total)


def select_in_pack(dpack, predictions):
    """Given some predictions, return only the ones that are
    in the given datapack.

    This could be useful for situations where you want to
    evaluate on parts of a (larger) datapack. If you have
    predictions on the whole pack and you only want the
    those which are relevant to the smaller slice, just
    pass in that smaller datapack to get the corresponding
    subset of predictions back
    """
    pairing_ids = {(e1.id, e2.id) for e1, e2 in dpack.pairings}
    return [(id1, id2, r) for id1, id2, r in predictions
            if (id1, id2) in pairing_ids]


def score_edges(dpack, predictions):
    """Count correctly predicted edges and labels
    Note that undirected label counts are undefined and
    hardcoded to 0

    :rtype: :py:class:`attelo.report.CountPair`
    """
    att_pack, _ = attached_only(dpack, dpack.target)
    dict_predicted = {(arg1, arg2): rel for arg1, arg2, rel in predictions
                      if rel != UNRELATED}

    # undirected
    u_predicted = set(tuple(sorted((arg1, arg2)))
                      for arg1, arg2 in dict_predicted)
    u_gold = set(tuple(sorted((edu1.id, edu2.id)))
                 for edu1, edu2 in att_pack.pairings)
    undirected = Count(tpos_attach=len(u_gold & u_predicted),
                       tpos_label=0,
                       tpos_fpos=len(u_predicted),
                       tpos_fneg=len(u_gold))

    # directed
    tpos_attach = 0
    tpos_label = 0
    for edu_pair, ref_label in zip(att_pack.pairings, att_pack.target):
        edu1, edu2 = edu_pair
        pred_label = dict_predicted.get((edu1.id, edu2.id))
        if pred_label is not None:
            tpos_attach += 1
            if att_pack.label_number(pred_label) == ref_label:
                tpos_label += 1
    directed = Count(tpos_attach=tpos_attach,
                     tpos_label=tpos_label,
                     tpos_fpos=len(dict_predicted.keys()),
                     tpos_fneg=len(att_pack.pairings))

    return CountPair(undirected=undirected,
                     directed=directed)


def score_edus(dpack, predictions):
    """compute the number of edus

    1. with correct attachments to their heads (ie. given edu
    e, every reference (p, e) link is present, and only such
    links are present)
    2. with correct attachments to their heads and labels
    (ie. given edu e, every reference (p, e) link is present,
    with the correct label, and only such links are present)

    This score may quite low if we are predicted a multiheaded
    graph

    :rtype: :py:class:`EduCount`
    """

    e_predictions = defaultdict(list)
    for parent, edu, rel in predictions:
        if rel == UNRELATED:
            continue
        e_predictions[edu].append((parent, dpack.label_number(rel)))

    e_reference = defaultdict(list)
    unrelated = dpack.label_number(UNRELATED)
    for edu_pair, ref_label in zip(dpack.pairings, dpack.target):
        if ref_label == unrelated:
            continue
        parent, edu = edu_pair
        e_reference[edu.id].append((parent.id, int(ref_label)))

    correct_attach = 0
    correct_label = 0
    for edu in dpack.edus:
        pred = sorted(e_predictions.get(edu.id, []))
        ref = sorted(e_reference.get(edu.id, []))
        if [x[0] for x in pred] == [x[0] for x in ref]:
            correct_attach += 1
        if pred == ref:
            correct_label += 1
    assert correct_label <= correct_attach

    return EduCount(correct_attach=correct_attach,
                    correct_label=correct_label,
                    total=len(dpack.edus))


def score_edges_by_label(dpack, predictions):
    """
    Return (as a generator) a list of pairs associating each
    label with scores for that label.

    If you are scoring mutiple folds you could loop over the
    folds, combining pre-existing scores for each label within
    the fold with its counterpart in the other folds
    """
    predictions = [(e1, e2, r) for (e1, e2, r) in predictions
                   if r != UNRELATED]

    for label in dpack.labels:
        if label == UNRELATED:
            continue
        label_num = dpack.label_number(label)
        # pylint: disable=no-member
        r_indices = numpy.where(dpack.target == label_num)[0]
        # pylint: disable=no-member
        r_dpack = dpack.selected(r_indices)
        r_predictions = [(e1, e2, r) for (e1, e2, r) in predictions
                         if r == label]
        yield label, score_edges(r_dpack, r_predictions)


def build_confusion_matrix(dpack, predictions):
    """return a confusion matrix show predictions vs desired labels
    """
    pred_target = [dpack.label_number(label) for _, _, label in predictions]
    # we want the confusion matrices to have the same shape regardless
    # of what labels happen to be used in the particular fold
    # pylint: disable=no-member
    labels = numpy.arange(0, len(dpack.labels))
    # pylint: enable=no-member
    return confusion_matrix(dpack.target, pred_target, labels)


def empty_confusion_matrix(dpack):
    """return a zero array that could be used to accumulate future
    confusion matrix results
    """
    llen = len(dpack.labels)
    # pylint: disable=no-member
    return numpy.zeros((llen, llen), dtype=numpy.int32)
    # pylint: disable=no-member


def discriminating_features(models, labels, vocab, top_n):
    """return the most discriminating features (and their weights)
    for each label in the models; or None if the model does not
    support this sort of query

    See :py:func:`attelo.report.show_discriminating_features`

    :param top_n number of features to return
    :type top_n: int

    :type models: Team(model)

    :type labels: [string]

    :param sequence of string labels, ie. one for each possible feature
    :type vocab: [string]

    :rtype: [(string, [(string, float)])] or None
    """
    def fmt(features):
        'label important features'
        return [(vocab[f], w) for f, w in features]

    rows = None
    if hasattr(models.attach, 'important_features'):
        imp = models.attach.important_features(top_n)
        if imp is not None:
            rows = rows or []
            rows.append(('(attach)', fmt(imp)))

    if hasattr(models.label, 'important_features_multi'):
        per_label = models.label.important_features_multi(top_n)
        if per_label is not None:
            rows = rows or []
            for lnum in sorted(per_label):
                label = get_label_string(labels, lnum)
                rows.append((label, fmt(per_label[lnum])))
    if hasattr(models.label, 'important_features'):
        imp = models.label.important_features(top_n)
        if imp is not None:
            rows = rows or []
            rows.append(('(label)', fmt(imp)))

    return rows
