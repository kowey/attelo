"""
Helpers for result reporting
"""

# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

# pylint: disable=too-few-public-methods

from __future__ import print_function
from collections import (namedtuple, defaultdict)
from os import path as fp
import codecs
import glob
import itertools as itr
import shutil
import sys

from attelo.io import (load_model,
                       load_predictions)
from attelo.fold import (select_testing)
from attelo.harness.util import (makedirs, md5sum_file)
from attelo.parser.intra import (IntraInterPair)
from attelo.report import (EdgeReport,
                           LabelReport,
                           EduReport,
                           CombinedReport,
                           show_confusion_matrix,
                           show_discriminating_features)
from attelo.score import (empty_confusion_matrix,
                          build_confusion_matrix,
                          discriminating_features,
                          score_edges,
                          score_edges_by_label,
                          score_edus,
                          select_in_pack)
from attelo.table import (DataPack,
                          idxes_fakeroot,
                          idxes_inter,
                          idxes_intra)
from attelo.util import (Team)

from .util import makedirs
from .graph import (mk_graphs, mk_test_graphs)


class ReportPack(namedtuple('ReportPack',
                            ['edge',
                             'edge_by_label',
                             'edu',
                             'confusion',
                             'confusion_labels'])):
    """
    Collection of reports of various sorts

    Missing reports can be set to None (but confusion_labels
    should be set if confusion is set)

    :type edge_by_label: dict(string, CombinedReport)
    :type confusion: dict(string, array)
    """

    def _filenames(self, output_dir):
        """
        Return a dictionary of filenames; keys are internal
        to this class
        """
        makedirs(output_dir)
        filenames = {}

        if self.edge is not None:
            # edgewise scores
            filenames['edge'] = fp.join(output_dir, 'scores.txt')

        if self.edu is not None:
            # edu scores
            filenames['edu'] = fp.join(output_dir, 'edu-scores.txt')

        if self.edge_by_label is not None:
            label_dir = fp.join(output_dir, 'label-scores')
            makedirs(label_dir)
            for key in self.edge_by_label:
                fkey = ('label', key)
                filenames[fkey] = fp.join(label_dir, '-'.join(key))

        if self.confusion is not None:
            confusion_dir = fp.join(output_dir, 'confusion')
            makedirs(confusion_dir)
            for key in self.confusion:
                fkey = ('confusion', key)
                filenames[fkey] = fp.join(confusion_dir, '-'.join(key))
        return filenames

    def dump(self, output_dir, header=None):
        """
        Save reports to an output directory
        """
        makedirs(output_dir)
        fnames = self._filenames(output_dir).values()
        # touch every file
        for fname in fnames:
            with open(fname, 'w'):
                pass
        self.append(output_dir, header=header)

    def append(self, output_dir, header):
        """
        Append reports to a pre-existing output directory

        :param header: optional name for the tables
        :type header: string or None
        """
        fnames = self._filenames(output_dir)
        if self.edge is not None:
            # edgewise scores
            with open(fnames['edge'], 'a') as ostream:
                print(self.edge.table(main_header=header), file=ostream)
                print(file=ostream)

        if self.edu is not None:
            # edu scores
            with open(fnames['edu'], 'a') as ostream:
                print(self.edu.table(main_header=header), file=ostream)
                print(file=ostream)

        if self.edge_by_label is not None:
            for key, report in self.edge_by_label.items():
                fkey = ('label', key)
                with open(fnames[fkey], 'a') as ostream:
                    print(report.table(sortkey=lambda (_, v): 0 - v.count,
                                       main_header=header),
                          file=ostream)
                    print(file=ostream)

        if self.confusion is not None:
            for key, matrix in self.confusion.items():
                fkey = ('confusion', key)
                with open(fnames[fkey], 'a') as ostream:
                    print(show_confusion_matrix(self.confusion_labels, matrix,
                                                main_header=header),
                          file=ostream)
                    print(file=ostream)


class Slice(namedtuple('Slice',
                       ['fold',
                        'configuration',
                        'predictions',
                        'enable_details'])):
    '''
    A piece of the full report that we would like to build.
    See :py:func:`full_report`

    :type fold: int

    :param configuration: arbitrary strings to identify individual
                          configuration; we'd typically expect the
                          a configuration to appear over multiple
                          folds
    :type configuration: (string, string...)

    :param predictions: list of edges as you'd get in attelo decode

    :param enable_details: True if we want to enable potentially slower
                         more expensive detailed reporting

    :type enable_details: bool
    '''
    pass


# pylint: disable=too-many-locals
# it's a bit hard to write this sort score accumulation code
# local help
def full_report(mpack, fold_dict, slices,
                adjust_pack=None):
    """
    Generate a report across a set of folds and configurations.

    This is a bit tricky as the the idea is that we have to acculumate
    per-configuration results over the folds.

    Here configurations are just arbitrary strings

    :param slices: the predictions for each configuration, for each fold.
                   Folds should be contiguous for maximum efficiency.
                   It may be worthwhile to generate this lazily
    :type slices: iterable(:py:class:`Slice`)

    :param adjust_pack: (optional) function that modifies a DataPack, for
                        example by picking out a subset of the pairings.
    :type adjust_pack: (DataPack -> DataPack) or None
    """
    if not mpack:
        raise ValueError("Can't report with empty multipack")
    edge_count = defaultdict(list)
    edge_lab_count = defaultdict(lambda: defaultdict(list))
    edu_reports = defaultdict(EduReport)
    dpack0 = mpack.values()[0]
    confusion = defaultdict(lambda: empty_confusion_matrix(dpack0))

    fold = None
    is_first_slice = True

    # avoid slicing the predictions if we can help it (slow)
    adjust_pack = adjust_pack or (lambda x: x)
    adjust_predictions = select_in_pack if adjust_pack else (lambda _, x: x)

    for slc in slices:
        if is_first_slice and slc.fold is None:
            fpack = DataPack.vstack([adjust_pack(x)
                                     for x in mpack.values()])
            is_first_slice = False
        elif is_first_slice or slc.fold != fold:
            f_mpack = select_testing(mpack, fold_dict, slc.fold)
            fpack = DataPack.vstack([adjust_pack(x)
                                     for x in f_mpack.values()])
            fold = slc.fold
            is_first_slice = False
        key = slc.configuration
        # accumulate scores
        predictions = adjust_predictions(fpack, slc.predictions)
        edge_count[key].append(score_edges(fpack, predictions))
        edu_reports[key].add(score_edus(fpack, predictions))
        confusion[key] += build_confusion_matrix(fpack, predictions)
        if slc.enable_details:
            details = score_edges_by_label(fpack, predictions)
            for label, lab_count in details:
                edge_lab_count[key][label].append(lab_count)

    edge_report = CombinedReport(EdgeReport,
                                 {k: EdgeReport(v)
                                  for k, v in edge_count.items()})
    # combine
    edge_by_label_report = {}
    for key, counts in edge_lab_count.items():
        report = CombinedReport(LabelReport,
                                {(label,): LabelReport(vs)
                                 for label, vs in counts.items()})
        edge_by_label_report[key] = report

    return ReportPack(edge=edge_report,
                      edge_by_label=edge_by_label_report or None,
                      edu=CombinedReport(EduReport, edu_reports),
                      confusion=confusion,
                      confusion_labels=dpack0.labels)
# pylint: enable=too-many-locals


def _report_key(econf):
    """
    Rework an evaluation config key so it looks nice in
    our reports

    :rtype tuple(string)
    """
    if not isinstance(econf.learner, IntraInterPair):
        learner_key = econf.learner.key
    elif econf.learner.intra.key == econf.learner.inter.key:
        learner_key = econf.learner.intra.key
    else:
        learner_key = '{}S_D{}'.format(econf.learner.intra.key,
                                       econf.learner.inter.key)
    return (learner_key,
            econf.parser.key[len(econf.settings.key) + 1:],
            econf.settings.key)


def _fold_report_slices(hconf, fold):
    """
    Report slices for a given fold
    """
    print('Scoring fold {}...'.format(fold),
          file=sys.stderr)
    dkeys = [econf.key for econf in hconf.detailed_evaluations]
    for econf in hconf.evaluations:
        p_path = hconf.decode_output_path(econf, fold)
        yield Slice(fold=fold,
                    configuration=_report_key(econf),
                    predictions=load_predictions(p_path),
                    enable_details=econf.key in dkeys)


def _model_info_path(hconf, rconf, test_data, fold=None, intra=False):
    """
    Path to the model output file
    """
    template = "discr-features{grain}.{learner}.txt"
    return fp.join(hconf.report_dir_path(test_data, fold=fold),
                   template.format(grain='-sent' if intra else '',
                                   learner=rconf.key))


def _mk_model_summary(hconf, dconf, rconf, test_data, fold):
    "generate summary of best model features"
    _top_n = 3

    if not isinstance(rconf, IntraInterPair):
        rconf = IntraInterPair(intra=rconf, inter=rconf)

    def _write_discr(discr, subconf, intra):
        "write discriminating features to disk"
        if discr is None:
            print(('No discriminating features for {name} {grain} model'
                   '').format(name=subconf.key,
                              grain='sent' if intra else 'doc'),
                  file=sys.stderr)
            return
        output = _model_info_path(hconf, subconf, test_data,
                                  fold=fold,
                                  intra=intra)
        with codecs.open(output, 'wb', 'utf-8') as fout:
            print(show_discriminating_features(discr),
                  file=fout)

    dpack0 = dconf.pack.values()[0]
    labels = dpack0.labels
    vocab = dpack0.vocab
    mpaths = hconf.model_paths(rconf, fold)
    # doc level discriminating features
    models = Team(attach=mpaths['attach'],
                  label=mpaths['label']).fmap(load_model)
    discr = discriminating_features(models, labels, vocab, _top_n)
    _write_discr(discr, rconf.inter, False)

    # sentence-level
    if 'intra:attach' not in mpaths or 'intra:label' not in mpaths:
        return
    s_attach_path = mpaths['intra:attach']
    s_label_path = mpaths['intra:label']
    if not fp.exists(s_attach_path) or not fp.exists(s_label_path):
        return
    models = Team(attach=s_attach_path,
                  label=s_label_path).fmap(load_model)
    discr = discriminating_features(models, labels, vocab, _top_n)
    _write_discr(discr, rconf.intra, True)


def _mk_hashfile(hconf, dconf, test_data):
    "Hash the features and models files for long term archiving"

    hash_me = list(hconf.mpack_paths(False))
    if hconf.test_evaluation is not None:
        hash_me.extend(hconf.mpack_paths(True))
    learners = frozenset(e.learner for e in hconf.evaluations)
    for rconf in learners:
        mpaths = hconf.model_paths(rconf, None)
        hash_me.extend(mpaths.values())
    if not test_data:
        for fold in sorted(frozenset(dconf.folds.values())):
            for rconf in learners:
                mpaths = hconf.model_paths(rconf, fold)
                hash_me.extend(mpaths.values())
    provenance_dir = fp.join(hconf.report_dir_path(test_data, None),
                             'provenance')
    makedirs(provenance_dir)
    with open(fp.join(provenance_dir, 'hashes.txt'), 'w') as stream:
        for path in hash_me:
            if not fp.exists(path):
                continue
            fold_basename = fp.basename(fp.dirname(path))
            if fold_basename.startswith('fold-'):
                nice_path = fp.join(fold_basename, fp.basename(path))
            else:
                nice_path = fp.basename(path)
            print('\t'.join([nice_path, md5sum_file(path)]),
                  file=stream)


def _copy_version_files(hconf, test_data):
    "Hash the features and models files for long term archiving"
    provenance_dir = fp.join(hconf.report_dir_path(test_data, None),
                             'provenance')
    makedirs(provenance_dir)
    for vpath in glob.glob(fp.join(hconf.eval_dir,
                                   'versions-*.txt')):
        shutil.copy(vpath, provenance_dir)
    for cpath in hconf.config_files:
        shutil.copy(cpath, provenance_dir)


def _mk_report(hconf, dconf, slices, fold, test_data=False):
    """helper for report generation

    :type fold: int or None
    """
    # we could just use slices = list(slices) here but we have a
    # bit of awkward lazy IO where it says 'scoring fold N...'
    # the idea being that we should only really see this when it's
    # actually scoring that fold. Hoop-jumping induced by the fact
    # that we are now generating multiple reports on the same slices
    slices_ = itr.tee(slices, 4)
    rpack = full_report(dconf.pack, dconf.folds, slices_[0])
    rdir = hconf.report_dir_path(test_data, fold)
    rpack.dump(rdir, header='whole')

    partitions = [(1, 'intra', lambda d: d.selected(idxes_intra(d))),
                  (2, 'inter', lambda d: d.selected(idxes_inter(d))),
                  (3, 'froot', lambda d: d.selected(idxes_fakeroot(d)))]
    for i, header, adjust_pack in partitions:
        rpack = full_report(dconf.pack, dconf.folds, slices_[i],
                            adjust_pack=adjust_pack)
        rpack.append(rdir, header=header)

    for rconf in set(e.learner for e in hconf.evaluations):
        _mk_model_summary(hconf, dconf, rconf, test_data, fold)


def mk_fold_report(hconf, dconf, fold):
    "Generate reports for the given fold"
    slices = _fold_report_slices(hconf, fold)
    _mk_report(hconf, dconf, slices, fold)


def mk_global_report(hconf, dconf):
    "Generate reports for all folds"
    slices = itr.chain.from_iterable(_fold_report_slices(hconf, f)
                                     for f in frozenset(dconf.folds.values()))
    _mk_report(hconf, dconf, slices, None)
    _copy_version_files(hconf, False)

    report_dir = hconf.report_dir_path(False, fold=None)
    final_report_dir = hconf.report_dir_path(False, fold=None, is_tmp=False)
    mk_graphs(hconf, dconf)
    _mk_hashfile(hconf, dconf, False)
    if fp.exists(final_report_dir):
        shutil.rmtree(final_report_dir)
    shutil.copytree(report_dir, final_report_dir)
    # this can happen if resuming a report; better copy
    # it again
    print('Report saved in ', final_report_dir,
          file=sys.stderr)


def mk_test_report(hconf, dconf):
    "Generate reports for test data"
    econf = hconf.test_evaluation
    if econf is None:
        return

    p_path = hconf.decode_output_path(econf, None)
    slices = [Slice(fold=None,
                    configuration=_report_key(econf),
                    predictions=load_predictions(p_path),
                    enable_details=True)]
    _mk_report(hconf, dconf, slices, None,
               test_data=True)
    _copy_version_files(hconf, True)

    report_dir = hconf.report_dir_path(True, fold=None)
    final_report_dir = hconf.report_dir_path(True, fold=None, is_tmp=False)
    mk_test_graphs(hconf, dconf)
    _mk_hashfile(hconf, dconf, True)
    # this can happen if resuming a report; better copy
    # it again
    if fp.exists(final_report_dir):
        shutil.rmtree(final_report_dir)
    shutil.copytree(report_dir, final_report_dir)
    print('TEST Report saved in ', final_report_dir,
          file=sys.stderr)
