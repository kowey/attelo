"""
Manipulating data tables (taking slices, etc)
"""

from __future__ import print_function
from collections import defaultdict, namedtuple

import numpy as np
import scipy.sparse

from .edu import FAKE_ROOT_ID
from .util import concat_l

# pylint: disable=too-few-public-methods

# pylint: disable=pointless-string-statement
UNRELATED = "UNRELATED"
"distinguished value for unrelated relation labels"

UNKNOWN = "__UNK__"
"distinguished internal value for post-labelling mode"
# pylint: enable=pointless-string-statement


class DataPackException(Exception):
    "An exception which arises when worknig with an attelo data pack"

    def __init__(self, msg):
        super(DataPackException, self).__init__(msg)


class Graph(namedtuple('Graph',
                       'prediction attach label')):
    '''
    A graph can only be interpreted in light of a datapack.

    It has predictions and attach/label weights. Predictions work like
    `DataPack.target`. The weights are useful within parsing pipelines,
    where it is sometimes useful for an intermediary parser to manipulate
    the weight vectors that a parser may calculate downstream.

    See the parser interface for more details.

    Parameters
    ----------
    prediction: array(int)
        label for each edge (each cell corresponds to edge)
    attach: array(float)
        attachment weights (each cell corresponds to an edge)
    label: 2D array(float)
        label attachment weights (edge by label)

    Notes
    -----
    Predictions are always labels; however, datapack targets may also
    be -1/0/1 when adapted to binary attachment task
    '''
    def selected(self, indices):
        '''
        Return a subset of the links indicated by the list/array
        of indices
        '''
        return Graph(prediction=self.prediction[indices],
                     attach=self.attach[indices],
                     label=self.label[indices])


    @classmethod
    def vstack(cls, graphs):
        '''
        Combine several graphs into one.
        '''
        if not graphs:
            raise ValueError('need non-empty list of graphs')
        graphs = list(graphs) # handle generater exp
        gzero = graphs[0]
        if gzero is None:
            return None
        return cls(prediction=np.concatenate([x.prediction for x in graphs]),
                   attach=np.concatenate([x.attach for x in graphs]),
                   label=np.concatenate([x.label for x in graphs]))

    def tweak(self,
              prediction=None,
              attach=None,
              label=None):
        '''
        Return a variant of the current graph with some values
        changed
        '''
        # I superstitiously believe that datapacks and graphs should
        # be immutable as much as possible, and that mutability in
        # the parsing pipeline would lead to confusion; hence this
        # and namedtuples instead of simple getting and setitng
        if prediction is None:
            prediction = self.prediction
        if attach is None:
            attach = self.attach
        if label is None:
            label = self.label
        return self.__class__(prediction=prediction,
                              attach=attach,
                              label=label)


class DataPack(namedtuple('DataPack',
                          ['edus',
                           'pairings',
                           'data',
                           'target',
                           'labels',
                           'vocab',
                           'graph'])):
    '''
    A set of data that can be said to belong together.

    A typical use of the datapack would be to group together
    data for a single document/grouping. But in cases where
    this distinction does not matter, it can also be convenient
    to combine data from multiple documents into a single pack.

    Notes
    -----
    A datapack is said to be

    * single document (the usual case) it corresponds to a single
      document or "stacked" if it is made by joining multiple
      datapacks together. Some functions may only behave correctly
      on single-document datapacks
    * weighted if the graphs tuple is set. You should never see
      weighted datapacks outside of a learner or decoder

    Parameters
    ----------
    edus (EDU)
        effectively a set of edus
    pairings ([(EDU, EDU)])
        edu pairs
    data 2D array(float)
        sparse matrix of features, each
        row corresponding to a pairing
    target 1D array (should be int, really)
        array of predictions for each pairing
    labels ([string])
        list of relation labels (NB: by convention label zero
        is always the unknown label)
    vocab ([string])
        feature names (corresponds to the feature
        indices) in data
    graph (None or Graph)
        if set, arrays representing the probabilities (or
        confidence scores) of attachment and labelling
    '''
    def __len__(self):
        return len(self.pairings)

    # pylint: disable=too-many-arguments
    @classmethod
    def load(cls, edus, pairings, data, target, labels, vocab):
        '''
        Build a data pack and run some sanity checks
        (see :py:method:sanity_check')
        (recommended if reading from disk)

        :rtype: :py:class:`DataPack`
        '''
        pack = cls(edus=edus,
                   pairings=pairings,
                   data=data,
                   target=target,
                   labels=labels,
                   vocab=vocab,
                   graph=None)
        pack.sanity_check()
        return pack
    # pylint: enable=too-many-arguments

    @classmethod
    def vstack(cls, dpacks):
        '''
        Combine several datapacks into one.

        The labels and vocabulary for all packs must be the same

        :type dpacks: [DataPack]
        '''
        if not dpacks:
            raise ValueError('need non-empty list of datapacks')
        dzero = dpacks[0]
        return DataPack(edus=concat_l(d.edus for d in dpacks),
                        pairings=concat_l(d.pairings for d in dpacks),
                        data=scipy.sparse.vstack(d.data for d in dpacks),
                        target=np.concatenate([d.target for d in dpacks]),
                        labels=dzero.labels,
                        vocab=dzero.vocab,
                        graph=Graph.vstack(d.graph for d in dpacks))

    def _check_target(self):
        '''
        sanity check target properties
        '''
        if self.labels is None:
            raise DataPackException('You did not supply any labels in the '
                                    'features file')

        if UNRELATED not in self.labels:
            raise DataPackException('The label "UNRELATED" is missing from '
                                    'the labels list ' + str(self.labels))

        oops = ('The number of labels given ({labels}) is less than '
                'the number of possible target labels ({target}) in '
                'the features file')
        max_class = len(self.labels) - 1
        max_target = int(max(self.target))
        if max_class < max_target:
            raise(DataPackException(oops.format(labels=max_class + 1,
                                                target=max_target + 1)))

    def _check_table_shape(self):
        '''
        sanity check row counts (raises DataPackException)
        '''
        num_insts = self.data.shape[0]
        num_pairings = len(self.pairings)
        num_targets = len(self.target)

        if num_insts != num_pairings:
            oops = ('The number of EDU pairs ({pairs}) does not match '
                    'the number of feature instances ({insts})')
            raise(DataPackException(oops.format(pairs=num_pairings,
                                                insts=num_insts)))

        if num_insts != num_targets:
            oops = ('The number of target elements ({tgts}) does not match '
                    'the number of feature instances ({insts})')
            raise(DataPackException(oops.format(tgts=num_targets,
                                                insts=num_insts)))

    def sanity_check(self):
        '''
        Raising :py:class:`DataPackException` if anything about
        this datapack seems wrong, for example if the number of
        rows in one table is not the same as in another
        '''
        if self.labels is not None:
            if not self.labels:
                oops = 'DataPack has no labels'
                raise DataPackException(oops)
            if self.labels[0] != UNKNOWN:
                oops = 'DataPack does not have {unk} as its first label'
                raise DataPackException(oops.format(unk=UNKNOWN))
        self._check_target()
        self._check_table_shape()

    def selected(self, indices):
        '''
        Return only the items in the specified rows
        '''
        sel_targets = np.take(self.target, indices)
        if self.labels is None:
            sel_labels = None
        else:
            sel_labels = self.labels
        sel_pairings = [self.pairings[x] for x in indices]
        sel_edus_ = set()
        for edu1, edu2 in sel_pairings:
            sel_edus_.add(edu1)
            sel_edus_.add(edu2)
        sel_edus = [e for e in self.edus if e in sel_edus_]
        sel_data = self.data[indices]
        if self.graph is None:
            graph = None
        else:
            graph = self.graph.selected(indices)
        return DataPack(edus=sel_edus,
                        pairings=sel_pairings,
                        data=sel_data,
                        target=sel_targets,
                        labels=sel_labels,
                        vocab=self.vocab,
                        graph=graph)

    def set_graph(self, graph):
        '''
        Return a copy of the datapack with weights set
        '''
        num_edges = len(self)
        num_labels = len(self.labels)
        want_shape_1d = (num_edges,)
        want_shape_2d = (num_edges, num_labels)
        if graph.prediction.shape != want_shape_1d:
            oops = ('Tried to plug a {got} predictions array into a '
                    'datapack expecting {want}'
                    '').format(got=graph.prediction.shape,
                               want=want_shape_1d)
            raise ValueError(oops)
        if graph.attach.shape != want_shape_1d:
            oops = ('Tried to plug a {got} attachment weights into a '
                    'datapack expecting {want}'
                    '').format(got=graph.attach.shape,
                               want=want_shape_1d)
            raise ValueError(oops)
        if graph.label.shape != want_shape_2d:
            oops = ('Tried to plug {got} label weights into a '
                    'datapack expecting {want}'
                    '').format(got=graph.label.shape,
                               want=want_shape_2d)
            raise ValueError(oops)
        return DataPack(edus=self.edus,
                        pairings=self.pairings,
                        data=self.data,
                        target=self.target,
                        labels=self.labels,
                        vocab=self.vocab,
                        graph=graph)

    def get_label(self, i):
        '''
        Return the class label for the given target value.

        Parameters
        ----------
        i (int, less than `len(self.labels)`)

            a target value

        See also
        --------
        `label_number`
        '''
        return get_label_string(self.labels, i)

    def label_number(self, label):
        '''
        Return the numerical label that corresponnds to the given
        string label

        Useful idiom: `unrelated = dpack.label_number(UNRELATED)`

        Parameters
        ----------
        label (string in `self.labels`)

            a label string

        See also
        --------
        `get_label`
        '''
        return self.labels.index(label)


def groupings(pairings):
    '''
    Given a list of EDU pairings, return a dictionary mapping
    grouping names to list of rows within the pairings.

    :rtype: dict(string, [int])
    '''
    res = defaultdict(list)
    for i, (edu1, edu2) in enumerate(pairings):
        grp1 = edu1.grouping
        grp2 = edu2.grouping
        if grp1 is None:
            grp = grp2
        elif grp2 is None:
            grp = grp1
        elif grp1 != grp2:
            oops = ('Grouping mismatch: {edu1} is in group {grp1}, '
                    'but {edu2} is in {grp2}')
            raise(DataPackException(oops.format(edu1=edu1,
                                                edu2=edu2,
                                                grp1=grp1,
                                                grp2=grp2)))
        else:
            grp = grp1
        res[grp].append(i)
    return res


def attached_only(dpack, target):
    '''
    Return only the instances which are labelled as
    attached (ie. this would presumably return an empty
    pack on completely unseen data)

    Returns
    -------
    dpack (DataPack)
    target (array(int))
    '''
    unrelated = dpack.label_number(UNRELATED)
    indices = np.where(target != unrelated)[0]
    dpack = dpack.selected(indices)
    target = target[indices]
    return dpack, target


def for_attachment(dpack, target):
    '''
    Adapt a datapack to the attachment task. This could involve

        * selecting some of the features (all for now, but may
          change in the future)
        * modifying the features/labels in some way
          (we binarise them to 0 vs not-0)

    Returns
    -------
    dpack (DataPack)
    target (array(int))
    '''
    unrelated = dpack.label_number(UNRELATED)
    tweak = np.vectorize(lambda x: -1 if x == unrelated else 1)
    dpack = DataPack(edus=dpack.edus,
                     pairings=dpack.pairings,
                     data=dpack.data,
                     target=tweak(dpack.target),
                     labels=[UNKNOWN, UNRELATED],
                     vocab=dpack.vocab,
                     graph=dpack.graph)
    target = tweak(target)
    return dpack, target


def for_labelling(dpack, target):
    '''
    Adapt a datapack to the relation labelling task (currently a no-op).
    This could involve

        * selecting some of the features (all for now, but may
          change in the future)
        * modifying the features/labels in some way (in practice
          no change)

    Returns
    -------
    dpack (DataPack)
    target (array(int))
    '''
    return dpack, target


def idxes_fakeroot(dpack):
    """Return datapack indices only the pairings which involve the
    fakeroot node
    """
    return [i for i, (edu1, _) in enumerate(dpack.pairings)
            if edu1.id == FAKE_ROOT_ID]


def idxes_intra(dpack, include_fake_root=False):
    """Return datapack indices for pairings which correspond to
    EDUs in the same sentence (or the fake root).
    """
    idxes = []
    for i, (edu1, edu2) in enumerate(dpack.pairings):
        if edu1.id == FAKE_ROOT_ID:
            if include_fake_root:
                idxes.append(i)
        elif (edu1.grouping == edu2.grouping and
              edu1.subgrouping == edu2.subgrouping):
            idxes.append(i)
    return idxes


def idxes_inter(dpack, include_fake_root=False):
    """Return datapack indices for pairings which correspond to
    EDUs in different sentences (or the fake root).
    """
    idxes = []
    for i, (edu1, edu2) in enumerate(dpack.pairings):
        if edu1.id == FAKE_ROOT_ID:
            if include_fake_root:
                idxes.append(i)
        elif (edu1.grouping != edu2.grouping or
              edu1.subgrouping != edu2.subgrouping):
            idxes.append(i)
    return idxes


class Multipack(dict):
    '''
    A multipack is a mapping from groupings to datapacks

    This class exists purely for documentation purposes; in
    practice, a dictionary of string to Datapack will do just
    fine
    '''
    pass


def _edu_positions(dpack):
    """Return a dictionary associating each EDU with a position
    identifier. The fake root always has position 0.

    Note that this will only work correctly on single-document
    datapacks.
    """
    position = {FAKE_ROOT_ID: 0}
    sorted_edus = sorted(dpack.edus, key=lambda x: x.span()[0])
    for i, edu in enumerate(sorted_edus):
        position[edu.id] = i
    return position


def select_window(dpack, window):
    '''Select only EDU pairs that are at most `window` EDUs apart
    from each other (adjacent EDUs would be considered `0` apart)

    Note that if the window is `None`, we simply return the
    original datapack

    Note that will only work correctly on single-document datapacks
    '''
    if window is None:
        return dpack
    position = _edu_positions(dpack)
    indices = []
    for i, (edu1, edu2) in enumerate(dpack.pairings):
        gap = abs(position[edu2.id] - position[edu1.id])
        if gap <= window:
            indices.append(i)
    return dpack.selected(indices)


def pairing_distances(dpack):
    """Return for each target value (label) in the datapack,
    the left and right maximum distances of edu pairings
    (in number of EDUs, so adjacent EDUs have distance of 0)

    Note that we assume a single-document datapack. If you
    give this a stacked datapack, you may get very large
    distances to the fake root

    :rtype dict(int, (int, int))
    """
    position = _edu_positions(dpack)
    max_l = defaultdict(int)
    max_r = defaultdict(int)
    for i, (edu1, edu2) in enumerate(dpack.pairings):
        gap = position[edu2.id] - position[edu1.id]
        lbl = dpack.target[i]
        if gap < 0:
            max_l[lbl] = max(max_l[lbl], -gap)
        else:
            max_r[lbl] = max(max_r[lbl], gap)
    keys = frozenset(max_l.keys() + max_r.keys())
    return {k: (max_l[k], max_r[k]) for k in keys}


def mpack_pairing_distances(mpack):
    """Return for each target value (label) in the multipack.
    See :py:func:`pairing_distances` for details

    :rtype dict(int, (int, int))
    """
    distances = {}
    for dpack in mpack.values():
        d_maxes = pairing_distances(dpack)
        for lbl, (dmax_l, dmax_r) in d_maxes.items():
            if lbl in distances:
                gmax_l, gmax_r = distances[lbl]
                distances[lbl] = (max(gmax_l, dmax_l),
                                  max(gmax_r, dmax_r))
            else:
                distances[lbl] = dmax_l, dmax_r
    return distances


def get_label_string(labels, i):
    '''
    Return the class label for the given target value.
    '''
    return labels[int(i)]


def locate_in_subpacks(dpack, subpacks):
    """
    Given a datapack and some of its subpacks, return a
    list of tuples identifying for each pair, its subpack
    and index in that subpack.

    If a pair is not found in the list of subpacks, we
    return None instead of tuple

    Returns
    -------
    [None or (DataPack, float)]
    """
    subpacks = list(subpacks)  # in case of iterable
    pmap = {}
    for subpack in subpacks:
        for i, pair in enumerate(subpack.pairings):
            pmap[pair] = subpack, i
    return [pmap.get(x) for x in dpack.pairings]
