"""
Basic interface that all parsers should respect
"""

import numpy as np

from abc import ABCMeta, abstractmethod
from six import with_metaclass

from attelo.table import (Graph, UNKNOWN, UNRELATED)

# pylint: disable=too-few-public-methods

class Parser(with_metaclass(ABCMeta, object)):
    """
    Parsers follow the scikit fit/transform idiom. They are learned from some
    training data via the `fit()` function. Once fitted to the training data,
    they can be set loose on anything you might want to parse: the `transform`
    function will produce graphs from the EDUs.

    If the learning process is expensive, it would make sense to offer the
    ability to initialise a parser from a cached model
    """
    @staticmethod
    def multiply(dpack, attach=None, label=None):
        """
        If the datapack is weighted, multiply its existing probabilities
        by the given ones, otherwise set them

        Parameters
        ----------
        attach (array(float), optional)
            If unset will default to ones
        label (2D array(float), optional)
            If unset will default to ones

        Returns
        -------
        The modified datapack
        """
        if dpack.graph is None:
            if attach is None:
                attach = np.ones(len(dpack))
            if label is None:
                label = np.ones((len(dpack), len(dpack.labels)))
            prediction = np.empty(len(dpack))
            prediction[:] = dpack.label_number(UNKNOWN)
        else:
            gra = dpack.graph
            prediction = gra.prediction
            if attach is None:
                attach = gra.attach
            else:
                attach = np.multiply(attach, gra.attach)

            if label is None:
                label = gra.label
            else:
                label = np.multiply(label, gra.label)
        graph = Graph(prediction=prediction,
                      attach=attach,
                      label=label)
        return dpack.set_graph(graph)

    @staticmethod
    def select(dpack, idxes):
        """ Mark any pairs except the ones indicated as unrelated

        See Also
        --------
        `Parser.deselect`"""
        unwanted_idxes = np.ones_like(dpack.graph.attach, dtype=bool)
        unwanted_idxes[idxes] = False
        return Parser.deselect(dpack, unwanted_idxes)

    @staticmethod
    def deselect(dpack, idxes):
        """
        Common parsing pattern: mark all edges at the given indices
        as unrelated with attachment score of 0. This should normally
        exclude them from attachment by a decoder.

        Warning: assumes a weighted datapack

        This is often a better bet than using something like
        `DataPack.selected` because it keeps the unwanted edges in
        the datapack
        """
        if dpack.graph is None:
            raise ValueError("Need a weighted datapack")
        attach = np.copy(dpack.graph.attach)
        prediction = np.copy(dpack.graph.prediction)
        attach[idxes] = 0
        prediction[idxes] = dpack.label_number(UNRELATED)
        graph = dpack.graph.tweak(attach=attach,
                                  prediction=prediction)
        return dpack.set_graph(graph)

    @staticmethod
    def dzip(fun, dpacks, targets):
        """
        Apply a function on each datapack and the corresponding target
        block

        Parameters
        ----------
        fun ((a, b) -> (a, b))
        dpacks [a]
        targets [b]

        Returns
        -------
        [a], [b]
        """
        pairs = [fun(d, t) for d, t in zip(dpacks, targets)]
        return zip(*pairs)

    @abstractmethod
    def fit(self, dpacks, targets, cache=None):
        """
        Extract whatever models or other information from the multipack
        that is necessary to make the parser operational

        Parameters
        ----------
        dpacks: [DataPack]

        targets: [array(int)]
            A block of labels for each datapack. Each block should
            have the same length as its corresponding datapack

        cache: dict(string, object), optional
            Paths to submodels. If set, this dictionary associates
            submodel names with filenames. The submodel names are
            arbitrary strings like "attach" or "label" (check the
            documentation for the parser itself to see what
            submodels it recognises) with some sort of cache.

            This usage is necessarily loose. The parser should be
            prepared to ignore a key if it does not exist in the
            cache. The typical cache value is a filepath containing
            a pickle to load or dump; but other objects may sometimes
            be used depending on the parser (eg. other caches if it's
            a parser that somehow combines other parsers together)
        """
        raise NotImplementedError


    @abstractmethod
    def transform(self, dpack):
        """
        Refine the parse for a single document: given a document and a
        graph (for the same document), add or remove edges from the
        graph (mostly remove).

        A standalone parser should be able to start from an unweighted
        datapack (a fully connected graph with all labels equally
        liekly) and pare it down with to a much more useful graph
        with one best label per edge.

        Standalone parsers ought to also do something sensible with
        weighted datapacks (partially instantiated graphs), but in
        practice they may ignore them.

        Not all parsers may necessarily standalone. Some may only be
        designed to refine already existing parses. Or may require
        further processing.

        Parameters
        ----------
        dpack: DataPack
            the graph to refine (can be unweighted for standalone
            parsers, MUST be weighted for other parsers)

        Returns
        -------
        predictions: DataPack
            the best graph/prediction for this document

            (TODO: support n-best)
        """
        raise NotImplementedError
