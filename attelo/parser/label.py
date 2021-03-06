"""
Labelling
"""

from os import path as fp

import numpy as np

from .interface import (Parser)
from attelo.io import (load_model, save_model)
from attelo.table import (UNKNOWN,
                          attached_only,
                          for_labelling)


class LabelClassifierWrapper(Parser):
    """
    Parser that extracts label weights from a label classifier.
    This parser is really meant to be used in conjunction with
    other parsers downstream that make use of these weights.

    If you use it in standalone mode, it will just provide the
    standard unknown prediction everywhere

    Notes
    -----
    *Cache keys*:

    * label: label model path
    """
    def __init__(self, learner):
        """
        Parameters
        ----------
        learner: LabelClassifier
        """
        super(LabelClassifierWrapper, self).__init__()
        self._learner = learner

    def fit(self, dpacks, targets, cache=None):
        """
        Extract whatever models or other information from the multipack
        that is necessary to make the labeller operational

        Returns
        -------
        self: object
        """
        cache = cache or {}
        cache_file = cache.get('label')
        if cache_file is not None and fp.exists(cache_file):
            self._learner = load_model(cache_file)
            return self
        else:
            dpacks, targets = self.dzip(attached_only, dpacks, targets)
            dpacks, targets = self.dzip(for_labelling, dpacks, targets)
            self._learner.fit(dpacks, targets)
            if cache_file is not None:
                save_model(cache_file, self._learner)
            return self

    def transform(self, dpack):
        dpack, _ = for_labelling(dpack, dpack.target)
        return self.multiply(dpack, label=self._learner.predict_score(dpack))


class SimpleLabeller(LabelClassifierWrapper):
    """
    A simple parser that assigns the best label to any edges with
    with unknown labels.

    This can be used as a standalone parser if the underlying
    classfier predicts UNRELATED

    Notes
    -----
    *Cache keys*

    label: label model path
    """

    def __init__(self, learner):
        """
        Parameters
        ----------
        learner: LabelClassifier
        """
        super(SimpleLabeller, self).__init__(learner)

    def fit(self, dpacks, targets, cache=None):
        """
        Extract whatever models or other information from the multipack
        that is necessary to make the labeller operational

        Returns
        -------
        self: object
        """
        return super(SimpleLabeller, self).fit(dpacks, targets, cache=cache)

    def transform(self, dpack):
        dpack = super(SimpleLabeller, self).transform(dpack)
        new_best_lbls = np.argmax(dpack.graph.label, axis=1)
        unk_lbl = dpack.label_number(UNKNOWN)
        prediction_ = (new if old == unk_lbl else old
                       for old, new in
                       zip(dpack.graph.prediction, new_best_lbls))
        prediction = np.fromiter(prediction_, dtype=np.dtype(np.int16))
        graph = dpack.graph.tweak(prediction=prediction)
        return dpack.set_graph(graph)
