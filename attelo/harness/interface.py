"""
Basic interface that all test harnesses should respect
"""

from os import path as fp

from abc import (ABCMeta, abstractmethod, abstractproperty)
from joblib import Parallel
from six import with_metaclass

from .config import RuntimeConfig


class HarnessException(Exception):
    """Things that go wrong in the test harness itself
    """
    pass


class Harness(with_metaclass(ABCMeta, object)):
    """Test harness configuration

    Among other things, this is about defining conventions for
    filepaths

    Notes
    -----
    You should have a method that calls `load`.
    It should be invoked once before running the harness
    A natural idiom may be to implement a single `run` function
    that does this
    """

    # pylint: disable=too-many-arguments
    def __init__(self,
                 dataset,
                 testset):
        """
        Parameters
        ----------
        dataset: string
            Basename for training data files

        testset: string or None
            Basename for test data files.
            None if there is no test set set

        """
        self.dataset = dataset
        self.testset = testset
        self._scratch_dir = None
        self._eval_dir = None
        self._runcfg = None
        self._loaded = False

    def load(self, runcfg, eval_dir, scratch_dir):
        """
        Parameters
        ----------
        eval_dir: filepath
            Directory to store evaluation results, basically anything
            that should be considered as important for long-term
            archiving and reproducibility

        scratch_dir: filepath
            Directory for relatively emphemeral intermediary results.
            One would be more inclined to delete scratch than eval

        runcfg: RuntimeConfig or None
            Runtime configuration. None for default options

        See also
        --------
        See `attelo.harness.evaluate.prepare_dirs`
        """
        self._eval_dir = eval_dir
        self._scratch_dir = scratch_dir
        self._runcfg = runcfg or RuntimeConfig.empty()
        self._loaded = True
    # pylint: enable=too-many-arguments

    def _die_unless_loaded(self):
        """Raise an exception if the harness has not yet been loaded
        """
        if not self._loaded:
            raise HarnessException('Harness not yet loaded')

    @property
    def runcfg(self):
        """Runtime configuration settings for the harness
        """
        self._die_unless_loaded()
        return self._runcfg

    @property
    def scratch_dir(self):
        """Directory for relatively emphemeral intermediary results.

        One would be more inclined to delete scratch than eval
        """
        self._die_unless_loaded()
        return self._scratch_dir

    @property
    def eval_dir(self):
        """Directory to store evaluation results.

        Basically anything that should be considered as important for
        long-term archiving and reproducibility
        """
        self._die_unless_loaded()
        return self._eval_dir

    @abstractproperty
    def evaluations(self):
        """
        List of evaluations to use on the training data
        """
        return NotImplementedError

    @abstractproperty
    def test_evaluation(self):
        """
        The test evaluation for this harness, or None if it's unset
        """
        return None

    # pylint: disable=no-self-use
    @property
    def config_files(self):
        """Files needed to reproduce the configuration behind a
        particular set of scores.

        Will be copied into the provenance section of the report.

        Some harnesses have parameter files that should be saved
        in case there is any need to reproduce results much
        futher into the future. Specifying them here gives you some
        extra insurance in case you neglect to put them under version
        control.
        """
        return []

    @property
    def detailed_evaluations(self):
        """
        Set of evaluations for which we would like detailed reporting
        """
        return []

    @property
    def graph_docs(self):
        """
        List of document names for which we would like to generate graphs
        """
        return []
    # pylint: enable=no-self-use

    @abstractmethod
    def create_folds(self, mpack):
        """
        Generate the folds dictionary for the given multipack, optionally
        caching them to disk

        In some harness configurations, it may make sense to have a fixed
        set of folds rather than generating them on the fly

        Return
        ------
        fold_dict: dict(string, int)
            dictionary from document names to fold
        """
        return NotImplementedError

    # ------------------------------------------------------
    # paths
    # ------------------------------------------------------

    @property
    def fold_file(self):
        """Path to the fold allocation dictionary
        """
        return fp.join(self.eval_dir,
                       "folds-%s.json" % self.dataset)

    @abstractmethod
    def mpack_paths(self, test_data, stripped=False):
        """
        Return a tuple of paths needed to read a datapack

        * features
        * edu input
        * pairings
        * vocabulary

        Parameters
        ----------
        test_data: bool
            if it's test data we wanted

        stripped: bool
            return path for a "stripped" version of the data
            (faster loading, but only useful for scoring)
        """
        return NotImplementedError

    @abstractmethod
    def model_paths(self, rconf, fold):
        """Return attelo model paths in dictionary form

        Parameters
        ----------
        rconf: LearnerConfig

        fold: int

        Returns
        -------
        Dictionary from attelo parser cache keys to paths
        """
        return NotImplementedError

    def combined_dir_path(self):
        """Return path to directory where combined/global models should
        be stored

        This would be for all training data, ie. without paying attention
        to folds

        Returns
        -------
        filepath
        """
        if not self._loaded:
            raise HarnessException('Harness not yet loaded')
        return fp.join(self.scratch_dir, 'combined')

    @staticmethod
    def _fold_dir_basename(fold):
        "Relative directory for working within a given fold"
        return "fold-%d" % fold

    def fold_dir_path(self, fold):
        """Return path to working directory for a given fold

        Parameters
        ----------
        fold: int

        Returns
        -------
        filepath
        """
        return fp.join(self.scratch_dir,
                       self._fold_dir_basename(fold))

    @staticmethod
    def _decode_output_basename(econf):
        "Basename for decoder output file"
        return ".".join(["output", econf.key])

    def decode_output_path(self, econf, fold):
        """Return path to output graph for given fold and config"""
        if fold is None:
            parent_dir = self.combined_dir_path()
        else:
            parent_dir = self.fold_dir_path(fold)
        return fp.join(parent_dir,
                       self._decode_output_basename(econf))

    def _report_dir_basename(self, test_data):
        """Relative directory for a report directory
        """
        dset = self.testset if test_data else self.dataset
        return "reports-%s" % dset

    def report_dir_path(self, test_data,
                        fold=None,
                        is_tmp=True):
        """
        Path to a directory containing reports.

        Parameters
        ----------
        test_data: bool

        fold: int

        is_tmp: bool
            Only return the path to a provisional report in progress
        """
        if not is_tmp:
            parent_dir = self.eval_dir
        elif fold is None:
            parent_dir = self.scratch_dir
        else:
            parent_dir = self.fold_dir_path(fold)
        return fp.join(parent_dir,
                       self._report_dir_basename(test_data))

    # ------------------------------------------------------
    # utility
    # ------------------------------------------------------

    def parallel(self, jobs):
        """
        Run delayed jobs in parallel according to our local parameters
        """
        if self.runcfg.n_jobs == 0:
            for func, args, kwargs in jobs:
                func(*args, **kwargs)
        else:
            Parallel(n_jobs=self.runcfg.n_jobs,
                     verbose=True)(jobs)
