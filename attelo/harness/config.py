"""
Configuring the harness
"""

# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

from collections import namedtuple
from enum import Enum

from attelo.util import Team

# pylint: disable=too-few-public-methods


class Keyed(namedtuple('Keyed', 'key payload')):
    '''
    A keyed object is just any object that is attached with a
    short unique (mnemonic) identifier.

    Keys often appear in filenames so it's best to avoid
    whitespace, fancy characters, and for portability reasons,
    anything non-ASCII.
    '''
    pass


class LearnerConfig(Team):
    """
    Combination of an attachment and a relation learner variant

    :type attach: Learner
    :type label: Learner
    """
    def __new__(cls, attach, label):
        team = super(LearnerConfig, cls).__new__(cls, attach, label)
        team.key = team.attach.key
        if team.attach.key != team.label.key:
            team.key += "_" + team.label.key
        return team


class EvaluationConfig(namedtuple("EvaluationConfig",
                                  "key settings learner parser")):
    """
    Combination of learners, decoders and decoder settings
    for an attelo evaluation

    The settings can really be of type that has a 'key'
    field; but you should have a way of extracting at
    least a :py:class:`DecodingMode` from it

    Parameters
    ----------
    learner: Keyed learnercfg
        Some sort of keyed learner configuration. This is usually
        of type `LearnerConfig` but there are cases where you have
        fancier objects in place
    parser: Keyed (learnercfg -> Parser)
        A (keyed) function that builds a parser from whatever
        learner configuration you used in `learner`
    settings: Keyed (???)
    """
    @classmethod
    def simple_key(cls, learner, decoder):
        """
        generate a short unique name for a learner/decoder combo
        """
        return "%s-%s" % (learner.key, decoder.key)


class DataConfig(namedtuple("DataConfig",
                            ["pack",
                             "folds"])):
    """Data tables read during harness evaluation

    This class may be folded into HarnessConfig eventually
    """


class ClusterStage(Enum):
    """
    What stage of cluster usage we are at

    This is used when you want to distribute the evaluation
    across multiple nodes of a cluster.

    The idea is that you would run the harness in separate
    stages:

        * a single "start" stage, then
        * in parallel
          * nodes running "main" stages for some folds
          * a node running a "combined_model" stage
        * finally, a single "end" stage

    """
    start = 1
    main = 2
    combined_models = 3
    end = 4


class RuntimeConfig(namedtuple('RuntimeConfig',
                               ['mode', 'n_jobs', 'folds', 'stage'])):
    """Harness runtime options.

    These are mostly relevant to when using the harness on
    a cluster.

    Parameters
    ----------
    mode: string ('resume' or 'jumpstart') or None
        * jumpstart: copy model and fold files from a previous evaluation
        * resume: pick an evaluation up from where it left off

    folds: [int] or None
        Which folds to run the harness on.
        None to run on all folds

    n_jobs: int (-1 or natural)
        Number of parallel jobs to run (-1 for max cores).
        See joblib doc for details

    stage: ClusterStage or None
        Which evaluation stage to run
    """
    @classmethod
    def empty(cls):
        """
        Empty configuration
        """
        return cls(mode=None,
                   n_jobs=-1,
                   folds=None,
                   stage=None)
