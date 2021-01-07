from __future__ import absolute_import

import os
from got10k.experiments import *

from siamfc import TrackerSiamFC


if __name__ == '__main__':
    net_path = 'pretrained/siamfc_alexnet_e50.pth'
    tracker = TrackerSiamFC(net_path=net_path)

    e = ExperimentOTB('/data/OTB', version=2015)
    e.run(tracker)
    e.report([tracker.name])
