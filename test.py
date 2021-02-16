from __future__ import absolute_import

import os
from got10k.experiments import *

from siamfc import TrackerSiamFC


if __name__ == '__main__':

    siamfc_dir = "/workspace/code/IA/tracking/siamfc"
    epoch = 98
    version = "[TRAIN]toptrain"
    net_path = f"{siamfc_dir}/{version}/siamfc_alexnet_e{epoch}.pth"
    output_name = "center_brut"
    tracker = TrackerSiamFC(net_path=net_path, output_name=output_name)

    e = ExperimentOTB('./data/OTB', version=2015)
    e.run(tracker, visualize=True)
    e.report([tracker.name])
