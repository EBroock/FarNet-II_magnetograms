FarNet-II is a deep learning model that improves the detection of activity on the farside of the Sun using farside phase-shift maps as an input, that is able to produce magnetograms using integers in each pixel that represent a range in magnetic field value.

The reliability of the model is tested by comparing the outputs with binary activity masks extracted nearside magnetograms 13.5 days later, once the farside activity has rotated into the nearside.

For more details, check Broock, E. G. et al. A&A, 2024 (send for publication).

This repository contains a production test for FarNet-II:

INPUTS: directory contains two inputs, each one with a batch of sequences of phase-shift maps sections, for dates outside the training set used to train the model. Dates on the name are the dates of the central element of the sequence of the first and last sequence on the file.

MASKS: contains the associated magnetograms, from Solar Orbiter and HMI, as a proxy of the reliability of the network (taken half a rotation before the farside outputs for HMI).

OUTPUTS: contains FarNet-II outputs for the given inputs (empty until execution).

Limits of magnetic levels for masks and outputs:

-4: [-inf , -108.92]
-3: [-108.92, -50.55]
-2: [-50.55, -24.75]
-1: [-24.75, -10.78]
0: [-10.78, 11.03]
1: [11.03, 25.01]
2: [25.01, 50.07]
3: [50.07, 104.84]
4: [104.84, inf]

GRAPHS: contains the image results from the execution of graphs.py and dice_vol.py, showing magnetograms from Solar Orbiter, HMI, and the outputs from FarNet-II, and the volumetric representations of all those images (empty until execution).

Farside_to_magnetograms_9levels.py: is the script that needs to run in order to produce the outputs.

Checkpoint.pth: is the file containing the parameters extracted from the training of the network.

FarNet-II_magnetograms.py: is the deep learning model of FarNet-II adapted to the production of magnetograms.

Graphs.py: is a script to display the results.