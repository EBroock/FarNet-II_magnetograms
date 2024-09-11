{\rtf1\ansi\ansicpg1252\cocoartf2761
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx560\tx1121\tx1681\tx2242\tx2803\tx3363\tx3924\tx4485\tx5045\tx5606\tx6166\tx6727\tx7288\tx7848\tx8409\tx8970\tx9530\tx10091\tx10651\tx11212\tx11773\tx12333\tx12894\tx13455\tx14015\tx14576\tx15136\tx15697\tx16258\tx16818\tx17379\tx17940\tx18500\tx19061\tx19621\tx20182\tx20743\tx21303\tx21864\tx22425\tx22985\tx23546\tx24106\tx24667\tx25228\tx25788\tx26349\tx26910\tx27470\tx28031\tx28591\tx29152\tx29713\tx30273\tx30834\tx31395\tx31955\tx32516\tx33076\tx33637\tx34198\tx34758\tx35319\tx35880\tx36440\tx37001\tx37561\tx38122\tx38683\tx39243\tx39804\tx40365\tx40925\tx41486\tx42046\tx42607\tx43168\tx43728\tx44289\tx44850\tx45410\tx45971\tx46531\tx47092\tx47653\tx48213\tx48774\tx49335\tx49895\tx50456\tx51016\tx51577\tx52138\tx52698\tx53259\tx53820\tx54380\tx54941\tx55501\tx56062\slleading20\pardirnatural\partightenfactor0

\f0\fs24 \cf2 FarNet-II usage example for magnetograms production.\
\
FarNet-II is a deep learning model that improves the detection of activity on the farside of the Sun using farside phase-shift maps as an input, that is able to produce magnetograms using integers in each pixel that represent a range in magnetic field value.\
\
The reliability of the model is tested by comparing the outputs with binary activity masks extracted nearside magnetograms 13.5 days later, once the farside activity has rotated into the nearside.\
\
For more details, check Broock, E. G. et al. A&A, 2024 (send for publication).\
\
This repository contains a production test for FarNet-II:\
\
\'b7 \'91inputs\'92: directory contains two inputs, each one with a batch of sequences of phase-shift maps sections, for dates outside the training set used to train the model. Dates on the name are the dates of the central element of the sequence of the first and last sequence on the file.\
\
\'b7 \'91masks\'92: contains the associated magnetograms, from Solar Orbiter and HMI, as a proxy of the reliability of the network (taken half a rotation before the farside outputs for HMI).\
\
\'b7 \'91outputs\'92: contains FarNet-II outputs for the given inputs (empty until execution).\
\
Limits of magnetic levels for masks and outputs:\
\
-4: [-inf , -108.92]\
-3: [-108.92, -50.55]\
-2: [-50.55, -24.75]\
-1: [-24.75, -10.78]\
0: [-10.78, 11.03]\
1: [11.03, 25.01]\
2: [25.01, 50.07]\
3: [50.07, 104.84]\
4: [104.84, inf]\
\
\'b7 \'91graphs\'92: contains the image results from the execution of graphs.py and dice_vol.py, showing magnetograms from Solar Orbiter, HMI and the outputs from FarNet-II, and the volumetric representations of all those images (empty until execution).\
\
\'b7 \'91farside_to_magnetograms_9levels.py\'92: is the script that needs to run in order to produce the outputs.\
\
\'b7 \'91checkpoint.pth\'92: is the file containing the parameters extracted from the training of the network.\
\
\'b7 \'91FarNet-II_magnetograms.py\'92: is the deep learning model of FarNet-II adapted to the production of magnetograms.\
\
\'b7 \'91graphs.py\'92: is a script to display the results.\
}