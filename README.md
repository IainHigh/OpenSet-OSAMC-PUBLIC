# Open-set Multi-label Automatic Modulation Classification for Overlapping Radio Frequency Digital Communications

This repository contains all the code associated with the academic paper: "Open-set Multi-label Automatic Modulation Classification for Overlapping Radio Frequency Digital Communications". Submitted for the IEEE 15th International Symposium on Communication Systems, Networks, and Digital Signal Processing (CSNDSP 2026).

This work was completed by Iain High (University of Edinburgh); David Sadler (Roke Manor Research Ltd.); Yoann Altmann (Heriot-Watt University); and Wasiu O. Popoola (University of Edinburgh).

For correspondance, please contact Mr. Iain High.
  Institutional email: i.high@sms.ed.ac.uk
  Personal email: iain.high@sky.com

## Abstract
Automatic modulation classification is a key technology for modern adaptive communication receivers. Practical deployments must operate in crowded radio frequency spectrum conditions. In these settings, transmissions can overlap and previously unseen modulation schemes may be present. This paper addresses overlapped signal automatic modulation classification under open-set conditions by formulating the task as multi-label classification with explicit unknown rejection. A synthetic co-channel dataset was generated, comprising five known modulation schemes for training and four additional unknown schemes for evaluation. A compact one-dimensional residual convolutional network mapped time-domain frames to per-class sigmoid probabilities and was trained using a class-weighted binary cross-entropy loss combined with a batch-hard triplet objective to structure a discriminative feature embedding space. Open-set recognition was then performed by fitting per-class diagonal Gaussian feature models on the training embeddings and filtering predicted labels using quantile-calibrated Mahalanobis distance thresholds. Results show that the proposed approach demonstrates strong accuracy on known modulations while reducing spurious predictions of known signals when unknown signals are present, enabling the reliable rejection of unknown-only transmissions.

## Acknowledgements
This work was supported by the Engineering and Physical Sciences Research Council and Ministry of Defence Centre for Doctoral Training in Sensing, Processing and AI for Defence and Security, [EP/Y013859/1]. This work has made use of the resources provided by the Edinburgh Compute and Data Facility (ECDF) - https://www.ecdf.ed.ac.uk.

## Setup Instructions
In addition to the python packages listed below, the code in this repo is dependent upon [liquid-dsp](https://github.com/jgaeddert/liquid-dsp). To install liquid-dsp, clone the repo linked, and follow the installation instructions in the README. Ensure that you rebind your dynamic libraries using `sudo ldconfig`.

1. Clone or download the repository.
2. Install liquid-dsp as described above.
3. Install the required Python packages.
   ```
   pip install -r requirements.txt
   ```
4. Compile the C modules.
   ```
   cd ./cmodules && make && cd ../
   ```
5. Adjust the paths in "CNN-Model/main.py" and "generator.py" to your working directory and desired dataset directory.
6. Generate datasets using the `generator.py` script with a configuration file from the `configs` directory.
   ```
   python generator.py ./configs/training_set.json
   python generator.py ./configs/testing_set.json
   ```
7. Train and test the model using the `main.py` script in the `CNN-Model` directory.
   ```
   python CNN-Model/main.py
   ```


## generator.py

Python tool to generate synthetic radio frequency (RF) datasets.

Datasets are saved in SigMF format.
Each dataset is a _SigMF Archive_ composed of multiple _SigMF Recordings_.
Each _SigMF Recording_ contains a single capture, saved as a binary file (.sigmf-data files), with an associated metadata file (.sigmf-meta) containing the parameters used to generate that capture.
See the [SigMF specification](https://github.com/gnuradio/SigMF/blob/master/sigmf-spec.md) to read more.

## CNN-Model
This directory contains the PyTorch implementation of the open-set multi-label architecture for overlapping signals automatic modulation classification.

### Files
- `classifier.py`: This module contains the implementation of the ModulationClassifier. A deep learning model for classifying modulation types in wireless signals.
- `configs.py`: Configuration settings for CNN-Model training.
- `dataset.py`: Defines a dataset for loading modulation samples of I/Q data.
- `loss.py`: Defines loss functions for the CNN modulation classifier, including a combined binary cross entropy and triplet loss.
- `main.py`: Entry point for CNN-Model training.
- `open_set.py`: open-set recognition utilities for the modulation classifier.

### Usage
1. Ensure datasets `training` and `testing` are generated. The path to these directories will be referred to as {TRAIN_DIR_PATH} and {TEST_DIR_PATH} respectively.
2. Adjust hyperparameters in `configs.py` if required.
3. Launch training and testing with:
   ```
   python main.py {TRAIN_DIR_PATH} {TEST_DIR_PATH}
   ```
   The script prints training progress and writes optional plots and results when enabled in the configuration file.

## License
This repository is licensed under "The MIT License" (https://opensource.org/license/mit)

Copyright 2026 Iain High

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
