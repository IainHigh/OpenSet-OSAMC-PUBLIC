# Open-set Multi-label Automatic Modulation Classification for Overlapping Radio Frequency Digital Communications

This repository contains all the code associated with the academic paper: "Open-set Multi-label Automatic Modulation Classification for Overlapping Radio Frequency Digital Communications". Submitted for the IEEE 15th International Symposium on Communication Systems, Networks, and Digital Signal Processing (CSNDSP 2026).

This work was completed by Iain High (University of Edinburgh); David Sadler (Roke Manor Research Ltd.); Yoann Altmann (Heriot-Watt University); and Wasiu O. Popoola (University of Edinburgh).

For correspondance, please contact Mr. Iain High.
Institutional email: i.high@sms.ed.ac.uk
Personal email: iain.high@sky.com

## Abstract

Automatic modulation classification is a key technology for modern adaptive communication receivers. Practical deployments must operate in crowded radio frequency spectrum conditions. In these settings, transmissions can overlap, and previously unseen modulation schemes may be present. These conditions reduce the efficacy of current modulation classification algorithms. This paper addresses overlapping signal automatic modulation classification under open-set conditions by formulating the task as multi-label classification with explicit unknown rejection. A synthetic co-channel dataset was generated, comprising four known modulation schemes for training and four additional unknown schemes for evaluation. A compact one-dimensional residual convolutional network was trained using class-weighted binary cross-entropy together with a triplet loss to structure the penultimate feature space. Open-set recognition was then performed by fitting diagonal Gaussian models to known modulation superclasses. During inference, a received frame is only assigned to the known superclass if its Mahalanobis distance lies within a calibrated quantile threshold. Results demonstrate that this formulation enables strong classification accuracy of overlapping transmissions and reliable rejection of transmissions containing previously unseen modulation schemes.

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
5. The data generation script and model scripts can now be run through the entry point 'gen_and_model.py'. This requires the user to only specify the root directory for the dataset.
   ```
   python3 gen_and_model.py --dir <path_to_dataset_root>
   ```
   
## License

This repository is licensed under "The MIT License" (https://opensource.org/license/mit)

Copyright 2026 Iain High

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
