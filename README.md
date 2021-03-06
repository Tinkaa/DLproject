# DLproject: focal loss for Faster R-CNN
In this project, we implemented Faster R-CNN and applied the focal loss function on the Region Proposal Network (RPN) to evaluate the performance of Faster R-CNN.  
Authors: Phu Pham, Kha Nguyen and Tinka Valentijn

### Usage
* Download [PASCAL VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) and then extract to root folder.
* Download [pre-train weights for VGG16 net](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5) and place in root folder.
* Training: `python train_frcnn -[options] [value]`

### Experiment
We replaced the normal (binary-cross-entropy) loss function in RPN with one that has negative modulator, i.e. negative samples will have more penalty. See file `frcnn/focal_loss.py` for implementation, and [paper](https://arxiv.org/pdf/1708.02002.pdf) for further explanation.
