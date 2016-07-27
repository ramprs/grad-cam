#!/bin/sh

cd models

# AlexNet
# wget -c https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt -O bvlc_alexnet_deploy.prototxt
# wget -c http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel

# VGG-16
wget -c https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/c3ba00e272d9f48594acef1f67e5fd12aff7a806/VGG_ILSVRC_16_layers_deploy.prototxt
wget -c http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel

# VGG-19
wget -c https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/bb2b4fe0a9bb0669211cf3d0bc949dfdda173e9e/VGG_ILSVRC_19_layers_deploy.prototxt
wget -c http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel

cd ..

# VQA
cd VQA_LSTM_CNN
wget -c https://filebox.ece.vt.edu/~jiasenlu/codeRelease/vqaRelease/train_only/data_train_val.zip
wget -c https://filebox.ece.vt.edu/~jiasenlu/codeRelease/vqaRelease/train_only/pretrained_lstm_train_val.t7.zip
unzip data_train_val.zip
unzip pretrained_lstm_train_val.t7.zip
cd ..

# neuraltalk2
cd neuraltalk2
wget http://cs.stanford.edu/people/karpathy/neuraltalk2/checkpoint_v1.zip
unzip checkpoint_v1.zip
cd ..
