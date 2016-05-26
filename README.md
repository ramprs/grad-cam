# Grad-CAM: Gradient-based Discriminative Localization & Visualization

![Overview](http://i.imgur.com/qBPR3aM.jpg)

## Usage

Download Caffe model(s) and prototxt for VGG-16/VGG-19/AlexNet using `sh models/download_models.sh`.

### Classification

```
th classification.lua -input_image_path images/cat_dog.jpg -label 243 -gpuid 0
th classification.lua -input_image_path images/cat_dog.jpg -label 283 -gpuid 0
```

#### Options

- `proto_file`: Path to the `deploy.prototxt` file for the CNN Caffe model. Default is `models/VGG_ILSVRC_16_layers_deploy.prototxt`.
- `model_file`: Path to the `.caffemodel` file for the CNN Caffe model. Default is `models/VGG_ILSVRC_16_layers.caffemodel`.
- `input_image_path`: Path to the input image. Default is `images/cat_dog.jpg`.
- `input_sz`: Input image size. Default is 224 (Change to 227 if using AlexNet).
- `layer_name`: Layer to use for Grad-CAM. Default is `relu5_3` (use `relu5_4` for VGG-19 and `relu5` for AlexNet).
- `label`: Class label to generate grad-CAM for. Default is 243 (283 = Tiger cat, 243 = Boxer). These correspond to ILSVRC synset IDs.
- `out_path`: Path to save images in. Default is `output/`.
- `gpuid`: 0-indexed id of GPU to use. Default is -1 = CPU.
- `backend`: Backend to use with [loadcaffe][3]. Default is `cudnn`.

### Visual Question Answering

Clone the [VQA][5] ([http://arxiv.org/abs/1505.00468][4]) sub-repository (`git submodule init && git submodule update`), and download and unzip the provided extracted features and pretrained model.

```
th visual_question_answering.lua -input_image_path images/cat_dog.jpg -question 'What animal?' -answer 'cat' -gpuid 0
th visual_question_answering.lua -input_image_path images/cat_dog.jpg -question 'What animal?' -answer 'dog' -gpuid 0

```

#### Options

- `proto_file`: Path to the `deploy.prototxt` file for the CNN Caffe model. Default is `models/VGG_ILSVRC_19_layers_deploy.prototxt`.
- `model_file`: Path to the `.caffemodel` file for the CNN Caffe model. Default is `models/VGG_ILSVRC_19_layers.caffemodel`.
- `input_image_path`: Path to the input image. Default is `images/cat_dog.jpg`.
- `input_sz`: Input image size. Default is 224 (Change to 227 if using AlexNet).
- `layer_name`: Layer to use for Grad-CAM. Default is `relu5_4` (use `relu5_3` for VGG-16 and `relu5` for AlexNet).
- `question`: Input question. Default is `What animal?`.
- `answer`: Answer to generate grad-CAM for. Default is 'cat'.
- `out_path`: Path to save images in. Default is `output/`.
- `model_path`: Path to VQA model checkpoint. Default is `VQA_LSTM_CNN/lstm.t7`.
- `gpuid`: 0-indexed id of GPU to use. Default is -1 = CPU.
- `backend`: Backend to use with [loadcaffe][3]. Default is `cudnn`.

### Image Captioning

## License

BSD

[3]: https://github.com/szagoruyko/loadcaffe
[4]: http://arxiv.org/abs/1505.00468
[5]: https://github.com/VT-vision-lab/VQA_LSTM_CNN