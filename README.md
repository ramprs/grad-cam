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

#### Examples

![](http://i.imgur.com/OAoSQYT.png)
![](http://i.imgur.com/iZuijZy.png)
![](http://i.imgur.com/o7RStQm.png)

'boxer' (243)

![](http://i.imgur.com/OAoSQYT.png)
![](http://i.imgur.com/NzXRy5E.png)
![](http://i.imgur.com/fP0Dd87.png)

'tiger cat' (283)

### Visual Question Answering

Clone the [VQA][5] ([http://arxiv.org/abs/1505.00468][4]) sub-repository (`git submodule init && git submodule update`), and download and unzip the provided extracted features and pretrained model.

```
th visual_question_answering.lua -input_image_path images/cat_dog.jpg -question 'What animal?' -answer 'dog' -gpuid 0
th visual_question_answering.lua -input_image_path images/cat_dog.jpg -question 'What animal?' -answer 'cat' -gpuid 0

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

#### Examples

![](http://i.imgur.com/OAoSQYT.png)
![](http://i.imgur.com/QBTstax.png)
![](http://i.imgur.com/NRyhfdL.png)

What animal? Dog

![](http://i.imgur.com/OAoSQYT.png)
![](http://i.imgur.com/hqBWRAm.png)
![](http://i.imgur.com/lwj5oAX.png)

What animal? Cat

![](http://i.imgur.com/CUIiOrd.png)
![](http://i.imgur.com/6oS8lQp.png)
![](http://i.imgur.com/1za35Sj.png)

What color is the hydrant? Yellow

![](http://i.imgur.com/CUIiOrd.png)
![](http://i.imgur.com/UY8moms.png)
![](http://i.imgur.com/DDsMv7A.png)

What color is the hydrant? Green

### Image Captioning

Clone the [neuraltalk2][6] sub-repository. Running sh models/download_models.sh will download the pretrained model and place it in the neuraltalk2 folder

```
th captioning.lua -input_image_path images/cat_dog.jpg -sentence 'a dog with a cat' -gpuid 0
th captioning.lua -input_image_path images/cat_dog.jpg -sentence '' -gpuid 0

```
#### Options

- `input_image_path`: Path to the input image. Default is `images/cat_dog.jpg`.
- `input_sz`: Input image size. Default is 224 (Change to 227 if using AlexNet).
- `layer`: Layer to use for Grad-CAM. Default is 30 (relu5_3 for vgg16)
- `sentence`: Input sentence. Default is the generated caption for the image.
- `out_path`: Path to save images in. Default is `output/`.
- `model_path`: Path to captioning model checkpoint. Default is `neuraltalk2/model_id1-501-1448236541.t7`.
- `gpuid`: 0-indexed id of GPU to use. Default is -1 = CPU.
- `backend`: Backend to use with [loadcaffe][3]. Default is `cudnn`.

#### Examples

![](http://i.imgur.com/OAoSQYT.png)
![](http://i.imgur.com/nRYCRd8.png)
![](http://i.imgur.com/vHyFqJi.png)
a dog and cat posing for a picture

![](http://i.imgur.com/gE6VXql.png)
![](http://i.imgur.com/AsYkclC.png)
![](http://i.imgur.com/eg9YpzD.png)
a bathroom with a toilet and a sink

## License

BSD

[3]: https://github.com/szagoruyko/loadcaffe
[4]: http://arxiv.org/abs/1505.00468
[5]: https://github.com/VT-vision-lab/VQA_LSTM_CNN
[6]: https://github.com/karpathy/neuraltalk2 
