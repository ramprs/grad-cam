
# Grad-CAM: Gradient-weighted Class Activation Mapping

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
- `label`: Class label to generate grad-CAM for (-1 = use predicted class, 283 = Tiger cat, 243 = Boxer). Default is -1. These correspond to ILSVRC synset IDs.
- `out_path`: Path to save images in. Default is `output/`.
- `gpuid`: 0-indexed id of GPU to use. Default is -1 = CPU.
- `backend`: Backend to use with [loadcaffe][3]. Default is `nn`.
- `save_as_heatmap`: Whether to save heatmap or raw Grad-CAM. 1 = save heatmap, 0 = save raw Grad-CAM. Default is 1.

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
- `answer`: Optional answer (For eg. "cat") to generate Grad-CAM for ('' = use predicted answer). Default is ''.
- `out_path`: Path to save images in. Default is `output/`.
- `model_path`: Path to VQA model checkpoint. Default is `VQA_LSTM_CNN/lstm.t7`.
- `gpuid`: 0-indexed id of GPU to use. Default is -1 = CPU.
- `backend`: Backend to use with [loadcaffe][3]. Default is `cudnn`.
- `save_as_heatmap`: Whether to save heatmap or raw Grad-CAM. 1 = save heatmap, 0 = save raw Grad-CAM. Default is 1.

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

Clone the [neuraltalk2][6] sub-repository. Running `sh models/download_models.sh` will download the pretrained model and place it in the neuraltalk2 folder.

Change lines 2-4 of `neuraltalk2/misc/LanguageModel.lua` to the following:

```
local utils = require 'neuraltalk2.misc.utils'
local net_utils = require 'neuraltalk2.misc.net_utils'
local LSTM = require 'neuraltalk2.misc.LSTM'
```


```
th captioning.lua -input_image_path images/cat_dog.jpg -caption 'a dog and cat posing for a picture' -gpuid 0
th captioning.lua -input_image_path images/cat_dog.jpg -caption '' -gpuid 0

```
#### Options

- `input_image_path`: Path to the input image. Default is `images/cat_dog.jpg`.
- `input_sz`: Input image size. Default is 224 (Change to 227 if using AlexNet).
- `layer`: Layer to use for Grad-CAM. Default is 30 (relu5_3 for vgg16)
- `caption`: Optional input caption. No input will use the generated caption as default.
- `out_path`: Path to save images in. Default is `output/`.
- `model_path`: Path to captioning model checkpoint. Default is `neuraltalk2/model_id1-501-1448236541.t7`.
- `gpuid`: 0-indexed id of GPU to use. Default is -1 = CPU.
- `backend`: Backend to use with [loadcaffe][3]. Default is `cudnn`.
- `save_as_heatmap`: Whether to save heatmap or raw Grad-CAM. 1 = save heatmap, 0 = save raw Grad-CAM. Default is 1.

#### Examples

![](http://i.imgur.com/OAoSQYT.png)
![](http://i.imgur.com/TiKdMMw.png)
![](http://i.imgur.com/GSQeR2M.png)

a dog and cat posing for a picture

![](http://i.imgur.com/gE6VXql.png)
![](http://i.imgur.com/K3E9TWS.png)
![](http://i.imgur.com/em2oHRy.png)

a bathroom with a toilet and a sink

## License

BSD

### 3rd-party

- [VQA_LSTM_CNN][5], BSD
- [neuraltalk2][6], BSD

[3]: https://github.com/szagoruyko/loadcaffe
[4]: http://arxiv.org/abs/1505.00468
[5]: https://github.com/VT-vision-lab/VQA_LSTM_CNN
[6]: https://github.com/karpathy/neuraltalk2 
