require 'torch'
require 'nn'
require 'image'
require 'loadcaffe'
utils = require 'utils'

cmd = torch.CmdLine()
cmd:text('Options')

-- Model parameters
cmd:option('-proto_file', 'models/VGG_ILSVRC_16_layers_deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_16_layers.caffemodel')
cmd:option('-input_sz', 224, 'Input image dimensions (use 227 for AlexNet)')
cmd:option('-backend', 'cudnn')

-- Grad-CAM parameters
cmd:option('-layer_name', 'relu5_3', 'Layer to use for Grad-CAM (use relu5_4 for VGG-19 and relu5 for AlexNet)')
cmd:option('-input_image_path', 'image.jpg', 'Input image path')
cmd:option('-label', 243, 'Class label to generate grad-CAM for (283 = Tiger cat)')

-- Miscellaneous
cmd:option('-seed', 123, 'Torch manual random number generator seed')
cmd:option('-gpuid', -1, '0-indexed id of GPU to use. -1 = CPU')
cmd:option('-out_path', 'output/', 'Output path')

-- Parse command-line parameters
opt = cmd:parse(arg or {})
print(opt)

torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')

if opt.gpuid >= 0 then
  require 'cunn'
  require 'cutorch'
  cutorch.setDevice(opt.gpuid + 1)
  cutorch.manualSeed(opt.seed)
end

-- Load CNN
cnn = loadcaffe.load(opt.proto_file, opt.model_file, opt.backend)
-- cnn = loadcaffe.load(opt.proto_file, opt.model_file, 'nn')

-- Get layer number corresponding to name
layer_id = utils.cnn_layer_id(cnn, opt.layer_name)
assert(layer_id ~= -1, 'incorrect layer name')

-- Set to evaluate and remove softmax layer
cnn:evaluate()
cnn:remove()

-- Clone & replace ReLUs for Guided Backprop
cnn_gb = cnn:clone()
cnn_gb:replace(utils.guidedbackprop)

-- Load image
img = utils.preprocess(opt.input_image_path, opt.input_sz, opt.input_sz)

weight = cnn:get(1).weight
weight_clone = weight:clone()
nchannels = weight:size(2)
for i=1,nchannels do
  weight:select(2,i):copy(weight_clone:select(2,nchannels+1-i))
end
weight:mul(255)

if opt.gpuid >= 0 then
  cnn:cuda()
  cnn_gb:cuda()
  img = img:cuda()
end

output = cnn:forward(img)
output_gb = cnn_gb:forward(img)

-- Set gradInput
doutput = cnn.output:clone()
doutput:fill(0)
doutput[243] = 1

grad_cam = utils.grad_cam(cnn, img, doutput, opt.layer_name)

image.save(opt.out_path .. 'map_243.png', image.toDisplayTensor(grad_cam))
