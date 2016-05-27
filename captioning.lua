require 'torch'
require 'nn'
require 'lfs'
require 'image'
require 'loadcaffe'
utils = require 'misc.utils'

cmd = torch.CmdLine()
cmd:text('Options')

-- Model parameters
cmd:option('-input_sz', 224, 'Input image dimensions (use 224 for VGG Net)')
cmd:option('-backend', 'cudnn')

-- Grad-CAM parameters
cmd:option('-layer', 30, 'Layer to use for Grad-CAM (use 30 for relu5_3 for VGG-16 )')
cmd:option('-input_image_path', 'images/cat_dog.jpg', 'Input image path')
cmd:option('-sentence', 'a dog and a cat posing for a picture', 'Input sentence. Default is the generated sentence')

-- Captioning model parameters
cmd:option('-model_path', 'neuraltalk2/model_id1-501-1448236541.t7', 'Path to captioning model checkpoint')

-- Miscellaneous
cmd:option('-seed', 123, 'Torch manual random number generator seed')
cmd:option('-gpuid', -1, '0-indexed id of GPU to use. -1 = CPU')
cmd:option('-out_path', 'output/', 'Output path')

-- Parse command-line parameters
opt = cmd:parse(arg or {})
print(opt)

torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')
lfs.mkdir(opt.out_path)

if opt.gpuid >= 0 then
  require 'cunn'
  require 'cudnn'
  require 'cutorch'
  cutorch.setDevice(opt.gpuid + 1)
  cutorch.manualSeed(opt.seed)
end

--Captioning specific dependencies
-- https://github.com/karpathy/neuraltalk2

local lm_misc_utils = require 'neuraltalk2.misc.utils'
require 'neuraltalk2.misc.LanguageModel'
local net_utils = require 'neuraltalk2.misc.net_utils'

-- Load the models
local cnn_lm_model = torch.load(opt.model_path)
local cnn = cnn_lm_model.protos.cnn
local lm = cnn_lm_model.protos.lm
local crit = cnn_lm_model.protos.crit
local vocab = cnn_lm_model.vocab

net_utils.unsanitize_gradients(cnn)
local lm_modules = lm:getModulesList()
for k,v in pairs(lm_modules) do 
  net_utils.unsanitize_gradients(v) 
  
end

local img = utils.preprocess(opt.input_image_path, opt.input_sz, opt.input_sz)
-- Ship model to GPU
if opt.gpuid >= 0 then
  cnn:cuda()
  lm:cuda() 
  --crit:cuda()
  img = img:cuda()
end
-- Set to evaluate mode
lm:evaluate()
cnn:evaluate()

-- Clone & replace ReLUs for Guided Backprop
local cnn_gb = cnn:clone()
cnn_gb:replace(utils.guidedbackprop)

-- Forward pass
im_feats = cnn:forward(img)
im_feat = im_feats:view(1, im_feats:size(1))
im_feat_gb = cnn_gb:forward(img)

-- get the prediction from model
local seq, seqlogps = lm:sample(im_feat, sample_opts)
seq[{{},1}] = seq

local caption = net_utils.decode_sequence(vocab, seq)
print("generated sentence: ", caption[1])

if opt.sentence == '' then 
  print("No sentence provided. Using generated caption as the sentence")
  opt.sentence = caption[1] end

local seq_length = opt.seq_length or 16

local labels = utils.sent_to_label(vocab, opt.sentence, seq_length)

if opt.gpuid >=0 then labels = labels:cuda() end

local logprobs = lm:forward({im_feat,labels})

local doutput = utils.create_grad_input_lm(logprobs, labels)

if opt.gpuid >=0 then doutput = doutput:cuda() end

-- lm backward
local dlm,ddummy = unpack(lm:cuda():backward({im_feat:cuda(),labels:cuda()},doutput:cuda()))

local dcnn = dlm[1]

-- Grad-CAM
local gcam = utils.grad_cam(cnn, opt.layer, dcnn, true)
gcam = image.scale(gcam:float(), opt.input_sz, opt.input_sz)
local hm = utils.to_heatmap(gcam)
image.save(opt.out_path .. 'caption_gcam_'  ..opt.sentence.. '.png', image.toDisplayTensor(hm))


-- Guided Backprop
local gb_viz = cnn_gb:backward(img, dcnn)
image.save(opt.out_path .. 'caption_gb_' .. opt.sentence.. '.png', image.toDisplayTensor(gb_viz))

-- Guided Grad-CAM
local gb_gcam = gb_viz:float():cmul(gcam:expandAs(gb_viz))
image.save(opt.out_path .. 'caption_gb_gcam_' ..opt.sentence .. '.png', image.toDisplayTensor(gb_gcam))
