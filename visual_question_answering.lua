require 'torch'
require 'nn'
require 'lfs'
require 'image'
require 'loadcaffe'
utils = require 'misc.utils'

cmd = torch.CmdLine()
cmd:text('Options')

-- Model parameters
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-input_sz', 224, 'Input image dimensions (use 227 for AlexNet)')
cmd:option('-backend', 'nn')

-- Grad-CAM parameters
cmd:option('-layer_name', 'relu5_4', 'Layer to use for Grad-CAM (use relu5_3 for VGG-16 and relu5 for AlexNet)')
cmd:option('-input_image_path', 'images/cat_dog.jpg', 'Input image path')
cmd:option('-question', 'What animal?', 'Input question')
cmd:option('-answer', '', 'Optional answer (For eg. "cat") to generate Grad-CAM for ("" = use predicted answer).')
cmd:option('-save_as_heatmap', 1, 'Whether to save heatmap or raw Grad-CAM. 1 = save heatmap, 0 = save raw Grad-CAM.')

-- VQA model parameters
cmd:option('-model_path', 'VQA_LSTM_CNN/lstm.t7', 'Path to VQA model checkpoint')
cmd:option('-input_encoding_size', 200, 'Encoding size of each token in the vocabulary')
cmd:option('-rnn_size', 512, 'Size of the LSTM hidden state')
cmd:option('-rnn_layers', 2, 'Number of the LSTM layers')
cmd:option('-common_embedding_size', 1024, 'Size of the common embedding vector')
cmd:option('-num_output', 1000, 'Number of output answers')

-- Miscellaneous
cmd:option('-seed', 123, 'Torch manual random number generator seed')
cmd:option('-gpuid', -1, '0-indexed id of GPU to use. -1 = CPU')
cmd:option('-out_path', 'output/', 'Output path')

-- Parse command-line parameters
opt = cmd:parse(arg or {})
print(opt)

torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.DoubleTensor')
lfs.mkdir(opt.out_path)

if opt.gpuid >= 0 then
  require 'cunn'
  require 'cutorch'
  cutorch.setDevice(opt.gpuid + 1)
  cutorch.manualSeed(opt.seed)
end

-- Load CNN
local cnn = loadcaffe.load(opt.proto_file, opt.model_file, opt.backend)

-- Set to evaluate and remove linear+softmax layer
cnn:evaluate()
cnn:remove()
cnn:remove()
cnn:add(nn.Normalize(2))

-- Clone & replace ReLUs for Guided Backprop
local cnn_gb = cnn:clone()
cnn_gb:replace(utils.guidedbackprop)

-- VQA-specific dependencies
-- https://github.com/VT-vision-lab/VQA_LSTM_CNN/blob/master/eval.lua
require 'VQA_LSTM_CNN/misc.netdef'
require 'VQA_LSTM_CNN/misc.RNNUtils'
LSTM = require 'VQA_LSTM_CNN/misc.LSTM'
cjson = require 'cjson'

-- Load vocabulary
local file = io.open('VQA_LSTM_CNN/data_prepro.json','r')
local text = file:read()
file:close()
local json_file = cjson.decode(text)
local vocabulary_size_q = 0
for i, w in pairs(json_file['ix_to_word']) do vocabulary_size_q = vocabulary_size_q + 1 end

-- VQA model definition
local embedding_net_q = nn.Sequential()
  :add(nn.Linear(vocabulary_size_q, opt.input_encoding_size))
  :add(nn.Dropout(0.5))
  :add(nn.Tanh())

local encoder_net_q = LSTM.lstm_conventional(opt.input_encoding_size, opt.rnn_size, 1, opt.rnn_layers, 0.5)

local multimodal_net = nn.Sequential()
  :add(netdef.AxB(2 * opt.rnn_size * opt.rnn_layers, 4096, opt.common_embedding_size, 0.5))
  :add(nn.Dropout(0.5))
  :add(nn.Linear(opt.common_embedding_size, opt.num_output))

local dummy_state_q = torch.Tensor(opt.rnn_size * opt.rnn_layers * 2):fill(0)
local dummy_output_q = torch.Tensor(1):fill(0)

-- Ship model to GPU
if opt.gpuid >= 0 then
  embedding_net_q:cuda()
  encoder_net_q:cuda()
  multimodal_net:cuda()
  dummy_state_q = dummy_state_q:cuda()
  dummy_output_q = dummy_output_q:cuda()
end

-- Set to evaluate
embedding_net_q:evaluate()
encoder_net_q:evaluate()
multimodal_net:evaluate()

-- Zero gradients
embedding_net_q:zeroGradParameters()
encoder_net_q:zeroGradParameters()
multimodal_net:zeroGradParameters()

-- Load pretrained VQA model
embedding_w_q, embedding_dw_q = embedding_net_q:getParameters()
encoder_w_q, encoder_dw_q = encoder_net_q:getParameters()
multimodal_w, multimodal_dw = multimodal_net:getParameters()

model_param = torch.load(opt.model_path)
embedding_w_q:copy(model_param['embedding_w_q'])
encoder_w_q:copy(model_param['encoder_w_q'])
multimodal_w:copy(model_param['multimodal_w'])

local encoder_net_buffer_q = dupe_rnn(encoder_net_q, 26)

-- Load image
local img = utils.preprocess(opt.input_image_path, opt.input_sz, opt.input_sz)

-- Ship CNNs and image to GPU
if opt.gpuid >= 0 then
  cnn:cuda()
  cnn_gb:cuda()
  img = img:cuda()
end

-- Forward pass
fv_im = cnn:forward(img)
fv_im_gb = cnn_gb:forward(img)

-- Tokenize question
local cmd = 'python misc/prepro_ques.py --question "'.. opt.question..'"'
os.execute(cmd)
file = io.open('ques_feat.json')
text = file:read()
file:close()
q_feats = cjson.decode(text)

question = right_align(torch.LongTensor{q_feats.ques}, torch.LongTensor{q_feats.ques_length})
fv_sorted_q = sort_encoding_onehot_right_align(question, torch.LongTensor{q_feats.ques_length}, vocabulary_size_q)

-- Ship question features to GPU
if opt.gpuid >= 0 then
  fv_sorted_q[1] = fv_sorted_q[1]:cuda()
  fv_sorted_q[3] = fv_sorted_q[3]:cuda()
  fv_sorted_q[4] = fv_sorted_q[4]:cuda()
else
    fv_sorted_q[1] = fv_sorted_q[1]:double()
end

local question_max_length = fv_sorted_q[2]:size(1)

-- Embedding forward
local word_embedding_q = split_vector(embedding_net_q:forward(fv_sorted_q[1]), fv_sorted_q[2])

-- Encoder forward
local states_q, _ = rnn_forward(encoder_net_buffer_q, torch.repeatTensor(dummy_state_q:fill(0), 1, 1), word_embedding_q, fv_sorted_q[2])

-- Multimodal forward
local tv_q = states_q[question_max_length + 1]:index(1, fv_sorted_q[4])
local scores = multimodal_net:forward({tv_q, fv_im})

-- Get predictions
_, pred = torch.max(scores:double(), 2)
answer = json_file['ix_to_ans'][tostring(pred[{1, 1}])]

local inv_vocab = utils.table_invert(json_file['ix_to_ans'])
-- Replace out of vocabulary answers with predicted answer
if opt.answer ~= '' and inv_vocab[opt.answer] ~= nil then answer_idx = inv_vocab[opt.answer] else opt.answer = answer answer_idx = inv_vocab[answer] end

print("Question: ", opt.question)
print("Predicted answer: ", answer)
print("Grad-CAM answer: ", opt.answer)

-- Set gradInput
local doutput = utils.create_grad_input(multimodal_net.modules[#multimodal_net.modules], answer_idx)

-- Multimodal backward
local tmp = multimodal_net:backward({tv_q, fv_im}, doutput:view(1,-1))
local dcnn = tmp[2]

-- Grad-CAM
local gcam = utils.grad_cam(cnn, opt.layer_name, dcnn)
gcam = image.scale(gcam:float(), opt.input_sz, opt.input_sz)
local hm = utils.to_heatmap(gcam)
if opt.save_as_heatmap == 1 then
  image.save(opt.out_path .. 'vqa_gcam_hm_' .. opt.answer .. '.png', image.toDisplayTensor(hm))
else
  image.save(opt.out_path .. 'vqa_gcam_' .. opt.answer .. '.png', image.toDisplayTensor(gcam))
end

-- Guided Backprop
local gb_viz = cnn_gb:backward(img, dcnn)
-- BGR to RGB
gb_viz = gb_viz:index(1, torch.LongTensor{3, 2, 1})
image.save(opt.out_path .. 'vqa_gb_' .. opt.answer .. '.png', image.toDisplayTensor(gb_viz))

-- Guided Grad-CAM
local gb_gcam = gb_viz:float():cmul(gcam:expandAs(gb_viz))
image.save(opt.out_path .. 'vqa_gb_gcam_' .. opt.answer .. '.png', image.toDisplayTensor(gb_gcam))
