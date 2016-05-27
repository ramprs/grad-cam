local utils = {}

-- Preprocess the image before passing it to a Caffe model.
function utils.preprocess(path, width, height)
  local width = width or 224
  local height = height or 224

  -- load image
  local orig_image = image.load(path)

  -- if the image is grayscale, repeat the tensor
  if orig_image:nDimension() == 2 then
    orig_image = orig_image:repeatTensor(3, 1, 1)
  end

  -- get the dimensions of the original image
  local im_height = orig_image:size(2)
  local im_width = orig_image:size(3)

  -- scale and subtract mean
  local img = image.scale(orig_image, width, height):double()
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  img = img:index(1, torch.LongTensor{3, 2, 1}):mul(255.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img, im_height, im_width
end

-- Replace ReLUs with DeconvReLUs
function utils.deconv(m)
  require 'misc.DeconvReLU'
  local name = torch.typename(m)
  if name == 'nn.ReLU' or name == 'cudnn.ReLU' then
    return nn.DeconvReLU()
  else
    return m
  end
end

-- Replace ReLUs with DeconvReLUs
function utils.guidedbackprop(m)
  require 'misc.GuidedBackpropReLU'
  local name = torch.typename(m)
  if name == 'nn.ReLU' or name == 'cudnn.ReLU' then
    return nn.GuidedBackpropReLU()
  else
    return m
  end
end

-- Get layer id from name
function utils.cnn_layer_id(cnn, layer_name)
  for i = 1, #cnn.modules do
    local layer = cnn:get(i)
    local name = layer.name
    if name == layer_name then
      return i
    end
  end
  return -1
end

-- Synthesize gradInput tensor
function utils.create_grad_input(module, label)
  local doutput = module.output:clone():view(-1)
  doutput:fill(0)
  doutput[label] = 1
  return doutput
end

-- Creates gradInput for neuraltalk2 Language Model
function utils.create_grad_input_lm(input, labels)
  local output = torch.zeros(input:size()):fill(0)
  for t =1,labels:size(1) do
    if labels[t][1]~=0 then
      output[t+1][1][labels[t][1]] = 1
    end
  end
  return output
end

-- Generate Grad-CAM
function utils.grad_cam(cnn, layer_name, doutput)
  -- Split model into two
  local model1, model2 = nn.Sequential(), nn.Sequential()
  if type(layer_name) == "string" then
   for i = 1, #cnn.modules do
      model1:add(cnn:get(i))
      layer_id = i
      if cnn:get(i).name == layer_name then
        break
      end
    end
  else
    layer_id = layer_name
    for i = 1, #cnn.modules do
      model1:add(cnn:get(i))
    if i == layer_id then
        break
      end
    end
  end

  for i = layer_id+1, #cnn.modules do
    model2:add(cnn:get(i))
  end

  -- Get activations and gradients
  model2:zeroGradParameters()
  model2:backward(model1.output, doutput)
  local activations = model1.output
  local gradients = model2.gradInput

  -- Global average pool gradients
  local weights = torch.sum(gradients:view(activations:size(1), -1), 2)

  -- Summing and rectifying weighted activations across depth
  local map = torch.sum(torch.cmul(activations, weights:view(activations:size(1), 1, 1):expandAs(activations)), 1)
  map = map:cmul(torch.gt(map,0):typeAs(map))

  return map
end

function utils.table_invert(t)
  local s = {}
  for k,v in pairs(t) do
    s[v] = k
  end
  return s
end

function utils.sent_to_label(vocab, sent, seq_length)
  local inv_vocab = utils.table_invert(vocab)
  local labels = torch.zeros(seq_length,1)
  local i =0
  for word in sent:gmatch'%w+' do
    local ix_word = inv_vocab[word]
    if ix_word == nil then print("error: word ".. word " doesn't exist in vocab")
      break
    end
    i = i+1
    labels[{{i},{1}}] = ix_word
  end
  return labels
end

function utils.to_heatmap(map)
  map = image.toDisplayTensor(map)
  local cmap = torch.Tensor(3, map:size(2), map:size(3)):fill(1)
  for i = 1, map:size(2) do
    for j = 1, map:size(3) do
      local value = map[1][i][j]
      if value <= 0.25 then
        cmap[1][i][j] = 0
        cmap[2][i][j] = 4*value
      elseif value <= 0.5 then
        cmap[1][i][j] = 0
        cmap[3][i][j] = 2 - 4*value
      elseif value <= 0.75 then
        cmap[1][i][j] = 4*value - 2
        cmap[3][i][j] = 0
      else
        cmap[2][i][j] = 4 - 4*value
        cmap[3][i][j] = 0
      end
    end
  end
  return cmap
end

return utils
