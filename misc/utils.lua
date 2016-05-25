local utils = {}

-- Preprocess the image before passing it to a Caffe model.
function utils.preprocess(path, width, height)
  local width = width or 224
  local height = height or 224
  local img = image.scale(image.load(path), width, height):double()
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  img = img:index(1, torch.LongTensor{3, 2, 1}):mul(255.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img
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
  local doutput = module.output:clone()
  doutput:fill(0)
  doutput[label] = 1
  return doutput
end

-- Generate Grad-CAM
function utils.grad_cam(cnn, layer_name, doutput)
  -- Split model into two
  local model1, model2 = nn.Sequential(), nn.Sequential()
  for i = 1, #cnn.modules do
    model1:add(cnn:get(i))
    layer_id = i
    if cnn:get(i).name == layer_name then
      break
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
  map = map:cmul(torch.gt(map,0))

  return map
end

return utils
