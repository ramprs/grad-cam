local utils = {}

-- Preprocess the image before passing it to a Caffe model.
function utils.preprocess(path, width, height)
  -- local width = width or 224
  -- local height = height or 224
  -- local img = image.scale(image.load(path), width, height):double()
  -- local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  -- img = img:index(1, torch.LongTensor{3, 2, 1}):mul(255.0)
  -- mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  -- img:add(-1, mean_pixel)
  -- return img
  local I = image.load(path)
  local mean_pix = {123.68/255,116.779/255,103.939/255}

  if I:dim() == 2 then
      I = I:view(1,I:size(1),I:size(2))
  end

  if I:size(1) == 1 then
      I = I:expand(3,I:size(2),I:size(3))
  end

  I = image.scale(I,width,width)
  assert(I:size(2) == width and I:size(3) == width)

  for i=1,3 do
      I[i]:add(-mean_pix[i])
  end

  return I
end

-- Undo the above preprocessing.
-- function utils.deprocess(img)
  -- local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  -- mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  -- img = img + mean_pixel
  -- local perm = torch.LongTensor{3, 2, 1}
  -- img = img:index(1, perm):div(255.0)
  -- img
  -- return img
-- end

-- Replace ReLUs with DeconvReLUs
function utils.deconv(m)
  require 'DeconvReLU'
  local name = torch.typename(m)
  if name == 'nn.ReLU' or name == 'cudnn.ReLU' then
    return nn.DeconvReLU()
  else
    return m
  end
end

-- Replace ReLUs with DeconvReLUs
function utils.guidedbackprop(m)
  require 'GuidedBackpropReLU'
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

-- Get Grad-CAM
function utils.grad_cam(cnn, input, doutput, layer_name)
  -- Get layer id
  local layer_id = utils.cnn_layer_id(cnn, layer_name)

  -- Get activations and grad input
  cnn:zeroGradParameters()
  cnn:backward(input, doutput:cuda())

  activations = cnn:get(layer_id).output
  gradients = cnn:get(layer_id+1).gradInput

  weights = torch.sum(gradients:view(activations:size(1), -1), 2)
  map = torch.sum(torch.cmul(activations, weights:view(activations:size(1), 1, 1):expandAs(activations)), 1)

  map = image.scale(map:float(), opt.input_sz, opt.input_sz)
  map = torch.div(map, torch.sum(map)):repeatTensor(3,1,1)
  map = map:float():cmul(torch.gt(map:float(), 0):float())

  return map
end

return utils
