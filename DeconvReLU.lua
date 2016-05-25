--[[

    Implementation of DeconvReLU that backpropagates positive gradients
    irrespective of activations; From the paper:

    Visualizing and Understanding Convolutional Networks
    Matthew D. Zeiler, Rob Fergus
    https://arxiv.org/abs/1311.2901

]]--

local DeconvReLU = torch.class('nn.DeconvReLU', 'nn.Module')

function DeconvReLU:updateOutput(input)
  self.output:resizeAs(input):copy(input)
  return self.output:cmul(torch.gt(input,0):typeAs(input))
end

function DeconvReLU:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  return self.gradInput:cmul(torch.gt(gradOutput,0):typeAs(input))
end
