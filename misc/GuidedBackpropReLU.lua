--[[

    Implementation of GuidedBackpropReLU that backpropagates positive gradients
    to input elements with positive activations; From the paper:

    Striving for Simplicity: The All Convolutional Net
    Jost Tobias Springenberg, Alexey Dosovitskiy, Thomas Brox, Martin Riedmiller
    http://arxiv.org/abs/1412.6806

]]--

local GuidedBackpropReLU = torch.class('nn.GuidedBackpropReLU', 'nn.Module')

function GuidedBackpropReLU:updateOutput(input)
  self.output:resizeAs(input):copy(input)
  return self.output:cmul(torch.gt(input,0):typeAs(input))
end

function GuidedBackpropReLU:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  return self.gradInput:cmul(torch.gt(input,0):typeAs(input)):cmul(torch.gt(gradOutput,0):typeAs(input))
end
