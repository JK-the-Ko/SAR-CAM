import torch
import torch.nn as nn

class Loss(nn.Module) :
    def __init__(self, device, lambda_tv) :
        # Inheritance
        super(Loss, self).__init__()

        # Initialize Device
        self._device_ = device

        # Initialize Loss Weight
        self._lambda_tv_ = lambda_tv

        # Create Loss Instance
        self._loss_function_ = nn.MSELoss()
        self._dg_loss_ = DGLoss(self._loss_function_, self._device_)
        self._tv_loss_ = TVLoss()

    def forward(self, inputs, preds, targets) :
        # Pixel Loss
        dg_loss = self._dg_loss_(inputs, preds, targets)

        # TV Loss
        tv_loss = self._tv_loss_(preds)

        return dg_loss + self._lambda_tv_ * tv_loss

class DGLoss(nn.Module) :
    def __init__(self, loss_function, device) :
        # Inheritance
        super(DGLoss, self).__init__()

        # Initialize Device
        self._device_ = device

        # Initialize Loss Function
        self._loss_function_ = loss_function

    def forward(self, inputs, preds, targets) :
        # Get Loss
        loss_denominator = self._loss_function_(preds, targets)
        loss_numerator = self._loss_function_(inputs, targets)

        # Calculate DG Loss
        dg_loss = torch.log10(loss_denominator / loss_numerator)

        return dg_loss

class TVLoss(nn.Module) :
    def __init__(self) :
        # Inheritance
        super(TVLoss, self).__init__()

    def forward(self, x) :
        # Initialize Variables
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]

        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])

        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()

        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t) :
        return t.size()[1] * t.size()[2] * t.size()[3]