from .SSANet import SSANet
from .SSANet_dist import SSANet_dist
from .loss import model_loss_train, model_loss_test


__models__ = {
    "SSANet": SSANet,
    "SSANet_dist": SSANet_dist
}
