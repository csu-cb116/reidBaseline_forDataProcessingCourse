from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .cross_entropy_loss import CrossEntropyLoss
from .hard_mine_triplet_loss import TripletLoss


def DeepSupervision(criterion, xs, y):
    """
    Args:
    - criterion: loss function
    - xs: tuple of inputs
    - y: ground truth
    """
    loss = 0.
    for i in range(len(xs)):
        if i == len(xs) - 1:
            ratio = 0.7
        else:
            ratio = 0.3 / (len(xs)-1)
        loss += ratio * criterion(xs[i], y)
    # for x in xs:
    #     loss += criterion(x, y)
    # loss /= len(xs)
    return loss
