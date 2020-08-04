#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import torch
import torch.nn.functional as F
from collections import namedtuple
from colorama import *
init()

from .policy import ScheduledTrainingPolicy, PolicyLoss, LossComponent
from pytorch_toolbelt.losses.focal import FocalLoss

DistillationLossWeights = namedtuple('DistillationLossWeights',
                                     ['distill', 'student', 'teacher'])

def showTensor(v, name):
    print(name + " shape:{0} range:[{1}, {2}]".format(v.shape, torch.min(v), torch.max(v)))

def add_distillation_args(argparser, arch_choices=None, enable_pretrained=False):
    """
    Helper function to make it easier to add command line arguments for knowledge distillation to any script

    Arguments:
        argparser (argparse.ArgumentParser): Existing parser to which to add the arguments
        arch_choices: Optional list of choices to be enforced by the parser for model selection
        enable_pretrained (bool): Flag to enable/disable argument for "pre-trained" models.
    """
    group = argparser.add_argument_group('Knowledge Distillation Training Arguments')
    group.add_argument('--kd-teacher', choices=arch_choices, metavar='ARCH',
                       help='Model architecture for teacher model')
    if enable_pretrained:
        group.add_argument('--kd-pretrained', action='store_true', help='Use pre-trained model for teacher')
    group.add_argument('--kd-resume', type=str, default='', metavar='PATH',
                       help='Path to checkpoint from which to load teacher weights')
    group.add_argument('--kd-temperature', '--kd-temp', dest='kd_temp', type=float, default=1.0, metavar='TEMP',
                       help='Knowledge distillation softmax temperature')
    group.add_argument('--kd-distill-wt', '--kd-dw', type=float, default=0.5, metavar='WEIGHT',
                       help='Weight for distillation loss (student vs. teacher soft targets)')
    group.add_argument('--kd-student-wt', '--kd-sw', type=float, default=0.5, metavar='WEIGHT',
                       help='Weight for student vs. labels loss')
    group.add_argument('--kd-teacher-wt', '--kd-tw', type=float, default=0.0, metavar='WEIGHT',
                       help='Weight for teacher vs. labels loss')
    group.add_argument('--kd-start-epoch', type=int, default=0, metavar='EPOCH_NUM',
                       help='Epoch from which to enable distillation')
    group.add_argument('--kd-loss-type', dest='kd_loss_type', default="KL",
                       help='specify knowledge distillation loss type')
    group.add_argument('--kd-focal-alpha', type=float, dest='kd_focal_alpha', default="0.5",
                       help='balancing factor')
    group.add_argument('--kd-focal-adaptive', type=bool, dest='kd_focal_adaptive', default=False,
                       help='use adaptive focal distillation')

class KnowledgeDistillationPolicy(ScheduledTrainingPolicy):
    """
    Policy which enables knowledge distillation from a teacher model to a student model, as presented in [1].

    Notes:
        1. In addition to the standard policy callbacks, this class also provides a 'forward' function that must
           be called instead of calling the student model directly as is usually done. This is needed to facilitate
           running the teacher model in addition to the student, and for caching the logits for loss calculation.
        2. [TO BE ENABLED IN THE NEAR FUTURE] Option to train the teacher model in parallel with the student model,
           described as "scheme A" in [2]. This can be achieved by passing teacher loss weight > 0.
        3. [1] proposes a weighted average between the different losses. We allow arbitrary weights to be assigned
           to each loss.

    Arguments:
        student_model (nn.Module): The student model, that is - the main model being trained. If only initialized with
            random weights, this matches "scheme B" in [2]. If it has been bootstrapped with trained FP32 weights,
            this matches "scheme C".
        teacher_model (nn.Module): The teacher model from which soft targets are generated for knowledge distillation.
            Usually this is a pre-trained model, however in the future it will be possible to train this model as well
            (see Note 1 above)
        temperature (float): Temperature value used when calculating soft targets and logits (see [1]).
        loss_weights (DistillationLossWeights): Named tuple with 3 loss weights
            (a) 'distill' for student predictions (default: 0.5) vs. teacher soft-targets
            (b) 'student' for student predictions vs. true labels (default: 0.5)
            (c) 'teacher' for teacher predictions vs. true labels (default: 0). Currently this is just a placeholder,
                and cannot be set to a non-zero value.

    [1] Hinton et al., Distilling the Knowledge in a Neural Network (https://arxiv.org/abs/1503.02531)
    [2] Mishra and Marr, Apprentice: Using Knowledge Distillation Techniques To Improve Low-Precision Network Accuracy
        (https://arxiv.org/abs/1711.05852)

    """
    def __init__(self, student_model, teacher_model, temperature=1.0,
                 loss_weights=DistillationLossWeights(0.5, 0.5, 0),
                 *,
                 loss_type: str = "KL",
                 use_focal: bool = True,
                 use_adaptive: bool = False,
                 focal_alpha: float = 0.5,
                 use_tb: bool = False,
                 verbose: int = 0):
        super(KnowledgeDistillationPolicy, self).__init__()

        if loss_weights.teacher != 0:
            raise NotImplementedError('Using teacher vs. labels loss is not supported yet, '
                                      'for now teacher loss weight must be set to 0')

        self.active = False

        self.student = student_model
        self.teacher = teacher_model
        self.temperature = temperature
        self.loss_wts = loss_weights
        self.loss_type = loss_type
        self.batch_size = None
        self.num_boxes = None
        self.num_classes = None

        self.last_students_logits = None
        self.last_teacher_logits = None

        # for Focal loss
        self.gamma = 2
        self.alpha = focal_alpha
        self.beta = 1.50
        self.normalized = False

        self.distance_type = "KL"
        self.use_focal = use_focal
        self.use_adaptive = use_adaptive
        self.use_tb = use_tb
        self.cls_dim = 1
        self.verbose = verbose

        if self.use_tb:
            self.criterion = FocalLoss(reduction="none", cls_dim=self.cls_dim, verbose=verbose)
        else:
            self.criterion = None

    def forward(self, *inputs):
        """
        Performs forward propagation through both student and teached models and caches the logits.
        This function MUST be used instead of calling the student model directly.

        Returns:
            The student model's returned output, to be consistent with what a script using this would expect
        """
        if not self.active:
            return self.student(*inputs)

        if self.loss_wts.teacher == 0:
            with torch.no_grad():
                self.last_teacher_logits, _ = self.teacher(*inputs)
        else:
            self.last_teacher_logits, _ = self.teacher(*inputs)

        confidence, localization = self.student(*inputs)
        self.last_students_logits = confidence.clone()

        return confidence, localization

    # Since the "forward" function isn't a policy callback, we use the epoch callbacks to toggle the
    # activation of distillation according the schedule defined by the user
    def on_epoch_begin(self, model, zeros_mask_dict, meta, **kwargs):
        self.active = True

    def on_epoch_end(self, model, zeros_mask_dict, meta):
        self.active = False

    def before_backward_pass(self, model, epoch, minibatch_id, minibatches_per_epoch, loss, zeros_mask_dict,
                             optimizer=None):
        """
        References
        ----------
        Implementation of Focal Loss was hinted by:
        https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/losses/functional.py#L10
        https://github.com/qfgaohao/pytorch-ssd/blob/master/vision/nn/multibox_loss.py
        https://pytorch.org/docs/stable/nn.functional.html

        Convert class indicator into one-hot encoding without torch.nn.functional.one_hot:
        https://blog.shikoan.com/pytorch-onehotencoding/
        """
        # TODO: Consider adding 'labels' as an argument to this callback, so we can support teacher vs. labels loss
        # (Otherwise we can't do it with a sub-class of ScheduledTrainingPolicy)

        if self.verbose > 0:
            print(Fore.CYAN + "KnowledgeDistillationPolicy.before_backward_pass [in] -------------------------" + Style.RESET_ALL)

        if not self.active:
            return None

        if self.last_teacher_logits is None or self.last_students_logits is None:
            raise RuntimeError("KnowledgeDistillationPolicy: Student and or teacher logits were not cached. "
                               "Make sure to call KnowledgeDistillationPolicy.forward() in your script instead of "
                               "calling the model directly.")

        if self.batch_size is None:
            self.batch_size = self.last_students_logits.shape[0]
        if self.num_boxes is None:
            self.num_boxes = self.last_students_logits.shape[1]
        if self.num_classes is None:
            self.num_classes = self.last_students_logits.shape[2]

        # extract hard-target loss
        if isinstance(loss, tuple):
            regression_loss, classification_loss = loss
            if self.verbose > 0:
                showTensor(regression_loss, "regression_loss")
                showTensor(classification_loss, "classification_loss")
            num_pos = self.num_boxes * self.num_classes

        # Calculate distillation loss
        soft_log_probs = F.log_softmax(self.last_students_logits.reshape(-1, self.num_classes) / self.temperature, dim=1)
        soft_targets = F.softmax(self.last_teacher_logits.reshape(-1, self.num_classes) / self.temperature, dim=1)
        #soft_probs = soft_log_probs.exp() # same result to F.softmax(self.last_students_logits / self.temperature, dim=2)

        if self.verbose > 0:
            print("last_students_logits shape:{0} range [{1}, {2}]".format(self.last_students_logits.shape, torch.min(self.last_students_logits), torch.max(self.last_students_logits)))
            print("last_teacher_logits shape:{0} range [{1}, {2}]".format(self.last_teacher_logits.shape, torch.min(self.last_teacher_logits), torch.max(self.last_teacher_logits)))
            showTensor(soft_log_probs, "soft_log_probs")
            showTensor(soft_targets, "soft_targets")

        # The loss passed to the callback is the student's loss vs. the true labels, so we can use it directly, no
        # need to calculate again

        # Each element of KL divergence can be negative value. But, the sum of them must be non-negative.
        # https://stats.stackexchange.com/a/41300
        if self.distance_type == "KL":
            soft_kl_div = F.kl_div(soft_log_probs, soft_targets.detach(), reduction="none")
        else:
            raise Exception("unknown loss type")

        if self.use_focal:
            # compute focal term
            if self.use_tb:
                # use 3rd party tool (pytorch-toolbelt)
                # https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/losses/focal.py
                if self.verbose > 0:
                    print("use pytorch-toolbelt")
                if self.cls_dim == 1:
                    focal_distillation_loss = self.criterion(self.last_students_logits.reshape(-1, self.num_classes)/self.temperature, soft_targets.reshape(-1, self.num_classes))
                elif self.cls_dim == 2:
                    focal_distillation_loss = self.criterion(self.last_students_logits/self.temperature, soft_targets)
            else:
                # The averaging used in PyTorch KL Div implementation is wrong, so we work around as suggested in
                # https://pytorch.org/docs/stable/nn.html#kldivloss
                # (Also see https://github.com/pytorch/pytorch/issues/6622, https://github.com/pytorch/pytorch/issues/2259)

                # https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/focal.html#FocalLoss
                if self.use_adaptive:
                    # Normal entropy is calculated by multiplying probability and log-probability
                    # https://discuss.pytorch.org/t/calculating-the-entropy-loss/14510
                    if self.verbose > 0:
                        print("use automated adaptative distillation")
                    soft_distance = soft_kl_div - self.beta * (soft_targets * soft_targets.log())
                else:
                    if abs(self.alpha) < 1.0e-10 or abs(self.alpha - 1.0) < 1.0e-10:
                        if self.verbose > 0:
                            print("Alpha-balancing is disabled")
                        soft_distance = - soft_log_probs * soft_targets.detach()
                    else:
                        if self.verbose > 0:
                            print("Alpha-balancing is enabled")
                        soft_distance = - (np.log(self.alpha) + soft_log_probs) * (1 - self.alpha) * soft_targets.detach()
                focal_term = torch.pow(1. - torch.exp( - soft_distance), self.gamma)

                if self.normalized:
                    norm_factor = 1.0 / (focal_term.sum() + 1e-5)
                else:
                    norm_factor = 1.0
                if self.verbose > 0:
                    print("focal_term shape:{0} range[{1}, {2}]".format(focal_term.shape, torch.min(focal_term), torch.max(focal_term)))
                    print("norm_factor: {0}".format(norm_factor))
                focal_distillation_loss = focal_term.reshape(-1, self.num_classes) * norm_factor * soft_kl_div
            sum_focal_distillation_loss = focal_distillation_loss.sum() / num_pos
            sum_classification_loss = classification_loss.sum()
            sum_regression_loss = regression_loss.sum()
            overall_loss = self.loss_wts.distill * sum_focal_distillation_loss + self.loss_wts.student * (sum_regression_loss + sum_classification_loss)
        else:
            overall_loss = self.loss_wts.student * loss + self.loss_wts.distill * soft_kl_div

        if self.verbose > 0:
            showTensor(soft_distance, "soft_distance")
            showTensor(soft_kl_div, "soft_kl_div")
            print("overall_loss(reduced): {0}".format(overall_loss))
            print(Fore.CYAN + "KnowledgeDistillationPolicy.before_backward_pass [out] -------------------------" + Style.RESET_ALL)

        if self.use_focal:
            return PolicyLoss(overall_loss, [
                        LossComponent('focal distillation Loss', sum_focal_distillation_loss),
                        LossComponent('hard classification Loss', sum_classification_loss),
                        LossComponent('hard regression Loss', sum_regression_loss)
                    ])
        else:
            return PolicyLoss(overall_loss, [
                        LossComponent('soft_distance', soft_distance),
                    ])
