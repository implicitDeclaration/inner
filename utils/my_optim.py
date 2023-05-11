import torch
import torch.optim._functional as F
from torch.optim.optimizer import Optimizer, required


class Adam(Optimizer):
    r"""Implements Adam algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \beta_1, \beta_2
                \text{ (betas)},\theta_0 \text{ (params)},f(\theta) \text{ (objective)}          \\
            &\hspace{13mm}      \lambda \text{ (weight decay)},  \: amsgrad                      \\
            &\textbf{initialize} :  m_0 \leftarrow 0 \text{ ( first moment)},
                v_0\leftarrow 0 \text{ (second moment)},\: \widehat{v_0}^{max}\leftarrow 0\\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}\textbf{if} \: \lambda \neq 0                                           \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{5mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{5mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
            &\hspace{5mm}\textbf{if} \: amsgrad                                                  \\
            &\hspace{10mm}\widehat{v_t}^{max} \leftarrow \mathrm{max}(\widehat{v_t}^{max},
                \widehat{v_t})                                                                   \\
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}^{max}} + \epsilon \big)                                 \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `Adam: A Method for Stochastic Optimization`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, mask=None,
                 weight_decay=0, mask_ratio=0.15, mask_type='linear', device=None, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)
        if mask is not None:
            self.d_p_masks = mask
        else:
            self.d_p_masks = self.get_mask(ratio=mask_ratio, type=mask_type, device=device)  # random mask

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def get_mask(self, ratio, type, device, grain='neuron'):
       # print("getting mask...")
        masks = []
        if grain == 'neuron':  # neuron level mask
            for group in self.param_groups:
                for p in group['params']:
                    if type == 'conv':
                        if len(p.size()) > 2:  # only mask conv layers and
                            n_rand = torch.rand(p.size()[0], device=device)
                            d_p_mask = torch.where(n_rand < ratio, 1, 0)
                            d_p_mask = d_p_mask.unsqueeze(dim=1)
                            d_p_mask = d_p_mask.unsqueeze(dim=2)
                            d_p_mask = d_p_mask.unsqueeze(dim=3)
                            d_p_mask = d_p_mask.expand(p.size())
                            d_p_mask.to(device)
                            masks.append(d_p_mask)
                    else:
                        if len(p.size()) == 2:
                            n_rand = torch.rand(p.size()[0], device=device)
                            d_p_mask = torch.where(n_rand < ratio, 1, 0)
                            d_p_mask = d_p_mask.unsqueeze(dim=1)
                            d_p_mask = d_p_mask.expand(p.size())
                            d_p_mask.to(device)
                            masks.append(d_p_mask)
            return masks
        else:  # weight level mask
            for group in self.param_groups:
                for p in group['params']:
                    if len(p.size()) > 2:  # only mask conv layers
                        d_p_mask = torch.rand(p.size(), device=device)
                        d_p_mask = torch.where(d_p_mask < ratio, 1, 0)
                        d_p_mask.to(device)
                        masks.append(d_p_mask)
            return masks

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        p_cnt = 0
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    if len(p.size()) > 2:
                        p.grad.data = torch.mul(p.grad.data, self.d_p_masks[p_cnt])
                        p_cnt += 1

                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            F.adam(params_with_grad,
                   grads,
                   exp_avgs,
                   exp_avg_sqs,
                   max_exp_avg_sqs,
                   state_steps,
                   amsgrad=group['amsgrad'],
                   beta1=beta1,
                   beta2=beta2,
                   lr=group['lr'],
                   weight_decay=group['weight_decay'],
                   eps=group['eps'])
        return loss
'''F.adam(params_with_grad,
                   grads,
                   exp_avgs,
                   exp_avg_sqs,
                   max_exp_avg_sqs,
                   state_steps,
                   amsgrad=group['amsgrad'],
                   beta1=beta1,
                   beta2=beta2,
                   lr=group['lr'],
                   weight_decay=group['weight_decay'],
                   eps=group['eps'])
        return loss'''



class MySGD(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0, weight_decay1=0, weight_decay2=0, nesterov=False,
                 mask_ratio=0.2, mask_type='conv', assign_mask=None, device=None):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay1=weight_decay1, weight_decay2=weight_decay2, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(MySGD, self).__init__(params, defaults)

        if assign_mask is None:
            self.d_p_masks = self.get_mask(ratio=mask_ratio, type=mask_type, device=device)  # random mask
        else:
            # mask_check = self.get_mask(ratio=mask_ratio, type=mask_type, device=device)
            self.d_p_masks = assign_mask
            # for m1, m2 in zip(mask_check, assign_mask):
            #     print(m1.shape)
            #     print(m2.shape)


    def __setstate__(self, state):
        super(MySGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def get_mask(self, ratio, type, device, grain='neuron'):
       # print("getting mask...")
        masks = []
        if grain == 'neuron':  # neuron level mask
            for group in self.param_groups:
                for p in group['params']:
                    if type == 'conv':
                        if len(p.size()) > 2:  # only mask conv layers and
                            n_rand = torch.rand(p.size()[0], device=device)
                            d_p_mask = torch.where(n_rand < ratio, 1, 0)
                            d_p_mask = d_p_mask.unsqueeze(dim=1)
                            d_p_mask = d_p_mask.unsqueeze(dim=2)
                            d_p_mask = d_p_mask.unsqueeze(dim=3)
                            d_p_mask = d_p_mask.expand(p.size())
                            d_p_mask.to(device)
                            masks.append(d_p_mask)
                    else:
                        if len(p.size()) == 2:
                            n_rand = torch.rand(p.size()[0], device=device)
                            d_p_mask = torch.where(n_rand < ratio, 1, 0)
                            d_p_mask = d_p_mask.unsqueeze(dim=1)
                            d_p_mask = d_p_mask.expand(p.size())
                            d_p_mask.to(device)
                            masks.append(d_p_mask)
            return masks
        else:  # weight level mask
            for group in self.param_groups:
                for p in group['params']:
                    if len(p.size()) > 2:  # only mask conv layers
                        d_p_mask = torch.rand(p.size(), device=device)
                        d_p_mask = torch.where(d_p_mask < ratio, 1, 0)
                        d_p_mask.to(device)
                        masks.append(d_p_mask)
            return masks

    def step(self, closure=None):
        """Performs a single optimization step. Arguments: closure (callable, optional): A closure that reevaluates the model and returns the loss. """
        loss = None
        if closure is not None:
            loss = closure()
        p_cnt = 0
        for group in self.param_groups:
            weight_decay1 = group['weight_decay1']
            weight_decay2 = group['weight_decay2']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay1 != 0:
                    d_p.add_(weight_decay1, torch.sign(p.data))
                if weight_decay2 != 0:
                    d_p.add_(weight_decay2, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                # if p_cnt < len(self.d_p_masks):
                #     if self.d_p_masks[p_cnt].size() == p.size():
                #         d_p = torch.mul(d_p, self.d_p_masks[p_cnt])
                #         p_cnt += 1
                if p.size() == self.d_p_masks.size():
                    # p.grad.data *= self.d_p_masks[p]
                # if p in self.d_p_masks:
                    d_p *= self.d_p_masks

                p.data.add_(-group['lr'], d_p)

        return loss
