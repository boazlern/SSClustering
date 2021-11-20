from copy import deepcopy


class ModelEMA(object):
    def __init__(self, model, decay=0.999, device='cpu'):
        self.model = model
        self.ema = deepcopy(model).to(device)
        self.ema.eval()
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, 'module')
        self.step = 0
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        self.buffer_keys = [k for k, _ in self.model.named_buffers()]
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update_params(self):
        decay = min(self.decay, (self.step + 1) / (self.step + 10))
        model_state = self.model.state_dict()
        ema_state = self.ema.state_dict()
        for name in self.param_keys:
            ema_state[name].copy_(
                decay * ema_state[name]
                + (1 - decay) * model_state[name].detach()
            )
        self.step += 1

    def update_buffer(self):
        model_state = self.model.state_dict()
        ema_state = self.ema.state_dict()
        for name in self.buffer_keys:
            ema_state[name].copy_(model_state[name].detach())
