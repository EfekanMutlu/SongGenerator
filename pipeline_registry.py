import os
import torch

class PipelineRegistry:
    def __init__(self):
        self._modules = {}

    def register_modules(self, **modules):
        for name, module in modules.items():
            self._modules[name] = module
            setattr(self, name, module)

    @classmethod
    def load_modules(cls, in_dir, device):
        data = torch.load(os.path.join(in_dir, "pipeline_modules.pt"), map_location=device)
        return data