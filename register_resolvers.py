# register_resolvers.py
from omegaconf import OmegaConf

OmegaConf.register_resolver("eval", lambda x: eval(x))
OmegaConf.register_resolver("concat", lambda *x: [xxx for xx in x for xxx in xx])
OmegaConf.register_resolver("get_fname", lambda: "dummy")
OmegaConf.register_resolver("load_yaml", lambda x: list(OmegaConf.load(x)))