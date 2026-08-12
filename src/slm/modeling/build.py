from slm.utils.registry import Registry

META_ARCH_REGISTRY = Registry("META_ARCH")


def build_model(cfg):
    """Build the top-level model from cfg.model.meta_architecture."""
    name = cfg.model.meta_architecture
    return META_ARCH_REGISTRY.get(name)(cfg)
