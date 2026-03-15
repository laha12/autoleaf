import yaml


def _convert_str_to_num(value):
    """辅助函数：将字符串形式的数字转为int/float，其他类型保持不变"""
    if isinstance(value, str):
        # 处理整数（如 "32" → 32）
        try:
            return int(value)
        except ValueError:
            pass
        # 处理浮点数（如 "1e-8"、"0.001" → 对应浮点数）
        try:
            return float(value)
        except ValueError:
            pass
    # 嵌套字典递归转换
    if isinstance(value, dict):
        return {k: _convert_str_to_num(v) for k, v in value.items()}
    # 非字符串/字典类型直接返回
    return value


class Config:
    def __init__(self, cfg_dict):
        # 先对字典做全局类型转换（字符串数字→数值）
        converted_dict = _convert_str_to_num(cfg_dict)
        # 再递归转为Config对象
        for k, v in converted_dict.items():
            if isinstance(v, dict):
                v = Config(v)
            setattr(self, k, v)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    return Config(cfg_dict)


def cfg_to_dict(cfg):
    result = {}
    for k, v in cfg.__dict__.items():
        if hasattr(v, "__dict__"):
            result[k] = cfg_to_dict(v)
        else:
            result[k] = v
    return result