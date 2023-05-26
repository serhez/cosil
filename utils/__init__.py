from . import co_adaptation, imitation, math, model, rl


def dict_add(target, new_data):
    for key, val in new_data.items():
        if key in target:
            if isinstance(val, dict):
                dict_add(target[key], val)
            elif hasattr(val, "__add__"):
                target[key] += val
            else:
                # Unknown data type,just leave it
                pass
        else:
            # target[key] = copy.deepcopy(val)
            target[key] = val


def dict_div(target, value):
    for key, val in target.items():
        if isinstance(val, dict):
            dict_div(target[key], value)
        elif hasattr(val, "__truediv__"):
            target[key] /= value
        else:
            # Unknown data type,just leave it
            pass
