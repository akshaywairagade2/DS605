""" Optimizer Factory w/ Custom Weight Decay
Hacked together by / Copyright 2021 Ross Wightman
"""
import json
from itertools import islice
from typing import Optional, Callable, Tuple, Dict, Union


import torch
import torch.nn as nn
import torch.optim as optim

#from timm.models.helpers import group_parameters

from timm.optim.adabelief import AdaBelief
from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lamb import Lamb
from timm.optim.lars import Lars 
from timm.optim.lookahead import Lookahead
from timm.optim.madgrad import MADGRAD
from timm.optim.nadam import Nadam 
from timm.optim.nvnovograd import NvNovoGrad
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP



try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False




def param_groups_weight_decay(
        model: nn.Module,
        weight_decay=1e-5,
        no_weight_decay_list=()
):
    # Convert the list of parameters with no weight decay to a set for efficient lookup
    no_weight_decay_list = set(no_weight_decay_list)

    # Initialize two lists to separate parameters with and without weight decay
    decay = []
    no_decay = []

    # Iterate through all named parameters in the model
    for name, param in model.named_parameters():
        # Skip parameters that do not require gradients
        if not param.requires_grad:
            continue

        # Check if the parameter has dimension <= 1, is a bias, or is in the no_weight_decay_list
        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            # Parameters without weight decay
            no_decay.append(param)
        else:
            # Parameters with weight decay
            decay.append(param)

    # Return a list of dictionaries, each specifying a group of parameters with its own weight decay
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]



def _group(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def _layer_map(model, layers_per_group=12, num_groups=None):
    def _in_head(n, hp):
        # Check if the parameter name 'n' is in the head prefix 'hp'
        if not hp:
            return True
        elif isinstance(hp, (tuple, list)):
            return any([n.startswith(hpi) for hpi in hp])
        else:
            return n.startswith(hp)

    # Get the head prefix from the model's pretrained_cfg, if available
    head_prefix = getattr(model, 'pretrained_cfg', {}).get('classifier', None)

    # Initialize lists to store parameter names in the trunk and head
    names_trunk = []
    names_head = []

    # Iterate through named parameters in the model
    for n, _ in model.named_parameters():
        # Append parameter name to the head or trunk list based on whether it is in the head or not
        names_head.append(n) if _in_head(n, head_prefix) else names_trunk.append(n)

    # Group non-head layers based on the specified number of layers per group or number of groups
    num_trunk_layers = len(names_trunk)
    if num_groups is not None:
        layers_per_group = -(num_trunk_layers // -num_groups)
    names_trunk = list(_group(names_trunk, layers_per_group))

    # Get the number of trunk groups
    num_trunk_groups = len(names_trunk)

    # Create a dictionary mapping each parameter name to its corresponding trunk group index
    layer_map = {n: i for i, l in enumerate(names_trunk) for n in l}

    # Update the dictionary to include head layers mapped to a separate group index
    layer_map.update({n: num_trunk_groups for n in names_head})

    return layer_map




# A global variable to track the ordinal of the previous group during iteration
MATCH_PREV_GROUP = [-1]

def group_with_matcher(
        named_objects,
        group_matcher: Union[Dict, Callable],
        output_values: bool = False,
        reverse: bool = False
):
    # If the group matcher is specified as a dictionary, compile regular expressions and prepare match specifications
    if isinstance(group_matcher, dict):
        compiled = []
        # Iterate over the dictionary to compile regular expressions and create match specifications
        for group_ordinal, (group_name, mspec) in enumerate(group_matcher.items()):
            if mspec is None:
                continue
            # Map all matching specifications into a 3-tuple (compiled re, prefix, suffix)
            if isinstance(mspec, (tuple, list)):
                # Multi-entry match specifications require each sub-spec to be a 2-tuple (re, suffix)
                for sspec in mspec:
                    compiled += [(re.compile(sspec[0]), (group_ordinal,), sspec[1])]
            else:
                compiled += [(re.compile(mspec), (group_ordinal,), None)]
        # Update group_matcher to the compiled match specifications
        group_matcher = compiled

    def _get_grouping(name):
        # Function to determine the grouping of a named object based on the group matcher
        if isinstance(group_matcher, (list, tuple)):
            for match_fn, prefix, suffix in group_matcher:
                r = match_fn.match(name)
                if r:
                    parts = (prefix, r.groups(), suffix)
                    # Map all tuple elements to float for numeric sort, filter out None entries
                    return tuple(map(float, chain.from_iterable(filter(None, parts))))
            return float('inf'),  # Unmatched layers (neck, head) mapped to the largest ordinal
        else:
            # If the group matcher is a callable, call it to determine the ordinal
            ord = group_matcher(name)
            if not isinstance(ord, collections.abc.Iterable):
                return ord,
            return tuple(ord)

    # Map named objects (e.g., parameters) into groups based on ordinals from the matcher
    grouping = defaultdict(list)
    for k, v in named_objects:
        grouping[_get_grouping(k)].append(v if output_values else k)

    # Remap groups to integers
    layer_id_to_param = defaultdict(list)
    lid = -1
    for k in sorted(filter(lambda x: x is not None, grouping.keys())):
        if lid < 0 or k[-1] != MATCH_PREV_GROUP[0]:
            lid += 1
        layer_id_to_param[lid].extend(grouping[k])

    if reverse:
        assert not output_values, "Reverse mapping only sensible for name output"
        # Output reverse mapping from parameters to layer IDs
        param_to_layer_id = {}
        for lid, lm in layer_id_to_param.items():
            for n in lm:
                param_to_layer_id[n] = lid
        return param_to_layer_id

    return layer_id_to_param


def group_parameters(
        module: nn.Module,
        group_matcher,
        output_values=False,
        reverse=False,
):
    return group_with_matcher(
        module.named_parameters(), group_matcher, output_values=output_values, reverse=reverse)


def group_modules(
        module: nn.Module,
        group_matcher,
        output_values=False,
        reverse=False,
):
    return group_with_matcher(
        named_modules_with_params(module), group_matcher, output_values=output_values, reverse=reverse)




import torch.nn as nn
from typing import Tuple, Optional
import json

def param_groups_layer_decay(
        model: nn.Module,
        weight_decay: float = 0.05,
        no_weight_decay_list: Tuple[str] = (),
        layer_decay: float = 0.75,
        end_layer_decay: Optional[float] = None,
):
    """
    Parameter groups for layer-wise lr decay & weight decay
    Based on BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    # Convert the list of parameters with no weight decay to a set for efficient lookup
    no_weight_decay_list = set(no_weight_decay_list)

    # Initialize dictionaries to store parameter groups and their names (for debugging)
    param_group_names = {}
    param_groups = {}

    # Check if the model has a 'group_matcher' attribute, else use the default layer map
    if hasattr(model, 'group_matcher'):
        # Use the group parameters function with a reversed mapping
        layer_map = group_parameters(model, model.group_matcher(coarse=False), reverse=True)
    else:
        # Fallback to the default layer map if 'group_matcher' is not present
        layer_map = _layer_map(model)

    # Determine the maximum layer ID and calculate layer scales
    num_layers = max(layer_map.values()) + 1
    layer_max = num_layers - 1
    layer_scales = list(layer_decay ** (layer_max - i) for i in range(num_layers))

    # Iterate through named parameters in the model
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine weight decay status based on parameter dimension and the provided list
        if param.ndim == 1 or name in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        # Get the layer ID for the parameter, default to the maximum layer ID if not found
        layer_id = layer_map.get(name, layer_max)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        # Create or update parameter group information
        if group_name not in param_groups:
            this_scale = layer_scales[layer_id]
            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "param_names": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        # Append parameter information to the corresponding group
        param_group_names[group_name]["param_names"].append(name)
        param_groups[group_name]["params"].append(param)

    # Print parameter group information for debugging
    print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    # Return the parameter groups as a list
    return list(param_groups.values())



def optimizer_kwargs(cfg):
    """ cfg/argparse to kwargs helper
    Convert optimizer args in argparse args or cfg like object to keyword args for updated create fn.
    """
    kwargs = dict(
        opt=cfg.opt,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        momentum=cfg.momentum,
        tuning_mode=cfg.tuning_mode)
    if getattr(cfg, 'opt_eps', None) is not None:
        kwargs['eps'] = cfg.opt_eps
    if getattr(cfg, 'opt_betas', None) is not None:
        kwargs['betas'] = cfg.opt_betas
    if getattr(cfg, 'layer_decay', None) is not None:
        kwargs['layer_decay'] = cfg.layer_decay
    if getattr(cfg, 'opt_args', None) is not None:
        kwargs.update(cfg.opt_args)
    return kwargs


def create_optimizer(args, model, filter_bias_and_bn=True):
    """ Legacy optimizer factory for backwards compatibility.
    NOTE: Use create_optimizer_v2 for new code.
    """
    return create_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args),
        filter_bias_and_bn=filter_bias_and_bn,
    )


import torch.nn as nn
from typing import Optional, Callable
import json
from torch.optim import SGD, Adam, AdamW, Adadelta, Adagrad, Adamax, RMSprop, SGD, Adafactor, Adamp
from .optimizers import SGDP, Nadam, RAdam, AdaBelief, Adafactor, Lamb, Lars, MADGRAD, MADGRADW, NvNovoGrad, RMSpropTF, Adahessian
from .apex import FusedSGD, FusedAdam, FusedLAMB, FusedNovoGrad
from .lookahead import Lookahead

def create_optimizer_v2(
        model_or_params,
        opt: str = 'sgd',
        lr: Optional[float] = None,
        weight_decay: float = 0.,
        momentum: float = 0.9,
        tuning_mode: str = None,
        filter_bias_and_bn: bool = True,
        layer_decay: Optional[float] = None,
        param_group_fn: Optional[Callable] = None,
        **kwargs):
    """ Create an optimizer.

    TODO currently the model is passed in and all parameters are selected for optimization.
    For more general use an interface that allows selection of parameters to optimize and lr groups, one of:
      * a filter fn interface that further breaks params into groups in a weight_decay compatible fashion
      * expose the parameters interface and leave it up to caller

    Args:
        model_or_params (nn.Module): model containing parameters to optimize
        opt: name of optimizer to create
        lr: initial learning rate
        weight_decay: weight decay to apply in optimizer
        momentum: momentum for momentum based optimizers (others may use betas via kwargs)
        filter_bias_and_bn: filter out bias, bn and other 1d params from weight decay
        **kwargs: extra optimizer specific kwargs to pass through

    Returns:
        Optimizer
    """
    if isinstance(model_or_params, nn.Module):
        # TODO: for fine-tuning 
        if tuning_mode:
            for name, param in model_or_params.named_parameters():
                if tuning_mode == 'linear_probe':
                    if "head." not in name:
                        param.requires_grad = False
                elif tuning_mode == 'ssf':
                    if "head." not in name and "ssf_scale" not in name and "ssf_shift_" not in name: 
                        param.requires_grad = False

                if param.requires_grad == True:
                    print(name)
                
            print('freezing parameters finished!')

        # a model was passed in, extract parameters and add weight decays to appropriate layers
        no_weight_decay = {}
        if hasattr(model_or_params, 'no_weight_decay'):
            no_weight_decay = model_or_params.no_weight_decay()

        if param_group_fn:
            parameters = param_group_fn(model_or_params)
        elif layer_decay is not None:
            parameters = param_groups_layer_decay(
                model_or_params,
                weight_decay=weight_decay,
                layer_decay=layer_decay,
                no_weight_decay_list=no_weight_decay)
            weight_decay = 0.
        elif weight_decay and filter_bias_and_bn:
            parameters = param_groups_weight_decay(model_or_params, weight_decay, no_weight_decay)
            weight_decay = 0.
        else:
            parameters = model_or_params.parameters()

    else:
        # iterable of parameters or param groups passed in
        parameters = model_or_params

    opt_lower = opt.lower()
    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(weight_decay=weight_decay, **kwargs)
    if lr is not None:
        opt_args.setdefault('lr', lr)

    # basic SGD & related
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        # NOTE 'sgd' refers to SGD + nesterov momentum for legacy / backwards compat reasons
        opt_args.pop('eps', None)
        optimizer = SGD(parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = SGD(parameters, momentum=momentum, nesterov=False, **opt_args)
    elif opt_lower == 'sgdp':
        optimizer = SGDP(parameters, momentum=momentum, nesterov=True, **opt_args)

    # adaptive
    elif opt_lower == 'adam':
        optimizer = Adam(parameters, **opt_args) 
    elif opt_lower == 'adamw':
        optimizer = AdamW(parameters, **opt_args)
    elif opt_lower == 'adamp':
        optimizer = Adamp(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'nadam':
        try:
            # NOTE PyTorch >= 1.10 should have native NAdam
            optimizer = Nadam(parameters, **opt_args)
        except AttributeError:
            optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamax':
        optimizer = Adamax(parameters, **opt_args)
    elif opt_lower == 'adabelief':
        optimizer = AdaBelief(parameters, rectify=False, **opt_args)
    elif opt_lower == 'radabelief':
        optimizer = AdaBelief(parameters, rectify=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = Adadelta(parameters, **opt_args)
    elif opt_lower == 'adagrad':
        opt_args.setdefault('eps', 1e-8)
        optimizer = Adagrad(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'lamb':
        optimizer = Lamb(parameters, **opt_args)
    elif opt_lower == 'lambc':
        optimizer = Lamb(parameters, trust_clip=True, **opt_args)
    elif opt_lower == 'larc':
        optimizer = Lars(parameters, momentum=momentum, trust_clip=True, **opt_args)
    elif opt_lower == 'lars':
        optimizer = Lars(parameters, momentum=momentum, **opt_args)
    elif opt_lower == 'nlarc':
        optimizer = Lars(parameters, momentum=momentum, trust_clip=True, nesterov=True, **opt_args)
    elif opt_lower == 'nlars':
        optimizer = Lars(parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt_lower == 'madgrad':
        optimizer = MADGRAD(parameters, momentum=momentum, **opt_args)
    elif opt_lower == 'madgradw':
        optimizer = MADGRAD(parameters, momentum=momentum, decoupled_decay=True, **opt_args)
    elif opt_lower == 'novograd' or opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = RMSprop(parameters, alpha=0.9, momentum=momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=momentum, **opt_args)

    # second order
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)

    # NVIDIA fused optimizers, require APEX to be installed
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)

    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer

