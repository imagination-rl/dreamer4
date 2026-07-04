import ast
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "train_halfcheetah_imagination_rl.py"


def get_main_function():
    module = ast.parse(SCRIPT_PATH.read_text())

    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node

    raise AssertionError("main() not found")


def keyword_map(call):
    return {
        keyword.arg: keyword.value
        for keyword in call.keywords
        if keyword.arg is not None
    }


def test_main_default_attention_shape_matches_model_dim():
    main_fn = get_main_function()

    arg_names = [arg.arg for arg in main_fn.args.args]
    defaults = main_fn.args.defaults
    default_offset = len(arg_names) - len(defaults)
    default_by_name = {
        name: ast.literal_eval(default)
        for name, default in zip(arg_names[default_offset:], defaults)
    }

    assert default_by_name["attn_heads"] * default_by_name["attn_dim_head"] == default_by_name["model_dim"]


def test_main_forwards_custom_attention_values_to_world_model():
    main_fn = get_main_function()

    world_model_call = next(
        node.value.func.value
        for node in ast.walk(main_fn)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "world_model" for target in node.targets)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "to"
        and isinstance(node.value.func.value, ast.Call)
        and isinstance(node.value.func.value.func, ast.Name)
        and node.value.func.value.func.id == "DynamicsWorldModel"
    )

    kwargs = keyword_map(world_model_call)

    assert isinstance(kwargs["attn_heads"], ast.Name)
    assert kwargs["attn_heads"].id == "attn_heads"

    assert isinstance(kwargs["attn_dim_head"], ast.Name)
    assert kwargs["attn_dim_head"].id == "attn_dim_head"


def test_main_uses_symlog_hl_gauss_reward_and_return_ranges():
    main_fn = get_main_function()

    reward_range_assign = next(
        node
        for node in ast.walk(main_fn)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "reward_range" for target in node.targets)
    )

    assert ast.literal_eval(reward_range_assign.value) == (-3.0, 3.0)

    value_range_assign = next(
        node
        for node in ast.walk(main_fn)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "value_range" for target in node.targets)
    )

    assert ast.literal_eval(value_range_assign.value) == (-9.90353755128617, 9.90353755128617)

    world_model_call = next(
        node.value.func.value
        for node in ast.walk(main_fn)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "world_model" for target in node.targets)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "to"
        and isinstance(node.value.func.value, ast.Call)
        and isinstance(node.value.func.value.func, ast.Name)
        and node.value.func.value.func.id == "DynamicsWorldModel"
    )

    kwargs = keyword_map(world_model_call)
    reward_kwargs = keyword_map(kwargs["reward_encoder_kwargs"])
    value_kwargs = keyword_map(kwargs["value_encoder_kwargs"])

    assert isinstance(reward_kwargs["reward_range"], ast.Name)
    assert reward_kwargs["reward_range"].id == "reward_range"
    assert ast.literal_eval(reward_kwargs["sigma_to_bin_ratio"]) == 0.75
    assert ast.literal_eval(reward_kwargs["min_max_value_on_bin_center"]) is True
    assert ast.literal_eval(reward_kwargs["use_symlog"]) is True

    assert isinstance(value_kwargs["reward_range"], ast.Name)
    assert value_kwargs["reward_range"].id == "value_range"
    assert ast.literal_eval(value_kwargs["num_bins"]) == 511
    assert ast.literal_eval(value_kwargs["sigma_to_bin_ratio"]) == 0.75
    assert ast.literal_eval(value_kwargs["min_max_value_on_bin_center"]) is True
    assert ast.literal_eval(value_kwargs["use_symlog"]) is True
