import ast
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "train_halfcheetah_imagination_rl.py"


def get_main_function():
    module = ast.parse(SCRIPT_PATH.read_text())

    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node

    raise AssertionError("main() not found")


def get_function(name):
    module = ast.parse(SCRIPT_PATH.read_text())

    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node

    raise AssertionError(f"{name}() not found")


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


def test_main_compiles_all_static_paths_by_default_on_cuda():
    main_fn = get_main_function()

    arg_names = [arg.arg for arg in main_fn.args.args]
    defaults = main_fn.args.defaults
    default_offset = len(arg_names) - len(defaults)
    default_by_name = {
        name: ast.literal_eval(default)
        for name, default in zip(arg_names[default_offset:], defaults)
    }

    assert default_by_name["compile"] is None
    assert default_by_name["compile_dynamic"] is False

    assignments = {
        target.id: ast.unparse(node.value)
        for node in ast.walk(main_fn)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    assert assignments["compile"] == "device.type == 'cuda'"
    assert assignments["compile_world_model"] == "compile or compile_world_model"
    assert assignments["compile_generate"] == "compile or compile_generate"
    assert assignments["compile_learn"] == "compile or compile_learn"
    assert assignments["static_compile_shapes"] == "compile_dynamic is False"

    calls = {
        node.func.id: keyword_map(node)
        for node in ast.walk(main_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("train_world_model", "train_agent_in_imagination")
    }

    assert ast.unparse(calls["train_world_model"]["static_batch_shape"]) == "compile_world_model and static_compile_shapes"
    assert ast.unparse(calls["train_agent_in_imagination"]["static_generate_shape"]) == "compile_generate and static_compile_shapes"


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

    assert ast.literal_eval(reward_range_assign.value) == (-4.0, 4.0)

    value_range_assign = next(
        node
        for node in ast.walk(main_fn)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "value_range" for target in node.targets)
    )

    assert ast.literal_eval(value_range_assign.value) == (-10.0, 10.0)

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
    assert ast.literal_eval(reward_kwargs["num_bins"]) == 51
    assert ast.literal_eval(reward_kwargs["sigma_to_bin_ratio"]) == 0.75
    assert ast.literal_eval(reward_kwargs["min_max_value_on_bin_center"]) is True
    assert ast.literal_eval(reward_kwargs["use_symlog"]) is True

    assert isinstance(value_kwargs["reward_range"], ast.Name)
    assert value_kwargs["reward_range"].id == "value_range"
    assert ast.literal_eval(value_kwargs["num_bins"]) == 51
    assert ast.literal_eval(value_kwargs["sigma_to_bin_ratio"]) == 0.75
    assert ast.literal_eval(value_kwargs["min_max_value_on_bin_center"]) is True
    assert ast.literal_eval(value_kwargs["use_symlog"]) is True


def test_static_generate_shape_requires_prompt_windows():
    train_fn = get_function("train_agent_in_imagination")

    guards = [
        ast.unparse(node.test)
        for node in ast.walk(train_fn)
        if isinstance(node, ast.If)
        and any(isinstance(child, ast.Raise) for child in node.body)
    ]

    assert any(
        "static_generate_shape" in guard
        and "prompt_length > 0" in guard
        and "not exists(prompt_iterator)" in guard
        for guard in guards
    )


def test_static_world_model_shape_repeats_short_batches():
    train_fn = get_function("train_world_model")

    static_shape_branch = next(
        node
        for node in ast.walk(train_fn)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "static_batch_shape"
    )

    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "repeat_batch_to_size"
        for node in ast.walk(static_shape_branch)
    )
