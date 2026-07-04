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
