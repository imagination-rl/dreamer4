import pytest
param = pytest.mark.parametrize
import torch
from contextlib import nullcontext
from einops import rearrange

def exists(v):
    return v is not None

E2E_CASES = (
    pytest.param({}, id = 'default'),
    pytest.param(
        dict(
            pred_orig_latent = True,
            grouped_query_attn = True,
            num_spatial_tokens = 8,
            dynamics_with_video_input = True,
            add_task_embeds = True,
        ),
        id = 'video-task-grouped',
    ),
    pytest.param(
        dict(
            prob_shortcut_train = 1.,
            signal_and_step_passed_in = True,
            use_time_cache = True,
            var_len = True,
        ),
        id = 'shortcut-cache-variable-length',
    ),
    pytest.param(
        dict(
            prob_shortcut_train = 0.,
            with_latent_ar_in_dynamics = True,
            with_latent_ar_action_conditioned = True,
            condition_on_actions = True,
        ),
        id = 'latent-ar-action-conditioned',
    ),
    pytest.param(
        dict(
            add_reward_embed_to_agent_token = True,
            add_state_pred_head = True,
            predict_terminals = True,
            time_attention_use_pope = True,
        ),
        id = 'agent-heads-and-pope',
    ),
)


@param('case', E2E_CASES)
def test_e2e(case):
    settings = dict(
        pred_orig_latent = False,
        grouped_query_attn = False,
        dynamics_with_video_input = False,
        prob_shortcut_train = None,
        add_task_embeds = False,
        num_spatial_tokens = 2,
        signal_and_step_passed_in = False,
        condition_on_actions = False,
        add_reward_embed_to_agent_token = False,
        add_state_pred_head = False,
        use_time_cache = False,
        var_len = False,
        time_attention_use_pope = False,
        with_latent_ar_in_dynamics = False,
        with_latent_ar_action_conditioned = False,
        predict_terminals = False,
    )
    settings.update(case)
    (
        pred_orig_latent,
        grouped_query_attn,
        dynamics_with_video_input,
        prob_shortcut_train,
        add_task_embeds,
        num_spatial_tokens,
        signal_and_step_passed_in,
        condition_on_actions,
        add_reward_embed_to_agent_token,
        add_state_pred_head,
        use_time_cache,
        var_len,
        time_attention_use_pope,
        with_latent_ar_in_dynamics,
        with_latent_ar_action_conditioned,
        predict_terminals,
    ) = (settings[key] for key in settings)
    from dreamer4.dreamer4 import VideoTokenizer, DynamicsWorldModel

    tokenizer = VideoTokenizer(
        16,
        encoder_depth = 1,
        decoder_depth = 1,
        time_block_every = 1,
        dim_latent = 16,
        patch_size = 16,
        attn_dim_head = 8,
        num_latent_tokens = 4,
        encoder_add_decorr_aux_loss = True,
        decorr_sample_frac = 1.,
        time_attention_use_pope = time_attention_use_pope
    )

    video = torch.randn(2, 3, 4, 32, 32)

    loss = tokenizer(video)
    assert loss.numel() == 1

    latents = tokenizer(video, return_latents = True)
    assert latents.shape[-1] == 16

    recon = tokenizer.decode(latents, 32, 32)
    assert recon.shape == video.shape

    query_heads, heads = (16, 4) if grouped_query_attn else (8, 8)

    dynamics = DynamicsWorldModel(
        dim = 16,
        video_tokenizer = tokenizer,
        dim_latent = 16,
        max_steps = 64,
        num_tasks = 2,
        num_latent_tokens = 4,
        depth = 1,
        num_spatial_tokens = num_spatial_tokens,
        pred_orig_latent = pred_orig_latent,
        num_discrete_actions = 4,
        attn_dim_head = 8,
        attn_heads = heads,
        attn_kwargs = dict(
            query_heads = query_heads,
        ),
        prob_shortcut_train = prob_shortcut_train,
        add_reward_embed_to_agent_token = add_reward_embed_to_agent_token,
        add_state_pred_head = add_state_pred_head,
        agent_predicts_state = add_state_pred_head,
        time_block_every = 1,
        time_attention_use_pope = time_attention_use_pope,
        latent_ar = with_latent_ar_in_dynamics,
        latent_ar_action_conditioned = with_latent_ar_action_conditioned,
        latent_ar_layer = 0 if with_latent_ar_in_dynamics else None,
        latent_ar_loss_weight = 1. if with_latent_ar_in_dynamics else 0.,
        latent_ar_sigreg_loss_weight = 0.05 if with_latent_ar_in_dynamics else 0.,
        latent_ar_sigreg_loss_kwargs = dict(num_slices = 2) if with_latent_ar_in_dynamics else None,
        predict_terminals = predict_terminals
    )

    signal_levels = step_sizes_log2 = None

    if signal_and_step_passed_in:
        signal_levels = torch.randint(0, 32, (2, 4))
        step_sizes_log2 = torch.randint(1, 5, (2,))

    if dynamics_with_video_input:
        dynamics_input = dict(video = video)
    else:
        dynamics_input = dict(latents = latents)

    tasks = None
    if add_task_embeds:
        tasks = torch.randint(0, 2, (2,))

    actions = None
    if condition_on_actions:
        actions = torch.randint(0, 4, (2, 3, 1))

    lens = None
    if var_len:
        lens = torch.randint(1, 4, (2,))

    terminals = None
    if predict_terminals:
        terminals = (torch.randn(2) > 0.).bool()

    flow_loss = dynamics(
        **dynamics_input,
        lens = lens,
        tasks = tasks,
        terminals = terminals,
        signal_levels = signal_levels,
        step_sizes_log2 = step_sizes_log2,
        discrete_actions = actions,
        add_autoregressive_action_loss = True
    )

    assert flow_loss.numel() == 1

    # generating

    generations = dynamics.generate(
        time_steps = 10,
        image_height = 32,
        image_width = 32,
        batch_size = 2,
        return_rewards_per_frame = True,
        return_terminals = predict_terminals,
        use_time_cache = use_time_cache
    )

    video_batch, video_channels, video_time, video_height, video_width = generations.video.shape
    assert video_batch == 2
    assert video_channels == 3
    assert video_time <= 10
    assert (video_height, video_width) == (32, 32)

    reward_batch, reward_time = generations.rewards.shape
    assert reward_batch == 2
    assert reward_time <= 10

    # rl

    rewards = torch.randn((2, 4)) * 100.

    flow_loss = dynamics(
        **dynamics_input,
        tasks = tasks,
        rewards = rewards
    )

def test_symexp_two_hot():
    import torch
    from dreamer4.dreamer4 import SymExpTwoHot

    two_hot_encoder = SymExpTwoHot(
        (-3., 3.),
        num_bins = 20,
        learned_embedding = True,
        dim_embed = 512
    )

    values = torch.randn((10))

    two_hot_encoded = two_hot_encoder(values)
    recon_values = two_hot_encoder.bins_to_scalar_value(two_hot_encoded, normalize = False)

    assert torch.allclose(recon_values, values, atol = 1e-6)

    reward_embeds = two_hot_encoder.embed(two_hot_encoded)
    assert reward_embeds.shape == (10, 512)

def test_hl_gauss_reward_encoder():
    import torch
    from dreamer4.dreamer4 import HLGaussRewardEncoder

    encoder = HLGaussRewardEncoder(
        (-3., 3.),
        num_bins = 20,
        learned_embedding = True,
        dim_embed = 512
    )

    values = torch.randn((10)).clamp(-3., 3.)

    probs = encoder(values)
    recon_values = encoder.bins_to_scalar_value(probs, normalize = False)
    empty_probs = encoder(values.new_empty(2, 0))

    assert probs.shape == (10, 20)
    assert empty_probs.shape == (2, 0, 20)
    assert recon_values.shape == values.shape
    assert recon_values.amin() >= -3.
    assert recon_values.amax() <= 3.

    reward_embeds = encoder.embed(probs)
    assert reward_embeds.shape == (10, 512)


def test_world_model_reward_mae_combines_reward_heads_and_masks_invalid_targets():
    from dreamer4.dreamer4 import DynamicsWorldModel

    model = DynamicsWorldModel(
        dim = 8,
        dim_latent = 4,
        max_steps = 8,
        num_latent_tokens = 1,
        num_spatial_tokens = 1,
        depth = 1,
        time_block_every = 1,
        attn_heads = 1,
        attn_dim_head = 8,
        num_discrete_actions = 0,
        num_continuous_actions = 1,
        continuous_dist_type = 'beta',
        continuous_target_action_range = (-1., 1.),
        multi_token_pred_len = 2,
        reward_encoder_kwargs = dict(num_bins = 5),
        value_encoder_kwargs = dict(num_bins = 5),
        policy_head_mlp_depth = 1,
        value_head_mlp_depth = 1,
    )

    model.reward_encoder.bins_to_scalar_value = lambda logits, normalize = True: logits.new_zeros(logits.shape[:-1])

    _, losses = model(
        latents = torch.randn(2, 4, 1, 4),
        rewards = torch.tensor(((0., 1., 2., 3.), (0., 4., 5., 100.))),
        continuous_actions = torch.zeros(2, 4, 1),
        lens = torch.tensor((4, 2)),
        return_all_losses = True,
    )

    latent_reward_mae = torch.tensor((1., 2., 3., 4., 5.)).mean()
    agent_reward_mae = torch.tensor((1., 2., 2., 3., 3., 4., 5., 5.)).mean()
    expected_reward_mae = (latent_reward_mae + agent_reward_mae) * 0.5

    assert torch.allclose(losses.reward_mae, expected_reward_mae)
    assert not losses.reward_mae.requires_grad

def test_dynamics_world_model_defaults_to_hl_gauss_reward_encoder():
    from dreamer4.dreamer4 import DynamicsWorldModel, HLGaussRewardEncoder, SymExpTwoHot, SymExpValueTwoHot, TDMPC2ValueTwoHot

    dynamics = DynamicsWorldModel(
        dim = 16,
        dim_latent = 8,
        num_latent_tokens = 4,
        depth = 1,
        reward_encoder_kwargs = dict(num_bins = 20),
        value_encoder_kwargs = dict(num_bins = 32),
    )

    assert isinstance(dynamics.reward_encoder, HLGaussRewardEncoder)
    assert isinstance(dynamics.value_encoder, HLGaussRewardEncoder)
    assert dynamics.reward_encoder.num_bins == 20
    assert dynamics.value_encoder.num_bins == 32

    dynamics = DynamicsWorldModel(
        dim = 16,
        dim_latent = 8,
        num_latent_tokens = 4,
        depth = 1,
        reward_encoder_type = 'symexp_two_hot',
    )

    assert isinstance(dynamics.reward_encoder, SymExpTwoHot)
    assert type(dynamics.value_encoder) is SymExpTwoHot

    dynamics = DynamicsWorldModel(
        dim = 16,
        dim_latent = 8,
        num_latent_tokens = 4,
        depth = 1,
        reward_encoder_type = 'hl_gauss',
        use_symexp_value_head = True,
        reward_encoder_kwargs = dict(num_bins = 20),
        value_encoder_kwargs = dict(reward_range = (-10., 10.), num_bins = 255),
    )

    assert isinstance(dynamics.reward_encoder, HLGaussRewardEncoder)
    assert isinstance(dynamics.value_encoder, SymExpValueTwoHot)
    assert dynamics.value_encoder.bin_values[0].item() < -22000.
    assert dynamics.value_encoder.bin_values[-1].item() > 22000.

    value_logits = dynamics.value_head(torch.randn(3, 16))
    values = dynamics.value_encoder.bins_to_scalar_value(value_logits)

    assert torch.equal(values, torch.zeros_like(values))

    with pytest.raises(ValueError, match = 'non-finite bins'):
        SymExpValueTwoHot((-1200., 1200.))

    dynamics = DynamicsWorldModel(
        dim = 16,
        dim_latent = 8,
        num_latent_tokens = 4,
        depth = 1,
        reward_encoder_type = 'hl_gauss',
        use_tdmpc2_value_head = True,
        reward_encoder_kwargs = dict(num_bins = 20),
        value_encoder_kwargs = dict(reward_range = (-10., 10.), num_bins = 101),
    )

    assert isinstance(dynamics.reward_encoder, HLGaussRewardEncoder)
    assert isinstance(dynamics.value_encoder, TDMPC2ValueTwoHot)
    assert dynamics.value_encoder.bin_values[0].item() < -22000.
    assert dynamics.value_encoder.bin_values[-1].item() > 22000.

    value_logits = dynamics.value_head(torch.randn(3, 16))
    values = dynamics.value_encoder.bins_to_scalar_value(value_logits)

    assert torch.equal(values, torch.zeros_like(values))


def test_tdmpc2_value_two_hot_uses_symlog_targets_and_inverse_expected_symlog():
    from dreamer4.dreamer4 import TDMPC2ValueTwoHot, symexp

    encoder = TDMPC2ValueTwoHot(reward_range = (-10., 10.), num_bins = 101)
    targets = torch.tensor([-1200., 0., 1200.])
    probs = encoder(targets)
    decoded = encoder.bins_to_scalar_value(probs, normalize = False)

    torch.testing.assert_close(decoded, targets, rtol = 1e-5, atol = 1e-4)

    probs = torch.zeros(1, 101)
    probs[0, 50] = 0.99
    probs[0, -1] = 0.01
    decoded = encoder.bins_to_scalar_value(probs, normalize = False)

    torch.testing.assert_close(decoded, symexp(torch.tensor([0.1])))

    with pytest.raises(ValueError, match = 'non-finite bins'):
        TDMPC2ValueTwoHot(reward_range = (-1200., 1200.), num_bins = 101)


def test_special_value_head_encoders_are_mutually_exclusive():
    with pytest.raises(ValueError, match = 'mutually exclusive'):
        _small_flag_dynamics(
            use_symexp_value_head = True,
            use_tdmpc2_value_head = True,
        )

    with pytest.raises(ValueError, match = 'mutually exclusive'):
        _small_flag_dynamics(
            use_symexp_value_head = True,
            use_symlog_hl_gauss_value_head = True,
        )


def test_symlog_hl_gauss_value_head_is_smooth_raw_decoded_and_zero_initialized():
    from dreamer4.dreamer4 import HLGaussRewardEncoder

    dynamics = _small_flag_dynamics(
        use_symlog_hl_gauss_value_head = True,
        value_encoder_kwargs = dict(
            reward_range = (-10., 10.),
            num_bins = 7,
            sigma_to_bin_ratio = 0.75,
            min_max_value_on_bin_center = False,
            use_symlog = True,
            decode_symlog_expected_raw = True,
        ),
    )

    assert isinstance(dynamics.value_encoder, HLGaussRewardEncoder)
    assert dynamics.value_encoder.use_symlog is True
    assert dynamics.value_encoder.decode_symlog_expected_raw is True

    logits = dynamics.value_head(torch.randn(3, 16))
    values = dynamics.value_encoder.bins_to_scalar_value(logits)

    assert torch.equal(logits, torch.zeros_like(logits))
    assert torch.equal(values, torch.zeros_like(values))


def test_hl_gauss_actor_critic_is_opt_in_and_owned_by_agent_optimizer():
    from dreamer4.dreamer4 import Actions, Experience, HLGaussRewardEncoder, SymExpValueTwoHot

    dynamics = _small_flag_dynamics(
        use_symexp_value_head = True,
        use_hl_gauss_actor_critic = True,
        value_encoder_kwargs = dict(reward_range = (-10., 10.), num_bins = 7),
        hl_gauss_actor_critic_kwargs = dict(
            reward_range = (-1200., 1200.),
            num_bins = 7,
            sigma_to_bin_ratio = 0.75,
            min_max_value_on_bin_center = False,
        ),
        num_continuous_actions = 2,
        pmpo_kl_div_loss_weight = 0.,
    )

    assert isinstance(dynamics.value_encoder, SymExpValueTwoHot)
    assert isinstance(dynamics.actor_critic_value_encoder, HLGaussRewardEncoder)
    assert dynamics.actor_critic_value_head is not None
    assert set(dynamics.actor_critic_value_head.parameters()) <= set(dynamics.agent_parameters())

    batch, time = 2, 4
    agent_embed = torch.randn(batch, time, 16)
    with torch.no_grad():
        actor_critic_logits = dynamics.actor_critic_value_head(agent_embed)
        old_values = dynamics.actor_critic_value_encoder.bins_to_scalar_value(actor_critic_logits)

    experience = Experience(
        latents = torch.randn(batch, time, 2, 8),
        agent_embed = agent_embed,
        rewards = torch.randn(batch, time),
        actions = Actions(None, torch.rand(batch, time, 2)),
        log_probs = Actions(None, torch.zeros(batch, time, 2)),
        values = old_values,
        step_size = 1,
        lens = torch.full((batch,), time, dtype = torch.long),
        is_truncated = torch.ones(batch, dtype = torch.bool),
        terminals = torch.zeros(batch, dtype = torch.bool),
        is_from_world_model = True,
    )
    _, value_loss, diagnostics = dynamics.learn_from_experience(
        experience,
        objective = 'pmpo',
        return_diagnostics = True,
    )

    assert {
        'extra/actor_critic_value_abs_error',
        'extra/critic_advantage_sign_disagreement',
    } <= diagnostics.keys()

    value_loss.backward()
    assert any(param.grad is not None for param in dynamics.value_head.parameters())
    assert any(param.grad is not None for param in dynamics.actor_critic_value_head.parameters())

    with pytest.raises(ValueError, match = 'requires `use_symexp_value_head`'):
        _small_flag_dynamics(use_hl_gauss_actor_critic = True)


def _small_flag_dynamics(**kwargs):
    from dreamer4.dreamer4 import DynamicsWorldModel

    value_encoder_kwargs = kwargs.pop('value_encoder_kwargs', dict(num_bins = 7))

    return DynamicsWorldModel(
        dim = 16,
        dim_latent = 8,
        num_latent_tokens = 2,
        num_spatial_tokens = 4,
        num_register_tokens = 2,
        depth = 1,
        time_block_every = 1,
        max_steps = 8,
        reward_encoder_kwargs = dict(num_bins = 7),
        value_encoder_kwargs = value_encoder_kwargs,
        **kwargs
    )


def test_opt_in_policy_all_tokens_uses_ordered_flattened_context_and_false_is_unchanged():
    from dreamer4.dreamer4 import DynamicsWorldModel

    current = _small_flag_dynamics(policy_all_tokens = False)
    old = _small_flag_dynamics(policy_all_tokens = True)

    assert current.policy_head.layers[0][0].in_features == 16
    assert old.policy_num_tokens == 7
    assert old.policy_head.layers[0][0].in_features == 7 * 16

    latents = torch.randn(1, 3, 2, 8)
    signal_levels = torch.zeros(1, 3, dtype = torch.long)
    step_sizes_log2 = torch.zeros(1, dtype = torch.long)
    _, (embeds, _) = old(
        latents = latents,
        signal_levels = signal_levels,
        step_sizes_log2 = step_sizes_log2,
        return_pred_only = True,
        return_intermediates = True
    )
    assert embeds.actor.shape == (1, 3, 7 * 16)


def test_old_behavior_flags_leave_model_default_state_unchanged_when_disabled():
    torch.manual_seed(123)
    implicit = _small_flag_dynamics()

    torch.manual_seed(123)
    explicit = _small_flag_dynamics(
        add_action_embed_to_agent_token = False,
        old_reward_shifting = False,
        old_pmpo = False,
        old_reward_head = False,
        old_terminal_head = False,
        old_dynamics_ff = False,
        old_policy_head = False,
        old_value_head = False,
        old_embedding_init = False,
        use_symexp_value_head = False,
        use_tdmpc2_value_head = False,
        use_symlog_hl_gauss_value_head = False,
        use_hl_gauss_actor_critic = False,
        use_baseline_symexp_value_head_init = False,
        value_head_init_value = None,
        final_special_cross_attn = None,
    )

    implicit_state = implicit.state_dict()
    explicit_state = explicit.state_dict()
    assert implicit_state.keys() == explicit_state.keys()
    assert all(torch.equal(implicit_state[key], explicit_state[key]) for key in implicit_state)


def test_cleanrl_ppo_closer_keeps_model_parameters_and_uses_scalar_value_loss():
    from dreamer4.dreamer4 import Actions, Experience

    torch.manual_seed(123)
    baseline = _small_flag_dynamics(
        num_continuous_actions = 2,
        continuous_dist_type = 'beta',
    )
    torch.manual_seed(123)
    closer = _small_flag_dynamics(
        num_continuous_actions = 2,
        continuous_dist_type = 'beta',
        cleanrl_ppo_closer = True,
        clip_values = True,
        value_clip = .2,
        policy_entropy_weight = 0.,
    )

    baseline_parameters = dict(baseline.named_parameters())
    closer_parameters = dict(closer.named_parameters())
    assert baseline_parameters.keys() == closer_parameters.keys()
    assert {
        name: parameter.shape
        for name, parameter in baseline_parameters.items()
    } == {
        name: parameter.shape
        for name, parameter in closer_parameters.items()
    }
    assert sum(parameter.numel() for parameter in baseline_parameters.values()) == sum(
        parameter.numel() for parameter in closer_parameters.values()
    )

    batch = 4
    agent_embed = torch.randn(batch, 1, 16)
    with torch.no_grad():
        predicted_values = closer.value_encoder.bins_to_scalar_value(
            closer.value_head(agent_embed)
        )

    old_values = predicted_values.detach() + torch.tensor((-.5, -.1, .3, .6)).reshape(batch, 1)
    returns = torch.tensor((1., -1., 2., -2.)).reshape(batch, 1)
    clipped_values = old_values + (predicted_values - old_values).clamp(-.2, .2)
    expected_value_loss = .5 * torch.maximum(
        (predicted_values - returns).square(),
        (clipped_values - returns).square(),
    ).mean()
    experience = Experience(
        latents = torch.randn(batch, 1, 2, 8),
        agent_embed = agent_embed,
        rewards = torch.zeros(batch, 1),
        actions = Actions(None, torch.rand(batch, 1, 2)),
        log_probs = Actions(None, torch.zeros(batch, 1, 2)),
        values = old_values,
        frozen_returns = returns,
        frozen_advantages = torch.zeros(batch, 1),
        step_size = 1,
        lens = torch.ones(batch, dtype = torch.long),
        is_truncated = torch.zeros(batch, dtype = torch.bool),
        terminals = torch.zeros(batch, dtype = torch.bool),
        is_from_world_model = True,
    )

    _, value_loss, diagnostics = closer.learn_from_experience(
        experience,
        objective = 'ppo',
        return_diagnostics = True,
    )

    assert value_loss.item() == pytest.approx(expected_value_loss.item())
    assert diagnostics['extra/value_mse'] == pytest.approx(
        (predicted_values - returns).square().mean().item()
    )
    assert 'extra/value_target_entropy' not in diagnostics


def test_cleanrl_ppo_closer_hl_gauss_value_loss_uses_unclipped_cross_entropy():
    from dreamer4.dreamer4 import Actions, Experience

    closer = _small_flag_dynamics(
        num_continuous_actions = 2,
        continuous_dist_type = 'beta',
        cleanrl_ppo_closer = True,
        cleanrl_ppo_closer_hl_gauss_value_loss = True,
        clip_values = True,
        value_clip = .2,
        policy_entropy_weight = 0.,
        use_symlog_hl_gauss_value_head = True,
        value_encoder_kwargs = dict(
            reward_range = (-10., 10.),
            num_bins = 7,
            use_symlog = True,
            decode_symlog_expected_raw = True,
        ),
    )

    batch = 4
    agent_embed = torch.randn(batch, 1, 16)
    returns = torch.tensor((1., -1., 20., -20.)).reshape(batch, 1)

    with torch.no_grad():
        value_logits = closer.value_head(agent_embed)
        predicted_values = closer.value_encoder.bins_to_scalar_value(value_logits)
        expected_value_loss = -(closer.value_encoder(returns) * value_logits.log_softmax(dim = -1)).sum(dim = -1).mean()

    # old_values far from the prediction would change a clipped loss.
    experience = Experience(
        latents = torch.randn(batch, 1, 2, 8),
        agent_embed = agent_embed,
        rewards = torch.zeros(batch, 1),
        actions = Actions(None, torch.rand(batch, 1, 2)),
        log_probs = Actions(None, torch.zeros(batch, 1, 2)),
        values = predicted_values + 5.,
        frozen_returns = returns,
        frozen_advantages = torch.zeros(batch, 1),
        step_size = 1,
        lens = torch.ones(batch, dtype = torch.long),
        is_truncated = torch.zeros(batch, dtype = torch.bool),
        terminals = torch.zeros(batch, dtype = torch.bool),
        is_from_world_model = True,
    )

    _, value_loss, diagnostics = closer.learn_from_experience(
        experience,
        objective = 'ppo',
        return_diagnostics = True,
    )

    assert value_loss.item() == pytest.approx(expected_value_loss.item(), rel = 1e-5)
    assert diagnostics['extra/value_mse'] == pytest.approx((predicted_values - returns).square().mean().item(), rel = 1e-5)
    assert 'extra/value_target_entropy' in diagnostics

    with pytest.raises(ValueError, match = 'requires `cleanrl_ppo_closer`'):
        _small_flag_dynamics(cleanrl_ppo_closer_hl_gauss_value_loss = True)


def _paper_shortcut_dynamics(**kwargs):
    from dreamer4.dreamer4 import DynamicsWorldModel

    return DynamicsWorldModel(
        dim = 8,
        dim_latent = 4,
        num_latent_tokens = 1,
        num_spatial_tokens = 1,
        num_register_tokens = 1,
        depth = 1,
        time_block_every = 1,
        attn_heads = 1,
        attn_dim_head = 8,
        max_steps = 8,
        reward_encoder_kwargs = dict(num_bins = 5),
        value_encoder_kwargs = dict(num_bins = 5),
        **kwargs,
    )


def test_paper_shortcut_flag_uses_piecewise_per_example_objective():
    torch.manual_seed(123)
    model = _paper_shortcut_dynamics(
        use_paper_aligned_shortcut_objective = True,
    )
    latents = torch.randn(16, 3, 1, 4)

    loss, losses = model(
        latents = latents,
        shortcut_train_mode = 'combined',
        return_all_losses = True,
        seed = 42,
    )

    diagnostics = losses.diagnostics

    assert diagnostics.raw_flow > 0.
    assert diagnostics.raw_shortcut > 0.
    assert torch.equal(
        diagnostics.raw_dynamics,
        diagnostics.raw_flow + diagnostics.raw_shortcut,
    )
    assert torch.equal(diagnostics.raw_dynamics, diagnostics.normalized_dynamics)

    _, finetune_losses = model(
        latents = latents,
        shortcut_train_mode = 'finetune',
        return_all_losses = True,
        seed = 42,
    )
    _, bootstrap_losses = model(
        latents = latents,
        shortcut_train_mode = 'bootstrap_only',
        return_all_losses = True,
        seed = 42,
    )

    assert finetune_losses.diagnostics.raw_flow > 0.
    assert finetune_losses.diagnostics.raw_shortcut == 0.
    assert bootstrap_losses.diagnostics.raw_flow == 0.
    assert bootstrap_losses.diagnostics.raw_shortcut > 0.

    loss.backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_paper_shortcut_flag_normalizes_one_piecewise_dynamics_loss():
    torch.manual_seed(123)
    model = _paper_shortcut_dynamics(
        use_loss_normalization = True,
        use_paper_aligned_shortcut_objective = True,
    )

    _, losses = model(
        latents = torch.randn(16, 3, 1, 4),
        shortcut_train_mode = 'combined',
        return_all_losses = True,
        seed = 42,
    )

    assert losses.diagnostics.raw_flow > 0.
    assert losses.diagnostics.raw_shortcut > 0.
    assert model.paper_dynamics_loss_normalizer.exp_avg_sq.item() != 1.
    assert model.flow_loss_normalizer.exp_avg_sq.item() == 1.
    assert model.shortcut_flow_loss_normalizer.exp_avg_sq.item() == 1.


def test_paper_shortcut_flag_accepts_explicit_inference_step():
    model = _paper_shortcut_dynamics(
        use_paper_aligned_shortcut_objective = True,
    )

    loss = model(
        latents = torch.randn(2, 3, 1, 4),
        signal_levels = torch.zeros(2, 3, dtype = torch.long),
        step_sizes_log2 = torch.zeros(2, dtype = torch.long),
    )

    assert torch.isfinite(loss)


def test_opt_in_old_heads_shift_and_embedding_initializers():
    old = _small_flag_dynamics(
        old_reward_head = True,
        old_reward_shifting = True,
        old_terminal_head = True,
        old_policy_head = True,
        old_value_head = True,
        old_dynamics_ff = True,
        old_embedding_init = True,
        value_head_init_value = 0.,
        predict_terminals = True,
    )

    assert old.to_agent_embed_reward_pred.net[1].bias is None
    assert old.to_agent_embed_terminal_pred is None
    assert old.to_latent_state_terminal_pred.layers[0][-1].__class__.__name__ == 'ReluSquared'
    assert old.embedding_init_eps == 1e-2
    assert old.transformer.layers[0][3].fn.use_glu is False
    assert old.transformer.layers[0][3].fn.activation.__class__.__name__ == 'ReluSquared'
    assert old.policy_head.layers[0][-1].__class__.__name__ == 'ReluSquared'
    assert old.value_head.layers[0][-1].__class__.__name__ == 'ReluSquared'

    # The configured run initializes the old value head at zero.
    values = old.value_encoder.bins_to_scalar_value(old.value_head(torch.randn(2, 16)))
    assert torch.allclose(values, torch.zeros_like(values), atol = 1e-5)

    # The final reward is bootstrap padding in the old alignment and must not
    # influence the post-transition reward objective.
    latents = torch.randn(1, 4, 2, 8)
    reward_prefix = torch.tensor([[1., 2., 3.]])

    def reward_loss_with_padding(padding):
        torch.manual_seed(0)
        _, losses = old(
            latents = latents,
            rewards = torch.cat((reward_prefix, torch.tensor([[padding]])), dim = 1),
            lens = torch.tensor([4]),
            return_all_losses = True,
        )
        return losses.agent_embed_reward

    assert torch.allclose(reward_loss_with_padding(-100.), reward_loss_with_padding(100.))


def test_opt_in_action_conditioning_and_final_special_cross_attention():
    from dreamer4.dreamer4 import DynamicsWorldModel

    no_action = _small_flag_dynamics(num_discrete_actions = 4)
    with_action = _small_flag_dynamics(num_discrete_actions = 4, add_action_embed_to_agent_token = True)
    assert with_action.action_embed_to_agent_token_scale.requires_grad
    assert torch.equal(with_action.action_embed_to_agent_token_scale.detach(), torch.ones(()))
    assert no_action.action_embed_to_agent_token_scale is None

    disabled = _small_flag_dynamics(final_special_cross_attn = False)
    default = _small_flag_dynamics()
    assert disabled.transformer.should_special_cross_attend is False
    assert default.transformer.should_special_cross_attend is True

@pytest.mark.skipif(not torch.cuda.is_available(), reason = 'no cuda')
@param('causal', (False, True))
@param('softclamp_value', (50., None))
@param('num_agent_tokens', (0, 1))
@param('causal_block_size', (1, 8))
@param('block_size_per_special', (1, 8))
@param('special_attend_only_itself', (False, True))
def test_attend_factory(
    causal,
    softclamp_value,
    num_agent_tokens,
    causal_block_size,
    block_size_per_special,
    special_attend_only_itself
):

    from dreamer4.dreamer4 import get_attend_fn

    q = torch.randn(1, 8, 1024, 512).cuda()
    k = torch.randn(1, 4, 1024, 512).cuda()
    v = torch.randn(1, 4, 1024, 512).cuda()

    attend_kwargs = dict(
        seq_len = 1024,
        k_seq_len = 1024,
        causal = causal,
        causal_block_size = causal_block_size,
        softclamp_value = softclamp_value,
        device = q.device,
        num_special_tokens = num_agent_tokens,
        block_size_per_special = block_size_per_special,
        special_attend_only_itself = special_attend_only_itself
    )

    attend = get_attend_fn(True, **attend_kwargs)
    flex_out = attend(q, k, v)

    attend = get_attend_fn(False, **attend_kwargs)
    out = attend(q, k, v)

    assert torch.allclose(flex_out, out, atol = 1e-6)


def test_flex_attention_backend_is_forwarded_in_kernel_options(monkeypatch):
    from dreamer4 import dreamer4 as dreamer_module

    captured_kwargs = {}

    def recording_flex_attention(q, k, v, **kwargs):
        captured_kwargs.update(kwargs)
        return q

    monkeypatch.setattr(dreamer_module, 'flex_attention', recording_flex_attention)
    q = torch.randn(1, 1, 4, 8)
    attend = dreamer_module.get_attend_fn(
        use_flex = True,
        seq_len = 4,
        k_seq_len = 4,
        flex_attention_backend = 'TRITON',
        device = q.device,
    )

    assert torch.equal(attend(q, q, q), q)
    assert captured_kwargs['kernel_options'] == {'BACKEND': 'TRITON'}


def test_flex_attention_auto_avoids_invalid_flex_decode_cuda_grid(monkeypatch):
    from dreamer4 import dreamer4 as dreamer_module

    selected_backends = []

    def recording_flex_attention(q, k, v, **kwargs):
        selected_backends.append(kwargs['kernel_options']['BACKEND'])
        return q

    monkeypatch.setattr(dreamer_module, 'flex_attention', recording_flex_attention)
    attend = dreamer_module.get_attend_fn(
        use_flex = True,
        seq_len = 1,
        k_seq_len = 1,
        flex_attention_backend = 'AUTO',
        device = torch.device('cpu'),
    )

    # FlexDecode uses grid.y = flattened_batch * kv_heads. The largest valid
    # CUDA Y grid stays AUTO, while one more batch-head pair must use regular
    # Triton attention.
    safe_qkv = torch.empty(16383, 4, 1, 1)
    unsafe_qkv = torch.empty(16384, 4, 1, 1)
    attend(safe_qkv, safe_qkv, safe_qkv)
    attend(unsafe_qkv, unsafe_qkv, unsafe_qkv)

    assert selected_backends == ['AUTO', 'TRITON']


@pytest.mark.skipif(not torch.cuda.is_available(), reason = 'no cuda')
def test_flex_attention_triton_backend_matches_auto():
    from dreamer4.dreamer4 import get_attend_fn

    device = torch.device('cuda')
    q = torch.randn(2, 4, 64, 32, device = device)
    k = torch.randn(2, 4, 64, 32, device = device)
    v = torch.randn(2, 4, 64, 32, device = device)

    attend_kwargs = dict(
        use_flex = True,
        seq_len = 64,
        k_seq_len = 64,
        causal = False,
        num_special_tokens = 1,
        device = device,
    )

    auto_out = get_attend_fn(**attend_kwargs)(q, k, v)
    triton_out = get_attend_fn(
        **attend_kwargs,
        flex_attention_backend = 'TRITON'
    )(q, k, v)

    assert torch.allclose(triton_out, auto_out, atol = 1e-5, rtol = 1e-5)


def test_axial_transformer_uses_configured_flex_backend_for_space_only(monkeypatch):
    from dreamer4 import dreamer4 as dreamer_module

    attend_calls = []
    original_get_attend_fn = dreamer_module.get_attend_fn

    def recording_get_attend_fn(*args, **kwargs):
        attend_calls.append((kwargs['causal'], kwargs['flex_attention_backend']))
        return original_get_attend_fn(*args, **kwargs)

    monkeypatch.setattr(dreamer_module, 'get_attend_fn', recording_get_attend_fn)
    transformer = dreamer_module.AxialSpaceTimeTransformer(
        dim = 8,
        depth = 1,
        attn_heads = 1,
        attn_dim_head = 8,
        time_block_every = 2,
        num_special_tokens = 1,
        use_attn_pool = False,
        value_residual = False,
        spatial_flex_attention_backend = 'TRITON',
    )

    transformer(torch.randn(1, 2, 3, 8))

    assert attend_calls == [(False, 'TRITON'), (True, 'AUTO')]

def test_action_with_world_model():
    from dreamer4.dreamer4 import VideoTokenizer, DynamicsWorldModel

    tokenizer = VideoTokenizer(
        512,
        dim_latent = 8,
        patch_size = 16,
        encoder_depth = 1,
        decoder_depth = 1,
        time_block_every = 1,
        attn_heads = 2,
        image_height = 32,
        image_width = 32,
        attn_kwargs = dict(
            query_heads = 16
        )
    )

    dynamics = DynamicsWorldModel(
        512,
        num_agents = 1,
        video_tokenizer = tokenizer,
        dim_latent = 8,
        depth = 1,
        time_block_every = 1,
        num_discrete_actions = 4,
        reward_encoder_kwargs = dict(num_bins = 20),
        value_encoder_kwargs = dict(num_bins = 32)
    )

    rewards = torch.randn(1, 4)
    discrete_actions = torch.randint(0, 4, (1, 4, 1))

    gen = dynamics.generate(
        16,
        batch_size = 4,
        return_rewards_per_frame = True,
        return_agent_actions = True,
        return_log_probs_and_values = True
    )

    assert gen.video.shape == (4, 3, 16, 32, 32)
    assert gen.rewards.shape == (4, 16)

    discrete_actions, continuous_actions = gen.actions

    assert discrete_actions.shape == (4, 16, 1)
    assert continuous_actions is None or continuous_actions.numel() == 0

    discrete_log_probs, _ = gen.log_probs

    assert discrete_log_probs.shape == (4, 16, 1)
    assert gen.values.shape == (4, 16)

    # take a reinforcement learning step

    actor_loss, critic_loss, diagnostics = dynamics.learn_from_experience(gen, return_diagnostics = True)

    assert {
        'critic/gae_return_mean',
        'critic/gae_return_min',
        'critic/gae_return_max',
        'critic/gae_return_abs_max',
        'critic/value_mean',
        'critic/value_min',
        'critic/value_max',
        'critic/value_abs_error',
        'critic/value_target_saturation_frac',
        'extra/positive_advantage_frac',
        'extra/negative_advantage_frac',
        'extra/advantage_mean',
        'extra/advantage_abs_mean',
        'extra/policy_entropy',
        'extra/importance_ratio_clip_frac',
        'extra/value_target_entropy',
        'extra/value_excess_cross_entropy'
    } <= diagnostics.keys()

    assert torch.isfinite(diagnostics['extra/advantage_mean'])
    assert torch.isfinite(diagnostics['extra/advantage_abs_mean'])
    assert torch.isfinite(diagnostics['extra/policy_entropy'])
    assert torch.isfinite(diagnostics['extra/value_target_entropy'])
    assert torch.isfinite(diagnostics['extra/value_excess_cross_entropy'])
    assert diagnostics['extra/importance_ratio_clip_frac'].item() == 0.

    actor_loss.backward(retain_graph = True)
    critic_loss.backward()


@param('action_kind, has_proprio, use_time_cache, autocast_dtype, policy_all_tokens', (
    ('continuous', False, False, None, False),
    ('discrete', True, True, torch.bfloat16, True),
))
def test_generate_preallocated_outputs_match_eager_contract(
    action_kind,
    has_proprio,
    use_time_cache,
    autocast_dtype,
    policy_all_tokens,
):
    from dreamer4.dreamer4 import DynamicsWorldModel

    use_continuous_actions = action_kind == 'continuous'

    dynamics = DynamicsWorldModel(
        dim = 16,
        dim_latent = 8,
        max_steps = 8,
        num_latent_tokens = 2,
        num_spatial_tokens = 2,
        depth = 1,
        time_block_every = 1,
        dim_proprio = 3 if has_proprio else None,
        num_discrete_actions = 4 if not use_continuous_actions else 0,
        num_continuous_actions = 2 if use_continuous_actions else 0,
        continuous_dist_type = "beta",
        policy_all_tokens = policy_all_tokens,
        reward_encoder_kwargs = dict(num_bins = 11),
        value_encoder_kwargs = dict(num_bins = 11),
    )

    prompt_latents = torch.randn(2, 2, 2, 8).clamp(-1., 1.)
    prompt_proprio = torch.randn(2, 2, 3) if has_proprio else None
    prompt_discrete_actions = torch.randint(0, 4, (2, 1, 1)) if not use_continuous_actions else None
    prompt_continuous_actions = torch.randn(2, 1, 2).sigmoid() if use_continuous_actions else None
    prompt_rewards = torch.randn(2, 1)

    generate_kwargs = dict(
        time_steps = 5,
        num_steps = 2,
        batch_size = 2,
        return_decoded_video = False,
        return_rewards_per_frame = True,
        return_agent_actions = True,
        return_log_probs_and_values = True,
        return_terminals = False,
        use_time_cache = use_time_cache,
        prompt_latents = prompt_latents,
        prompt_proprio = prompt_proprio,
        prompt_discrete_actions = prompt_discrete_actions,
        prompt_continuous_actions = prompt_continuous_actions,
        prompt_rewards = prompt_rewards,
    )
    generate_kwargs = {key: value for key, value in generate_kwargs.items() if value is not None}

    def assert_matching_tensor(preallocated_tensor, eager_tensor):
        assert exists(preallocated_tensor) == exists(eager_tensor)

        if not exists(preallocated_tensor):
            return

        assert preallocated_tensor.dtype == eager_tensor.dtype

        if preallocated_tensor.is_floating_point():
            assert torch.allclose(preallocated_tensor, eager_tensor)
        else:
            assert torch.equal(preallocated_tensor, eager_tensor)

    def assert_preallocated_matches_eager(**kwargs):
        autocast = torch.autocast('cpu', dtype = autocast_dtype) if exists(autocast_dtype) else nullcontext()

        with torch.no_grad(), autocast:
            torch.manual_seed(123)
            eager = dynamics.generate(**kwargs)

            torch.manual_seed(123)
            preallocated = dynamics.generate(**kwargs, preallocate_outputs = True)

        for preallocated_tensor, eager_tensor in (
            (preallocated.latents, eager.latents),
            (preallocated.proprio, eager.proprio),
            (preallocated.rewards, eager.rewards),
            (preallocated.agent_embed, eager.agent_embed),
            (preallocated.policy_input, eager.policy_input),
            (preallocated.actions.discrete, eager.actions.discrete),
            (preallocated.actions.continuous, eager.actions.continuous),
            (preallocated.log_probs.discrete, eager.log_probs.discrete),
            (preallocated.log_probs.continuous, eager.log_probs.continuous),
            (preallocated.values, eager.values),
        ):
            assert_matching_tensor(preallocated_tensor, eager_tensor)

    assert_preallocated_matches_eager(**generate_kwargs)

    no_prompt_generate_kwargs = dict(generate_kwargs)
    for key in (
        'prompt_latents',
        'prompt_proprio',
        'prompt_discrete_actions',
        'prompt_continuous_actions',
        'prompt_rewards',
    ):
        no_prompt_generate_kwargs.pop(key, None)

    assert_preallocated_matches_eager(**no_prompt_generate_kwargs)

    no_generate_kwargs = dict(generate_kwargs, time_steps = prompt_latents.shape[1])

    with torch.no_grad():
        eager = dynamics.generate(**no_generate_kwargs)
        preallocated = dynamics.generate(**no_generate_kwargs, preallocate_outputs = True)

    assert preallocated.agent_embed is eager.agent_embed is None
    assert preallocated.policy_input is eager.policy_input is None
    assert preallocated.values is eager.values is None
    assert_matching_tensor(preallocated.actions.discrete, eager.actions.discrete)
    assert_matching_tensor(preallocated.actions.continuous, eager.actions.continuous)

def test_action_embedder():
    from dreamer4.dreamer4 import ActionEmbedder

    # 1 discrete action with 4 choices

    embedder = ActionEmbedder(
        512,
        num_discrete_actions = 4
    )

    actions = torch.randint(0, 4, (2, 3, 1))
    action_embed = embedder(discrete_actions = actions)

    assert action_embed.shape == (2, 3, 512)

    # 2 discrete actions with 4 choices each

    embedder = ActionEmbedder(
        512,
        num_discrete_actions = (4, 4)
    )

    actions = torch.randint(0, 4, (2, 3, 2))
    action_embed = embedder(discrete_actions = actions)

    assert action_embed.shape == (2, 3, 512)

    # picking out only the second discrete action

    actions = torch.randint(0, 4, (2, 3, 1))
    action_embed = embedder(discrete_actions = actions, discrete_action_types = 1)

    assert action_embed.shape == (2, 3, 512)

    # 2 continuous actions

    embedder = ActionEmbedder(
        512,
        num_continuous_actions = 2,
        continuous_norm_stats = ((0., 2.), (1., 1.)) # (mean, std) for normalizing each action
    )

    actions = torch.randn((2, 3, 2))
    action_embed = embedder(continuous_actions = actions)

    assert action_embed.shape == (2, 3, 512)

    # 2 discrete actions with 4 choices each and 2 continuous actions

    embedder = ActionEmbedder(
        512,
        num_discrete_actions = (4, 4),
        num_continuous_actions = 2
    )

    discrete_actions = torch.randint(0, 4, (2, 3, 2))
    continuous_actions = torch.randn(2, 3, 2)

    action_embed = embedder(discrete_actions = discrete_actions, continuous_actions = continuous_actions)
    assert action_embed.shape == (2, 3, 512)

    # picking out one discrete and one continuous

    discrete_actions = torch.randint(0, 4, (2, 3, 1))
    continuous_actions = torch.randn(2, 3, 1)

    action_embed = embedder(discrete_actions = discrete_actions, continuous_actions = continuous_actions, discrete_action_types = 1, continuous_action_types = 0)

    assert action_embed.shape == (2, 3, 512)

    # unembed

    embedder = ActionEmbedder(
        512,
        num_discrete_actions = (4, 4),
        num_continuous_actions = 2,
        can_unembed = True
    )

    discrete_actions = torch.randint(0, 4, (2, 3, 2))
    continuous_actions = torch.randn(2, 3, 2)

    action_embed = embedder(discrete_actions = discrete_actions, continuous_actions = continuous_actions)

    discrete_logits, continuous_mean_log_var = embedder.unembed(action_embed)

    assert discrete_logits.shape == (2, 3, 8)
    assert continuous_mean_log_var.shape == (2, 3, 2, 2)

    # scalar prediction-head selection should match the first unsqueezed head

    multi_pred_embedder = ActionEmbedder(
        512,
        num_discrete_actions = (4, 4),
        num_continuous_actions = 2,
        can_unembed = True,
        num_unembed_preds = 2,
        squeeze_unembed_preds = False
    )

    all_discrete_logits, all_continuous_mean_log_var = multi_pred_embedder.unembed(action_embed)
    head_discrete_logits, head_continuous_mean_log_var = multi_pred_embedder.unembed(action_embed, pred_head_index = 0)

    assert torch.allclose(head_discrete_logits, all_discrete_logits[0], atol = 1e-6)
    assert torch.allclose(head_continuous_mean_log_var, all_continuous_mean_log_var[0], atol = 1e-6)

    # test kl div

    discrete_logits_tgt, continuous_mean_log_var_tgt = embedder.unembed(action_embed)

    discrete_kl_div, continuous_kl_div = embedder.kl_div((discrete_logits, continuous_mean_log_var), (discrete_logits_tgt, continuous_mean_log_var_tgt))

    assert discrete_kl_div.shape == (2, 3)
    assert continuous_kl_div.shape == (2, 3)

    # return discrete split by number of actions

    discrete_logits, continuous_mean_log_var = embedder.unembed(action_embed, return_split_discrete = True)
    assert discrete_logits[0].shape == discrete_logits[1].shape == (2, 3, 4)

    # unembed subset of actions

    discrete_logits, continuous_mean_log_var = embedder.unembed(action_embed, discrete_action_types = 1, continuous_action_types = 0)

    assert discrete_logits.shape == (2, 3, 4)
    assert continuous_mean_log_var.shape == (2, 3, 1, 2)

    # sample actions

    sampled_discrete_actions, sampled_continuous_actions = embedder.sample(action_embed, discrete_action_types = 1, continuous_action_types = 0)

    assert sampled_discrete_actions.shape == (2, 3, 1)
    assert sampled_continuous_actions.shape == (2, 3, 1)

    # rescale bounded continuous actions to the environment range

    beta_embedder = ActionEmbedder(
        512,
        num_continuous_actions = 3,
        continuous_dist_type = 'beta',
        continuous_target_action_range = (-1., 1.),
        can_unembed = True
    )

    native_actions = torch.tensor([0., 0.5, 1.])
    env_actions = beta_embedder.rescale_for_env(native_actions)

    assert torch.allclose(env_actions, torch.tensor([-1., 0., 1.]))

    # log probs

    assert discrete_logits.shape == (2, 3, 4)
    assert continuous_mean_log_var.shape == (2, 3, 1, 2)

    discrete_log_probs, continuous_log_probs = embedder.log_probs(
        action_embed,
        discrete_targets = discrete_actions,
        continuous_targets = continuous_actions,
        parallel_discrete_calc = False
    )

    assert discrete_log_probs.shape == (2, 3, 2)
    assert continuous_log_probs.shape == (2, 3, 2)

    _, (discrete_entropies, continuous_entropies) = embedder.log_probs(
        action_embed,
        discrete_targets = discrete_actions,
        continuous_targets = continuous_actions,
        parallel_discrete_calc = True,
        return_entropies = True
    )

    assert discrete_entropies.shape == (2, 3, 2)
    assert continuous_entropies.shape == (2, 3, 2)

    parallel_discrete_log_probs, _ = embedder.log_probs(
        action_embed,
        discrete_targets = discrete_actions,
        continuous_targets = continuous_actions,
        parallel_discrete_calc = True
    )

    assert torch.allclose(discrete_log_probs, parallel_discrete_log_probs, atol = 1e-5)

    # 0 discrete actions

    embedder = ActionEmbedder(
        512,
        num_discrete_actions = 0,
        num_continuous_actions = 2,
        can_unembed = True
    )

    continuous_actions = torch.randn(2, 3, 2)
    action_embed = embedder(continuous_actions = continuous_actions)

    assert action_embed.shape == (2, 3, 512)

    discrete_logits, continuous_mean_log_var = embedder.unembed(action_embed)

    assert discrete_logits is None
    assert continuous_mean_log_var.shape == (2, 3, 2, 2)

    sampled_discrete, sampled_continuous = embedder.sample(action_embed)

    assert sampled_discrete is None
    assert sampled_continuous.shape == (2, 3, 2)


def test_opt_in_legacy_continuous_action_encoder_matches_historical_projection():
    from dreamer4.dreamer4 import ActionEmbedder

    dim = 4
    num_actions = 3
    embedder = ActionEmbedder(
        dim,
        num_continuous_actions = num_actions,
        continuous_dist_type = 'beta',
        continuous_target_action_range = (-1., 1.),
        old_continuous_action_encoder = True,
    )

    assert isinstance(embedder.continuous_action_embed, torch.nn.Linear)
    assert embedder.continuous_action_embed.bias is None
    assert embedder.continuous_action_embed.weight.shape == (dim, num_actions)
    assert embedder.continuous_action_embed_scale.requires_grad
    assert torch.equal(embedder.continuous_action_embed_scale.detach(), torch.ones(()))

    # The old projection used a fan-in/fan-out normal initialization.  Use a
    # moderately wide layer so the sample standard deviation is stable.
    wide = ActionEmbedder(
        64,
        num_continuous_actions = 16,
        old_continuous_action_encoder = True,
    )
    expected_std = (64 * 16) ** -0.5
    observed_std = wide.continuous_action_embed.weight.detach().std().item()
    assert observed_std == pytest.approx(expected_std, rel = 0.15)

    with torch.no_grad():
        embedder.continuous_action_embed.weight.copy_(torch.arange(dim * num_actions, dtype = torch.float).reshape(dim, num_actions))
        embedder.continuous_action_embed_scale.fill_(1.5)

    native_actions = torch.tensor(
        (
            ((0.25, 0.5, 0.75), (0.1, 0.4, 0.9)),
            ((0.2, 0.6, 0.8), (0.3, 0.5, 0.7)),
        ),
        requires_grad = True,
    )
    expected_actions = native_actions * 2. - 1.
    expected = torch.nn.functional.linear(
        expected_actions,
        embedder.continuous_action_embed.weight,
    ) * embedder.continuous_action_embed_scale
    actual = embedder(continuous_actions = native_actions)
    assert actual.shape == (2, 2, dim)
    assert torch.allclose(actual, expected)

    loss = actual.square().sum()
    loss.backward()
    assert native_actions.grad is not None
    assert embedder.continuous_action_embed.weight.grad is not None
    assert embedder.continuous_action_embed_scale.grad is not None
    assert embedder.continuous_action_embed_scale in embedder.embed_parameters()
    assert embedder.continuous_action_embed.weight in embedder.embed_parameters()


def test_legacy_continuous_action_encoder_world_model_grouping_and_state_dict_migration():
    from dreamer4.dreamer4 import ActionEmbedder, DynamicsWorldModel

    # The modern mode has exactly the historical state-dict keys for its
    # embedding; opting in adds only the legacy scalar and changes the linear
    # weight orientation.
    modern = ActionEmbedder(8, num_continuous_actions = 2)
    legacy = ActionEmbedder(8, num_continuous_actions = 2, old_continuous_action_encoder = True)
    assert 'continuous_action_embed_scale' not in modern.state_dict()
    assert 'continuous_action_embed_scale' in legacy.state_dict()

    modern_state = modern.state_dict()
    legacy.load_state_dict(modern_state, strict = True)
    assert torch.equal(legacy.continuous_action_embed.weight, modern_state['continuous_action_embed.weight'].T)
    assert torch.equal(legacy.continuous_action_embed_scale.detach(), torch.ones(()))

    # A legacy checkpoint already contains the scale key, so square weights
    # must not be mistaken for a modern embedding and transposed on load.
    square_legacy = ActionEmbedder(2, num_continuous_actions = 2, old_continuous_action_encoder = True)
    with torch.no_grad():
        square_legacy.continuous_action_embed.weight.copy_(torch.tensor(((1., 2.), (3., 4.))))
    square_state = square_legacy.state_dict()
    square_reloaded = ActionEmbedder(2, num_continuous_actions = 2, old_continuous_action_encoder = True)
    square_reloaded.load_state_dict(square_state, strict = True)
    assert torch.equal(square_reloaded.continuous_action_embed.weight, square_state['continuous_action_embed.weight'])

    model = DynamicsWorldModel(
        dim = 16,
        dim_latent = 8,
        num_latent_tokens = 2,
        num_spatial_tokens = 2,
        num_register_tokens = 1,
        depth = 1,
        time_block_every = 1,
        max_steps = 8,
        num_continuous_actions = 2,
        old_continuous_action_encoder = True,
        reward_encoder_kwargs = dict(num_bins = 7),
        value_encoder_kwargs = dict(num_bins = 7),
    )
    scale = model.action_embedder.continuous_action_embed_scale
    world_params = model.world_model_parameters()
    agent_params = model.agent_parameters()
    assert any(parameter is scale for parameter in world_params)
    assert all(parameter is not scale for parameter in agent_params)


def test_modern_continuous_action_encoder_remains_embedding_by_default():
    from dreamer4.dreamer4 import ActionEmbedder

    embedder = ActionEmbedder(3, num_continuous_actions = 2)
    assert embedder.old_continuous_action_encoder is False
    assert isinstance(embedder.continuous_action_embed, torch.nn.Embedding)
    assert embedder.continuous_action_embed_scale is None

    # Even zero-action models retain the empty modern embedding parameter, as
    # they did before the legacy mode was introduced.
    zero_actions = ActionEmbedder(3, num_continuous_actions = 0)
    assert 'continuous_action_embed.weight' in zero_actions.state_dict()

    with torch.no_grad():
        embedder.continuous_action_embed.weight.copy_(torch.tensor(((1., 2., 3.), (4., 5., 6.))))

    actions = torch.tensor([[[2., 3.]]])
    expected = torch.tensor([[[14., 19., 24.]]])
    assert torch.allclose(embedder(continuous_actions = actions), expected)

def test_mtp():
    from dreamer4.dreamer4 import create_multi_token_prediction_targets

    rewards = torch.randn(3, 16) # (b t)

    reward_targets, mask = create_multi_token_prediction_targets(rewards, 3) # say three token lookahead

    assert reward_targets.shape == (3, 16, 3)
    assert mask.shape == (3, 16, 3)

    actions = torch.randint(0, 10, (3, 16, 2))
    action_targets, mask = create_multi_token_prediction_targets(actions, 3)

    assert action_targets.shape == (3, 16, 3, 2)
    assert mask.shape == (3, 16, 3)

    from dreamer4.dreamer4 import ActionEmbedder

    embedder = ActionEmbedder(
        512,
        num_discrete_actions = (4, 4),
        num_continuous_actions = 2,
        can_unembed = True,
        num_unembed_preds = 8
    )

    discrete_actions = torch.randint(0, 4, (2, 3, 2))
    continuous_actions = torch.randn(2, 3, 2)

    action_embed = torch.randn(2, 16, 512)
    discrete_logits, continuous_logits = embedder.unembed(action_embed)

    assert discrete_logits.shape == (8, 2, 16, 8)

    discrete_logits, continuous_logits = embedder.unembed(action_embed, pred_head_index = 0)

    assert discrete_logits.shape == (2, 16, 8)

def test_loss_normalizer():
    from torch import cat
    from dreamer4.dreamer4 import LossNormalizer

    loss_normalizer = LossNormalizer(4, beta = 0.)

    losses = torch.rand(4)

    _ = loss_normalizer(losses)
    normed_losses = loss_normalizer(losses)

    assert (normed_losses == 1.).all()


def test_loss_normalizer_preserves_scalar_shape_and_validates_before_update():
    from dreamer4.dreamer4 import LossNormalizer

    scalar_normalizer = LossNormalizer()
    scalar_loss = torch.tensor(2.)

    normalized = scalar_normalizer(scalar_loss)

    assert normalized.shape == scalar_loss.shape

    vector_normalizer = LossNormalizer(2)
    initial_stats = vector_normalizer.exp_avg_sq.clone()

    with pytest.raises(ValueError, match = "expected 2 loss values"):
        vector_normalizer(torch.tensor(1.))

    assert torch.equal(vector_normalizer.exp_avg_sq, initial_stats)


def test_world_model_loss_normalization_skips_inactive_losses():
    from dreamer4.dreamer4 import DynamicsWorldModel

    model = DynamicsWorldModel(
        dim = 8,
        dim_latent = 4,
        max_steps = 8,
        num_latent_tokens = 1,
        num_spatial_tokens = 1,
        depth = 1,
        time_block_every = 1,
        attn_heads = 1,
        attn_dim_head = 8,
        num_discrete_actions = 0,
        num_continuous_actions = 1,
        continuous_dist_type = "beta",
        continuous_target_action_range = (-1., 1.),
        continuous_action_loss_weight = 0.,
        discrete_action_loss_weight = 0.,
        multi_token_pred_len = 2,
        reward_encoder_kwargs = dict(num_bins = 5),
        value_encoder_kwargs = dict(num_bins = 5),
        policy_head_mlp_depth = 1,
        value_head_mlp_depth = 1,
        old_reward_head = True,
        old_reward_shifting = True,
        use_loss_normalization = True,
    )

    shortcut_stats = model.shortcut_flow_loss_normalizer.exp_avg_sq.clone()
    latent_reward_stats = model.latent_state_reward_loss_normalizer.exp_avg_sq.clone()
    action_stats = model.continuous_actions_loss_normalizer.exp_avg_sq.clone()

    total_loss, losses = model(
        latents = torch.randn(2, 4, 1, 4),
        rewards = torch.randn(2, 4),
        terminals = torch.zeros(2, dtype = torch.bool),
        continuous_actions = torch.rand(2, 4, 1),
        lens = torch.tensor((4, 3)),
        return_all_losses = True,
        shortcut_train_mode = "finetune",
    )

    assert total_loss.ndim == 0
    assert losses.continuous_actions.ndim == 0
    assert torch.equal(model.shortcut_flow_loss_normalizer.exp_avg_sq, shortcut_stats)
    assert torch.equal(model.latent_state_reward_loss_normalizer.exp_avg_sq, latent_reward_stats)
    assert torch.equal(model.continuous_actions_loss_normalizer.exp_avg_sq, action_stats)


def test_world_model_loss_normalization_skips_inactive_legacy_terminal_head():
    from dreamer4.dreamer4 import DynamicsWorldModel

    model = DynamicsWorldModel(
        dim = 8,
        dim_latent = 4,
        max_steps = 8,
        num_latent_tokens = 1,
        num_spatial_tokens = 1,
        depth = 1,
        time_block_every = 1,
        attn_heads = 1,
        attn_dim_head = 8,
        num_discrete_actions = 0,
        num_continuous_actions = 1,
        continuous_dist_type = "beta",
        continuous_action_loss_weight = 0.,
        discrete_action_loss_weight = 0.,
        multi_token_pred_len = 2,
        reward_encoder_kwargs = dict(num_bins = 5),
        value_encoder_kwargs = dict(num_bins = 5),
        policy_head_mlp_depth = 1,
        value_head_mlp_depth = 1,
        old_terminal_head = True,
        use_loss_normalization = True,
    )

    terminal_stats = model.agent_embed_terminal_loss_normalizer.exp_avg_sq.clone()

    total_loss, losses = model(
        latents = torch.randn(2, 4, 1, 4),
        rewards = torch.randn(2, 4),
        terminals = torch.zeros(2, dtype = torch.bool),
        continuous_actions = torch.rand(2, 4, 1),
        lens = torch.tensor((4, 3)),
        return_all_losses = True,
        shortcut_train_mode = "finetune",
    )

    assert total_loss.ndim == 0
    assert losses.agent_embed_terminal.ndim == 0
    assert torch.equal(model.agent_embed_terminal_loss_normalizer.exp_avg_sq, terminal_stats)

def test_tokenizer_trainer():
    from dreamer4.trainers import VideoTokenizerTrainer
    from dreamer4.dreamer4 import VideoTokenizer
    from torch.utils.data import Dataset

    class MockDataset(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return dict(
                video = torch.randn(3, 2, 64, 64),
                time_lens = 2
            )

    dataset = MockDataset()

    tokenizer = VideoTokenizer(
        16,
        encoder_depth = 1,
        decoder_depth = 1,
        time_block_every = 1,
        dim_latent = 16,
        patch_size = 32,
        attn_dim_head = 16,
        num_latent_tokens = 4
    )

    trainer = VideoTokenizerTrainer(
        tokenizer,
        dataset = dataset,
        num_train_steps = 1,
        batch_size = 1,
        cpu = True,
        max_grad_norm = 0.5
    )

    trainer()

def test_tokenizer_trainer_with_replay_buffer(tmp_path):
    from dreamer4.trainers import VideoTokenizerTrainer, VideoDatasetFromReplayBuffer
    from dreamer4.dreamer4 import VideoTokenizer, Experience

    time = 4
    batch = 5

    exp = Experience(
        latents = torch.randn(batch, time, 16),
        video = torch.randn(batch, time, 3, 32, 32),
        lens = torch.full((batch,), time)
    )

    buffer = Experience.create_memmap_replay_buffer(
        exp,
        str(tmp_path / 'buffer'),
        max_episodes = 10,
        max_timesteps = time,
        overwrite = True,
        circular = True
    )

    exp.add_to_memmap_buffer(buffer)

    dataset = VideoDatasetFromReplayBuffer(buffer, image_size = 64)

    tokenizer = VideoTokenizer(
        16,
        encoder_depth = 1,
        decoder_depth = 1,
        time_block_every = 1,
        dim_latent = 16,
        patch_size = 16,
        attn_dim_head = 16,
        num_latent_tokens = 4
    )

    trainer = VideoTokenizerTrainer(
        tokenizer,
        dataset = dataset,
        num_train_steps = 1,
        batch_size = 2,
        cpu = True,
        max_grad_norm = 0.5
    )

    trainer()

@param('with_actions', (True, False))
@param('with_rewards', (True, False))
@param('self_flow', (True, False))
def test_bc_trainer(
    with_actions,
    with_rewards,
    self_flow
):
    from dreamer4.trainers import BehaviorCloneTrainer
    from dreamer4.dreamer4 import DynamicsWorldModel, VideoTokenizer

    from torch.utils.data import Dataset

    class MockDataset(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            state = torch.randn(3, 2, 64, 64)

            pkg = dict(video = state)

            if with_actions:
                pkg.update(discrete_actions = torch.randint(0, 4, (2, 1)))

            if with_rewards:
                pkg.update(rewards = torch.randn(2,))

            return pkg

    dataset = MockDataset()

    tokenizer = VideoTokenizer(
        16,
        encoder_depth = 1,
        decoder_depth = 1,
        time_block_every = 1,
        dim_latent = 16,
        patch_size = 32,
        attn_dim_head = 16,
        num_latent_tokens = 1
    )

    world_model = DynamicsWorldModel(
        video_tokenizer = tokenizer,
        dim = 16,
        dim_latent = 16,
        max_steps = 64,
        num_tasks = 4,
        num_latent_tokens = 1,
        depth = 4,
        time_block_every = 1,
        num_spatial_tokens = 1,
        pred_orig_latent = True,
        num_discrete_actions = 4,
        attn_dim_head = 16,
        prob_shortcut_train = 0.9,
    )

    trainer = BehaviorCloneTrainer(
        world_model,
        dataset = dataset,
        batch_size = 1,
        num_train_steps = 1,
        log_video = False,
        cpu = True,
        self_flow = self_flow,
        self_flow_student_layer = -3,
        self_flow_layer = -1,
        self_flow_loss_weight = 1.0
    )

    trainer()

def test_dream_trainer():
    from dreamer4.dreamer4 import DynamicsWorldModel

    world_model = DynamicsWorldModel(
        dim = 16,
        dim_latent = 16,
        max_steps = 64,
        num_tasks = 4,
        num_latent_tokens = 1,
        depth = 4,
        time_block_every = 1,
        num_spatial_tokens = 1,
        pred_orig_latent = True,
        num_discrete_actions = 4,
        attn_dim_head = 16,
        prob_shortcut_train = 0.9
    )

    # training from self-generations (dreams)

    from dreamer4.trainers import DreamTrainer

    dream_trainer = DreamTrainer(
        world_model,
        batch_size = 2,
        num_train_steps = 1,
        cpu = True,
    )

    dream_trainer()

def test_cache_generate():
    from dreamer4.dreamer4 import DynamicsWorldModel

    dynamics = DynamicsWorldModel(
        dim = 16,
        dim_latent = 16,
        max_steps = 64,
        num_tasks = 4,
        num_latent_tokens = 4,
        depth = 1,
        time_block_every = 1,
        num_spatial_tokens = 1,
        pred_orig_latent = True,
        num_discrete_actions = 4,
        attn_dim_head = 16,
        prob_shortcut_train = 0.9
    )

    generated, time_cache = dynamics.generate(1, return_time_cache = True)
    generated, time_cache = dynamics.generate(1, time_cache = time_cache, return_time_cache = True)
    generated, time_cache = dynamics.generate(1, time_cache = time_cache, return_time_cache = True)

@param('vectorized', (False, True))
@param('objective', ('ppo', 'pmpo', 'spo'))
@param('env_can_terminate', (False, True))
@param('env_can_truncate', (False, True))
@param('store_agent_embed', (False, True))
@param('predict_terminals', (False, True))
@param('actor_spr', (False, True))
def test_online_rl(
    vectorized,
    objective,
    env_can_terminate,
    env_can_truncate,
    store_agent_embed,
    predict_terminals,
    actor_spr
):
    from dreamer4.dreamer4 import DynamicsWorldModel, VideoTokenizer

    tokenizer = VideoTokenizer(
        16,
        encoder_depth = 1,
        decoder_depth = 1,
        time_block_every = 1,
        dim_latent = 16,
        patch_size = 32,
        attn_dim_head = 16,
        num_latent_tokens = 1,
        image_height = 32,
        image_width = 32,
    )

    world_model_and_policy = DynamicsWorldModel(
        video_tokenizer = tokenizer,
        dim = 16,
        dim_latent = 16,
        max_steps = 64,
        num_tasks = 4,
        num_latent_tokens = 1,
        depth = 4,
        time_block_every = 1,
        num_spatial_tokens = 1,
        pred_orig_latent = True,
        num_discrete_actions = 4,
        attn_dim_head = 16,
        prob_shortcut_train = 0.9,
        actor_spr = actor_spr
    )

    from dreamer4.mocks import MockEnv
    from dreamer4.dreamer4 import combine_experiences

    mock_env = MockEnv(
        (32, 32),
        vectorized = vectorized,
        num_envs = 4,
        terminate_after_step = 2 if env_can_terminate else None,
        can_truncate = env_can_truncate,
        rand_terminate_prob = 0.1
    )

    # manually

    dream_experience = world_model_and_policy.generate(10, batch_size = 1, store_agent_embed = store_agent_embed, return_terminals = predict_terminals, return_for_policy_optimization = True)

    one_experience = world_model_and_policy.interact_with_env(mock_env, max_timesteps = 8, env_is_vectorized = vectorized, store_agent_embed = store_agent_embed)
    another_experience = world_model_and_policy.interact_with_env(mock_env, max_timesteps = 16, env_is_vectorized = vectorized, store_agent_embed = store_agent_embed)

    combined_experience = combine_experiences([dream_experience, one_experience, another_experience])

    # quick test moving the experience to different devices

    if torch.cuda.is_available():
        combined_experience = combined_experience.to(torch.device('cuda'))
        combined_experience = combined_experience.to(world_model_and_policy.device)

    if store_agent_embed:
        assert exists(combined_experience.agent_embed)

    actor_loss, critic_loss = world_model_and_policy.learn_from_experience(combined_experience, objective = objective)

    actor_loss.backward()
    critic_loss.backward()

    # with trainer

    from dreamer4.trainers import SimTrainer

    trainer = SimTrainer(
        world_model_and_policy,
        batch_size = 4,
        cpu = True
    )

    trainer(mock_env, num_episodes = 2, env_is_vectorized = vectorized)

@param('num_video_views', (1, 2))
def test_proprioception(
    num_video_views
):
    from dreamer4.dreamer4 import VideoTokenizer, DynamicsWorldModel

    tokenizer = VideoTokenizer(
        512,
        dim_latent = 32,
        patch_size = 32,
        encoder_depth = 2,
        decoder_depth = 2,
        time_block_every = 2,
        attn_heads = 8,
        image_height = 256,
        image_width = 256,
        attn_kwargs = dict(
            query_heads = 16
        )
    )

    dynamics = DynamicsWorldModel(
        512,
        num_agents = 1,
        video_tokenizer = tokenizer,
        dim_latent = 8,
        dim_proprio = 21,
        num_tasks = 2,
        num_video_views = num_video_views,
        num_discrete_actions = 4
    )

    if num_video_views > 1:
        video_shape = (2, num_video_views, 3, 4, 32, 32)
    else:
        video_shape = (2, 3, 4, 32, 32)

    video = torch.randn(*video_shape)
    rewards = torch.randn(2, 10)
    proprio = torch.randn(2, 10, 21)
    discrete_actions = torch.randint(0, 4, (2, 10, 1))
    tasks = torch.randint(0, 4, (2,))

    loss = dynamics(
        video = video,
        rewards = rewards,
        tasks = tasks,
        proprio = proprio,
        discrete_actions = discrete_actions
    )

    loss.backward()

    generations = dynamics.generate(
        10,
        batch_size = 2,
        return_decoded_video = True
    )

    assert exists(generations.proprio)
    assert generations.video.shape == video_shape

def test_epo():
    from dreamer4.dreamer4 import VideoTokenizer, DynamicsWorldModel

    tokenizer = VideoTokenizer(
        512,
        dim_latent = 32,
        patch_size = 32,
        encoder_depth = 2,
        decoder_depth = 2,
        time_block_every = 2,
        attn_heads = 8,
        image_height = 256,
        image_width = 256,
        attn_kwargs = dict(
            query_heads = 16
        )
    )

    dynamics = DynamicsWorldModel(
        512,
        num_agents = 1,
        video_tokenizer = tokenizer,
        dim_latent = 32,
        dim_proprio = 21,
        num_tasks = 4,
        num_latent_genes = 16,
        num_discrete_actions = 4
    )

    fitness = torch.randn(16,)
    dynamics.evolve_(fitness)

def test_cache_token_count():
    from dreamer4 import AxialSpaceTimeTransformer

    model = AxialSpaceTimeTransformer(
        dim = 32,
        depth = 4,
        attn_heads = 4,
        attn_dim_head = 8,
        time_block_every = 2,
        num_special_spatial_tokens = 0,
        rnn_time = False,
        value_residual = False,
        use_attn_pool = False
    )

    #first pass: no cache, two timesteps
    x=torch.randn(1,2,3,32)
    _, cache = model(x, return_intermediates=True)
    assert cache.token_count == 2

    #second pass: full context+cache
    x2 = torch.randn(1,3,3,32)
    _, cache2 = model(x2, cache = cache, return_intermediates = True)
    assert cache2.token_count == 3

def test_time_cache_max_lookback_attends_only_to_recent_steps():
    from dreamer4 import AxialSpaceTimeTransformer

    # depth 3 with time_block_every 2 has one time layer, so a cached K/V entry depends on its own frame only.
    # The rotary positions are relative, so a step with a limited cache equals a fresh forward over the last
    # lookback + 1 frames. A position offset from the cache length (not the true step count) fails this.

    kwargs = dict(dim = 32, depth = 3, attn_heads = 2, attn_dim_head = 16, time_block_every = 2, num_special_tokens = 0)
    lookback, steps = 3, 7

    torch.manual_seed(0)
    full = AxialSpaceTimeTransformer(**kwargs).eval()
    limited = AxialSpaceTimeTransformer(**kwargs, time_cache_max_lookback = lookback).eval()
    limited.load_state_dict(full.state_dict())

    tokens = torch.randn(2, steps, 5, 32)
    cache = None

    with torch.no_grad():
        for t in range(steps):
            out, cache = limited(tokens[:, t:t + 1], cache = cache, return_intermediates = True)

            assert cache.token_count == t + 1
            assert cache.next_kv_cache.shape[-2] == min(t + 1, lookback)

            expected = full(tokens[:, max(0, t - lookback):t + 1])[:, -1:]
            assert torch.allclose(out, expected, atol = 1e-5)

def test_small_terminal_head_changes_only_the_terminal_ensemble():
    torch.manual_seed(0)
    default = _small_flag_dynamics(num_continuous_actions = 2)
    small = _small_flag_dynamics(num_continuous_actions = 2, small_terminal_head = True)

    default_shapes = {name: param.shape for name, param in default.named_parameters()}
    small_shapes = {name: param.shape for name, param in small.named_parameters()}

    changed = {name for name in default_shapes if default_shapes[name] != small_shapes.get(name)}
    assert changed and all(name.startswith('to_agent_embed_terminal_pred.') for name in changed)

    def num_params(module):
        return sum(param.numel() for param in module.parameters())

    assert num_params(small.to_agent_embed_terminal_pred) * 10 < num_params(default.to_agent_embed_terminal_pred)

    latents = torch.randn(2, 4, 2, 8)
    loss = small(
        latents = latents,
        continuous_actions = torch.rand(2, 4, 2) * 2 - 1,
        rewards = torch.randn(2, 4),
        terminals = torch.tensor([True, False]),
        signal_levels = torch.randint(0, 8, (2, 4)),
        step_sizes_log2 = torch.randint(1, 3, (2,)),
    )
    loss.backward()

    assert all(param.grad is not None for param in small.to_agent_embed_terminal_pred.parameters())

    with pytest.raises(ValueError, match = 'exclude each other'):
        _small_flag_dynamics(small_terminal_head = True, old_terminal_head = True)

def test_images_to_video_tokenizer():
    import torch
    from dreamer4 import VideoTokenizer, DynamicsWorldModel, AxialSpaceTimeTransformer

    tokenizer = VideoTokenizer(
        dim = 512,
        dim_latent = 32,
        patch_size = 32,
        image_height = 256,
        image_width = 256,
        encoder_add_decorr_aux_loss = True
    )

    images = torch.randn(2, 3, 256, 256)
    loss, (losses, recon_images) = tokenizer(images, return_intermediates = True)
    loss.backward()

    assert images.shape == recon_images.shape

@param('vectorized', (False, True))
def test_interact_with_env_dict_obs(vectorized):
    from dreamer4.dreamer4 import VideoTokenizer, DynamicsWorldModel

    tokenizer = VideoTokenizer(
        512,
        dim_latent=32,
        patch_size=32,
        encoder_depth=2,
        decoder_depth=2,
        time_block_every=2,
        attn_heads=8,
        image_height=256,
        image_width=256,
        attn_kwargs=dict(
            query_heads=16
        )
    )

    dynamics = DynamicsWorldModel(
        512,
        num_agents=1,
        video_tokenizer=tokenizer,
        dim_latent=32,
        dim_proprio=21,
        num_tasks=4,
        num_latent_genes=16,
        num_discrete_actions=4
    )

    from dreamer4.mocks import MockDictEnv

    mock_env = MockDictEnv(
        (256, 256),
        21,
        vectorized = vectorized,
        num_envs = 4 if vectorized else 1,
        terminate_after_step = 3
    )

    experience = dynamics.interact_with_env(
        mock_env,
        max_timesteps = 4,
        env_is_vectorized = vectorized
    )

    assert exists(experience)
    assert experience.proprio.shape[-1] == 21
    assert experience.proprio.shape[1] > 0

@param('prompt_type', ('video', 'image', 'latents'))
@param('with_proprio', (False, True))
@param('with_discrete_actions', (False, True))
@param('with_continuous_actions', (False, True))
@param('with_rewards', (False, True))
def test_prompting_generation(
    prompt_type,
    with_proprio,
    with_discrete_actions,
    with_continuous_actions,
    with_rewards
):
    from einops import rearrange
    from dreamer4.dreamer4 import VideoTokenizer, DynamicsWorldModel

    tokenizer = VideoTokenizer(
        512,
        dim_latent = 32,
        patch_size = 32,
        encoder_depth = 1,
        decoder_depth = 1,
        time_block_every = 1,
        attn_heads = 2,
        image_height = 256,
        image_width = 256,
    )

    dynamics = DynamicsWorldModel(
        512,
        num_agents = 1,
        video_tokenizer = tokenizer,
        dim_latent = 32,
        num_tasks = 1,
        num_discrete_actions = (4, 4) if with_discrete_actions else 0,
        num_continuous_actions = 2 if with_continuous_actions else 0,
        dim_proprio = 21 if with_proprio else None,
        depth = 1,
        time_block_every = 1
    )

    # 1. generate the required prompt components

    prompt_kwargs = dict()

    if prompt_type == 'video':
        prompt_kwargs['prompt'] = torch.randn(2, 3, 1, 256, 256)
    elif prompt_type == 'image':
        prompt_kwargs['prompt'] = torch.randn(2, 3, 256, 256)
    elif prompt_type == 'latents':
        raw_video = torch.randn(2, 3, 1, 256, 256)
        latents = tokenizer.tokenize(raw_video)
        prompt_kwargs['prompt_latents'] = rearrange(latents, 'b t n d -> b t 1 n d')

    if with_proprio:
        prompt_kwargs['prompt_proprio'] = torch.randn(2, 1, 21)

    if with_discrete_actions:
        prompt_kwargs['prompt_discrete_actions'] = torch.randint(0, 4, (2, 1, 2))

    if with_continuous_actions:
        prompt_kwargs['prompt_continuous_actions'] = torch.randn(2, 1, 2)

    if with_rewards:
        prompt_kwargs['prompt_rewards'] = torch.randn(2, 1)

    # 2. execute generation

    generations = dynamics.generate(
        time_steps = 8,
        batch_size = 2,
        image_height = 256,
        image_width = 256,
        return_rewards_per_frame = with_rewards,
        return_agent_actions = (with_discrete_actions or with_continuous_actions),
        return_decoded_video = True,
        **prompt_kwargs
    )

    has_extras = with_rewards or with_proprio or with_discrete_actions or with_continuous_actions

    if has_extras:
        video = generations.video
    else:
        video = generations

    assert video.shape == (2, 3, 8, 256, 256)

    if with_rewards:
        assert generations.rewards.shape == (2, 8)

    if with_proprio:
        assert generations.proprio.shape == (2, 8, 21)

    if with_discrete_actions or with_continuous_actions:
        discrete_actions, continuous_actions = generations.actions

        if with_discrete_actions:
            assert discrete_actions.shape == (2, 8, 2)

        if with_continuous_actions:
            assert continuous_actions.shape == (2, 8, 2)

@param('steps', (1, 2, 4))
def test_rac_like_tokenizer(steps):
    from dreamer4.dreamer4 import VideoTokenizer

    model = VideoTokenizer(
        dim = 32,
        dim_latent = 32,
        patch_size = 4,
        image_height = 32,
        image_width = 32,
        num_latent_tokens = 16,
        encoder_depth = 2,
        decoder_depth = 2,
        time_block_every = 2,
        attn_dim_head = 16,
        attn_heads = 2,
        decoder_flow_steps = steps
    )

    batch = 2
    channels = 3
    time = 2
    height, width = 32, 32

    video = torch.randn(batch, channels, time, height, width)

    loss = model(video)
    assert loss.numel() == 1

    model.eval()
    latents = torch.randn(batch, time, 16, 32)
    with torch.no_grad():
        recon_video = model.decode(latents, height = height, width = width)

    assert recon_video.shape == (batch, channels, time, height, width)

@param('use_time_rnn', (False, True))
@param('use_causal_conv3d', (False, True))
@param('use_shifted_patch_tokenization', (False, True))
def test_e2e_sequential_parallel_cache(use_time_rnn, use_causal_conv3d, use_shifted_patch_tokenization):
    import torch
    from dreamer4.dreamer4 import VideoTokenizer, DynamicsWorldModel

    tokenizer = VideoTokenizer(
        dim = 16,
        dim_latent = 16,
        patch_size = 32,
        image_height = 64,
        image_width = 64,
        num_latent_tokens = 4,
        encoder_depth = 2,
        decoder_depth = 2,
        time_block_every = 2,
        attn_heads = 4,
        attn_dim_head = 16,
        use_causal_conv3d = use_causal_conv3d,
        use_shifted_patch_tokenization = use_shifted_patch_tokenization
    ).eval()

    dynamics = DynamicsWorldModel(
        dim = 32,
        dim_latent = tokenizer.dim_latent,
        num_latent_tokens = tokenizer.num_latent_tokens,
        num_discrete_actions = 2,
        video_tokenizer = tokenizer,
        depth = 1,
        time_block_every = 1,
        policy_head_mlp_depth = 2,
        value_head_mlp_depth = 2,
        attn_heads = 4,
        attn_dim_head = 16,
        use_time_rnn = use_time_rnn
    ).eval()

    video = torch.randn(1, 3, 4, 64, 64)
    discrete_actions = torch.randint(0, 2, (1, 4, 1))

    with torch.no_grad():
        latents_parallel, tokenizer_cache_parallel = tokenizer(
            video,
            return_latents = True,
            return_time_cache = True
        )

        step_sizes = torch.ones((1,))
        signal_levels = torch.full((1, 4), dynamics.max_steps - 1)

        _, (embeds_parallel, _) = dynamics(
            latents = latents_parallel,
            signal_levels = signal_levels,
            step_sizes = step_sizes,
            discrete_actions = discrete_actions,
            return_pred_only = True,
            return_intermediates = True,
            latent_is_noised = True
        )

        policy_embed_parallel = dynamics.policy_head(embeds_parallel.agent)

        tokenizer_cache_sequential = dynamics_cache_sequential = None
        policy_embed_sequential = []

        for frame_idx in range(4):
            latents_sequential, tokenizer_cache_sequential = tokenizer(
                video[:, :, frame_idx],
                return_latents = True,
                time_cache = tokenizer_cache_sequential,
                return_time_cache = True
            )
            seq_action = None if frame_idx == 0 else discrete_actions[:, frame_idx-1:frame_idx]

            _, (intermediates_sequential, dynamics_cache_sequential) = dynamics(
                latents = latents_sequential,
                signal_levels = signal_levels[:, frame_idx:frame_idx+1],
                step_sizes = step_sizes,
                discrete_actions = seq_action,
                time_cache = dynamics_cache_sequential,
                return_pred_only = True,
                return_intermediates = True,
                latent_is_noised = True
            )

            policy_embed_sequential.append(dynamics.policy_head(intermediates_sequential.agent))

        policy_embed_sequential = torch.cat(policy_embed_sequential, dim = 1)

    assert torch.allclose(policy_embed_parallel, policy_embed_sequential, atol = 1e-4)

@param('use_tokenizer', (False, True))
@param('use_aux_encoder', (False, True))
def test_aux_encoder_combinations(use_tokenizer, use_aux_encoder):
    if not (use_tokenizer or use_aux_encoder):
        return

    import torch
    from torch import nn
    from dreamer4.dreamer4 import DynamicsWorldModel, VideoTokenizer
    from dreamer4.mocks import MockEnv

    class MockAuxEncoder(nn.Module):
        def forward(self, x, **kwargs):
            return torch.randn(x.shape[0], x.shape[2], 2, 16, device = x.device)

    both = use_tokenizer and use_aux_encoder
    num_tokens = 4 if both else 2

    tokenizer = VideoTokenizer(
        dim = 16, dim_latent = 16, patch_size = 32, image_height = 64, image_width = 64,
        num_latent_tokens = 2, encoder_depth = 1, decoder_depth = 1,
        time_block_every = 1, attn_heads = 1, attn_dim_head = 8
    ) if use_tokenizer else None

    dynamics = DynamicsWorldModel(
        dim = 16, dim_latent = 16,
        num_latent_tokens = num_tokens, num_spatial_tokens = num_tokens,
        video_tokenizer = tokenizer,
        aux_image_encoder = MockAuxEncoder() if use_aux_encoder else None,
        num_discrete_actions = 2,
        depth = 1, time_block_every = 1, attn_heads = 1, attn_dim_head = 8
    )

    loss = dynamics(video = torch.randn(1, 3, 2, 64, 64))
    assert loss.numel() == 1

    mock_env = MockEnv((64, 64), vectorized = False, num_envs = 1, terminate_after_step = 2)

    experience = dynamics.interact_with_env(mock_env, max_timesteps = 2, env_is_vectorized = False)
    assert experience.latents.shape[-2] == num_tokens

@param('latent_consistency_loss_weight', (0., 1.))
def test_latent_consistency_loss(latent_consistency_loss_weight):
    import torch
    from dreamer4.dreamer4 import VideoTokenizer

    model = VideoTokenizer(
        dim = 16,
        dim_latent = 16,
        patch_size = 8,
        image_height = 64,
        image_width = 64,
        num_latent_tokens = 4,
        encoder_depth = 1,
        decoder_depth = 1,
        time_block_every = 1,
        attn_heads = 1,
        attn_dim_head = 16,
        latent_consistency_loss_weight = latent_consistency_loss_weight
    )

    video = torch.randn(2, 3, 2, 64, 64)
    loss, (losses, recon_video) = model(video, return_intermediates=True)

    assert loss.numel() == 1

    if latent_consistency_loss_weight > 0.:
        assert losses.latent_consistency.item() > 0.
    else:
        assert losses.latent_consistency.item() == 0.

def test_moss_sequential_caching():
    from dreamer4.dreamer4 import VideoTokenizer

    tokenizer = VideoTokenizer(
        dim = 128,
        dim_latent = 32,
        patch_size = 4,
        num_latent_tokens = 32,
        channels = 3,
        image_height = 16,
        image_width = 16,
        encoder_depth = 4,
        decoder_depth = 2,
        encoder_moss_layers = (2,),
        decoder_moss_layers = (1,),
        moss_kwargs = dict(causal = True),
        lpips_loss_weight = 0.,
        use_causal_conv3d = False,
        use_shifted_patch_tokenization = False
    )

    tokenizer.eval()

    video = torch.randn(1, 3, 5, 16, 16)

    with torch.no_grad():

        # parallel encode

        parallel_latents = tokenizer(video, return_latents = True)

        # sequential encode with caching

        time_cache = None
        seq_latents = []

        for t in range(5):
            frame = video[:, :, t:t+1]
            frame_latents, time_cache = tokenizer(frame, return_latents = True, return_time_cache = True, time_cache = time_cache)
            seq_latents.append(frame_latents)

        seq_latents = torch.cat(seq_latents, dim = 1)

        assert torch.allclose(parallel_latents, seq_latents, atol = 1e-4)

        # test decoder with moss

        loss = tokenizer(video)
        assert loss.numel() == 1

@param('slot_attention_initted_latents', (False, True))
@param('decoder_slot_attention_initted_spatial_tokens', (False, True))
def test_tokenizer_slot_attention(
    slot_attention_initted_latents,
    decoder_slot_attention_initted_spatial_tokens
):
    from dreamer4.dreamer4 import VideoTokenizer

    tokenizer = VideoTokenizer(
        16,
        encoder_depth = 1,
        decoder_depth = 1,
        time_block_every = 1,
        dim_latent = 16,
        patch_size = 16,
        attn_dim_head = 8,
        num_latent_tokens = 4,
        slot_attention_initted_latents = slot_attention_initted_latents,
        decoder_slot_attention_initted_spatial_tokens = decoder_slot_attention_initted_spatial_tokens
    )

    video = torch.randn(2, 3, 4, 32, 32)
    loss = tokenizer(video)
    assert loss.numel() == 1

    latents = tokenizer(video, return_latents = True)
    assert latents.shape[-1] == 16

    recon = tokenizer.decode(latents, 32, 32)
    assert recon.shape == video.shape
import pytest

def test_tokenizer_with_h_net():
    from dreamer4.dreamer4 import VideoTokenizer
    import torch

    tokenizer = VideoTokenizer(
        dim = 128,
        dim_latent = 32,
        patch_size = 4,
        num_latent_tokens = 32,
        channels = 3,
        image_height = 32,
        image_width = 32,
        encoder_depth = 2,
        decoder_depth = 2,
        time_block_every = 2,
        attn_dim_head = 32,
        attn_heads = 4,
        h_net_layer = 1,
        h_net_kwargs = dict(depth = 2, heads = 4, dim_head = 32)
    )

    video = torch.randn(2, 3, 4, 32, 32)
    loss, (losses, recon) = tokenizer(video, return_intermediates=True)

    assert recon.shape == video.shape
    assert isinstance(losses.h_net, torch.Tensor)
    assert losses.h_net.item() > 0.

def test_dynamics_apply_fire():
    from dreamer4.dreamer4 import DynamicsWorldModel
    import torch

    model = DynamicsWorldModel(
        dim = 32,
        dim_latent = 16,
        num_latent_tokens = 16,
        depth = 2,
        attn_heads = 2,
        num_continuous_actions = 4,
        num_discrete_actions = 0,
        multi_token_pred_len = 1,
        use_loss_normalization = False
    )

    model.apply_fire_(num_iters = 2)

def test_axial_space_time_transformer_with_h_net():
    from dreamer4.dreamer4 import AxialSpaceTimeTransformer
    import torch

    transformer = AxialSpaceTimeTransformer(
        dim = 128,
        depth = 4,
        attn_heads = 4,
        attn_dim_head = 32,
        time_block_every = 2,
        space_height = 8,
        space_width = 8,
        h_net_layer = 1,
        h_net_kwargs = dict(depth = 2, heads = 4, dim_head = 32)
    )

    tokens = torch.randn(2, 4, 8 * 8, 128) # (b, t, s, d)
    out, intermediates = transformer(tokens, return_intermediates = True)

    assert out.shape == tokens.shape
    assert isinstance(intermediates.h_net_loss, torch.Tensor)
    assert intermediates.h_net_loss.item() > 0.

def test_dynamics_model_with_h_net_caching():
    from dreamer4.dreamer4 import DynamicsWorldModel
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DynamicsWorldModel(
        dim=32,
        dim_latent=16,
        num_latent_tokens=16,
        depth=2,
        time_block_every=1,  # use 1 to test full temporal attention
        spatial_pre_encoder_depth=2,
        action_pre_encoder_depth=2,
        value_head_mlp_depth=1,
        policy_head_mlp_depth=1,
        attn_heads=2,
        num_continuous_actions=4,
        num_discrete_actions=0,
        multi_token_pred_len=1,
        use_loss_normalization=False,
        transformer_kwargs=dict(
            h_net_layer=1,
            h_net_kwargs=dict(depth=1, heads=2, dim_head=16)
        )
    ).to(device)

    model.eval()

    seq_len = 4
    batch_size = 2

    actions_seq = torch.randn(batch_size, seq_len, 4, device='cpu')
    latents = torch.randn(batch_size, seq_len, 16, 16, device='cpu')
    signal_levels = torch.randint(0, 10, (batch_size, seq_len), device='cpu')
    step_sizes_log2 = torch.zeros((batch_size,), device='cpu', dtype=torch.long)

    with torch.no_grad():
        parallel_pred, (embeds, _) = model(
            latents=latents,
            continuous_actions=actions_seq,
            signal_levels=signal_levels,
            step_sizes_log2=step_sizes_log2,
            latent_is_noised=True,
            return_pred_only=True,
            return_intermediates=True
        )

        seq_embeds_list = []
        seq_flow_list = []
        time_cache = None

        for t in range(seq_len):
            latents_t = latents[:, t:t+1]
            signal_levels_t = signal_levels[:, t:t+1]

            if t == 0:
                actions_t = torch.zeros_like(actions_seq[:, 0:1])
            else:
                actions_t = actions_seq[:, t-1:t]

            pred_t, (embeds_t, intermediates) = model(
                latents=latents_t,
                continuous_actions=actions_t,
                signal_levels=signal_levels_t,
                step_sizes_log2=step_sizes_log2,
                latent_is_noised=True,
                time_cache=time_cache,
                return_pred_only=True,
                return_intermediates=True
            )
            time_cache = intermediates
            seq_embeds_list.append(embeds_t.agent)
            seq_flow_list.append(pred_t.flow)

        seq_agent_tokens = torch.cat(seq_embeds_list, dim=1)
        seq_flow = torch.cat(seq_flow_list, dim=1)

    assert torch.allclose(embeds.agent, seq_agent_tokens, atol=1e-5), "Agent tokens diverge"
    assert torch.allclose(parallel_pred.flow, seq_flow, atol=1e-5), "Flow output diverges"

@pytest.mark.parametrize('mot_temporal', [True, False])
def test_tokenizer_mot_temporal(mot_temporal):
    from dreamer4.dreamer4 import VideoTokenizer
    import torch

    tokenizer = VideoTokenizer(
        dim = 16,
        dim_latent = 8,
        patch_size = 2,
        num_latent_tokens = 4,
        channels = 3,
        image_height = 8,
        image_width = 8,
        encoder_depth = 2,
        decoder_depth = 2,
        time_block_every = 2,
        attn_dim_head = 16,
        attn_heads = 2,
        mot_temporal = mot_temporal
    )

    tokenizer.eval()

    batch_size = 2
    seq_len = 3

    video = torch.randn(batch_size, 3, seq_len, 8, 8)

    with torch.no_grad():
        latents_parallel, _ = tokenizer(
            video,
            return_latents=True,
            return_time_cache=True
        )

        seq_latents_list = []
        time_cache = None

        for t in range(seq_len):
            video_t = video[:, :, t:t+1]
            latents_t, time_cache = tokenizer(
                video_t,
                return_latents=True,
                time_cache=time_cache,
                return_time_cache=True
            )
            seq_latents_list.append(latents_t)

        latents_seq = torch.cat(seq_latents_list, dim=1)

    assert torch.allclose(latents_parallel, latents_seq, atol = 1e-5), "Tokenizer latents diverge"

@pytest.mark.parametrize('mot_temporal', [True, False])
def test_dynamics_mot_temporal(mot_temporal):
    from dreamer4.dreamer4 import DynamicsWorldModel
    import torch
    import torch.nn.functional as F

    model = DynamicsWorldModel(
        dim = 16,
        dim_latent = 4,
        num_latent_tokens = 4,
        num_spatial_tokens = 4,
        num_agents = 1,
        num_discrete_actions = 2,
        depth = 2,
        mot_temporal = mot_temporal
    )

    model.eval()

    batch_size = 2
    seq_len = 3

    latents = torch.randn(batch_size, seq_len, 4, 4)
    actions = torch.randint(0, 2, (batch_size, seq_len, 1))
    signal_levels = torch.randint(0, 10, (batch_size, seq_len))
    step_sizes_log2 = torch.zeros((batch_size,), dtype=torch.long)

    with torch.no_grad():
        pred_parallel, _ = model(
            latents = latents,
            discrete_actions = actions,
            signal_levels = signal_levels,
            step_sizes_log2 = step_sizes_log2,
            latent_is_noised = True,
            return_pred_only = True,
            return_intermediates = True
        )

        seq_flow_list = []
        time_cache = None

        padded_actions = F.pad(actions[:, :-1], (0, 0, 1, 0), value = 0)

        for t in range(seq_len):
            latents_t = latents[:, t:t+1]
            signal_levels_t = signal_levels[:, t:t+1]
            actions_t = padded_actions[:, t:t+1]

            pred_t, (_, intermediates) = model(
                latents = latents_t,
                discrete_actions = actions_t,
                signal_levels = signal_levels_t,
                step_sizes_log2 = step_sizes_log2,
                latent_is_noised = True,
                time_cache = time_cache,
                return_pred_only = True,
                return_intermediates = True
            )
            time_cache = intermediates
            seq_flow_list.append(pred_t.flow)

        seq_flow = torch.cat(seq_flow_list, dim=1)

    assert torch.allclose(pred_parallel.flow, seq_flow, atol = 1e-5), "Dynamics flow output diverges"

def test_latent_init_patch_size():
    from dreamer4.dreamer4 import VideoTokenizer

    tokenizer = VideoTokenizer(
        dim = 16,
        encoder_depth = 1,
        decoder_depth = 1,
        time_block_every = 1,
        dim_latent = 16,
        patch_size = 8,
        latent_init_patch_size = 4,
        attn_dim_head = 8,
        num_latent_tokens = 4,
        image_height = 32,
        image_width = 32,
        slot_attention_initted_latents = True
    )

    video = torch.randn(2, 3, 2, 32, 32)
    latents = tokenizer(video, return_latents=True)
    assert latents.shape[-1] == 16
    assert latents.shape[-2] == 4

    loss = tokenizer(video)
    assert loss.numel() == 1

    # Also test same patch size
    tokenizer_same = VideoTokenizer(
        dim = 16,
        encoder_depth = 1,
        decoder_depth = 1,
        time_block_every = 1,
        dim_latent = 16,
        patch_size = 8,
        latent_init_patch_size = 8,
        attn_dim_head = 8,
        num_latent_tokens = 4,
        image_height = 32,
        image_width = 32,
        slot_attention_initted_latents = True
    )

    latents_same = tokenizer_same(video, return_latents=True)
    assert latents_same.shape[-1] == 16
    assert latents_same.shape[-2] == 4

def test_aug_conditioning():
    from dreamer4.dreamer4 import VideoTokenizer
    tokenizer = VideoTokenizer(
        dim = 128,
        dim_latent = 16,
        patch_size = 8,
        encoder_depth = 2,
        decoder_depth = 2,
        attn_dim_head = 8,
        num_latent_tokens = 4,
        image_height = 32,
        image_width = 32,
        has_aug_conditioning = True
    )

    video = torch.randn(2, 3, 5, 32, 32)
    aug_id = torch.randint(0, 3, (2,))

    latents = tokenizer(video, aug_id = aug_id, return_latents = True)
    assert latents.shape[-1] == 16
    assert latents.shape[-2] == 4

    loss = tokenizer(video, aug_id = aug_id)
    assert loss.numel() == 1
    assert loss >= 0.

def test_dynamics_aug_conditioning():
    from dreamer4.dreamer4 import VideoTokenizer, DynamicsWorldModel

    tokenizer = VideoTokenizer(
        dim = 16,
        dim_latent = 16,
        patch_size = 8,
        encoder_depth = 1,
        decoder_depth = 1,
        attn_dim_head = 8,
        num_latent_tokens = 4,
        image_height = 32,
        image_width = 32,
        has_aug_conditioning = True
    )

    dynamics = DynamicsWorldModel(
        dim = 16,
        video_tokenizer = tokenizer,
        dim_latent = 16,
        max_steps = 64,
        num_latent_tokens = 4,
        depth = 1,
        num_discrete_actions = 4,
        attn_dim_head = 8,
        has_aug_conditioning = True
    )

    video = torch.randn(2, 3, 5, 32, 32)
    discrete_actions = torch.randint(0, 4, (2, 4, 1))
    aug_id = torch.randint(0, 3, (2,))

    loss, *_ = dynamics(video = video, discrete_actions = discrete_actions, aug_id = aug_id, return_all_losses = True)

    assert loss.numel() == 1
    assert loss >= 0.

    loss_int, *_ = dynamics(video = video, discrete_actions = discrete_actions, aug_id = 1, return_all_losses = True)
    assert loss_int.numel() == 1

@param('loss_type, detach_target, sigreg_num_subspaces, predict_residual', [
    ('smooth_l1', False, 4, True),
    ('cosine', True, None, False)
])
def test_latent_ar_loss(loss_type, detach_target, sigreg_num_subspaces, predict_residual):
    from dreamer4.dreamer4 import LatentAutoregressiveLoss
    import torch

    latent_ar_loss = LatentAutoregressiveLoss(
        dim = 512,
        loss_type = loss_type,
        detach_target = detach_target,
        sigreg_num_subspaces = sigreg_num_subspaces,
        predict_residual = predict_residual
    )

    loss, sigreg, pred = latent_ar_loss(torch.randn(2, 4, 512))
    assert loss.ndim == 0

    loss, pred = latent_ar_loss(torch.randn(2, 4, 512), return_unreduced_loss = True)
    assert loss.shape == (2, 3, 512)

@param('decoder_flow_steps, separate_flow_decoder, override_grad_frac', [
    (1, False, None),
    (4, False, None),
    (4, True, None),
    (4, True, lambda t: (t < 0.5).float())
])
def test_separate_flow_decoder(
    decoder_flow_steps,
    separate_flow_decoder,
    override_grad_frac
):
    from dreamer4.dreamer4 import VideoTokenizer
    import torch

    tokenizer = VideoTokenizer(
        dim = 256,
        dim_latent = 16,
        patch_size = 4,
        image_height = 32,
        image_width = 32,
        num_latent_tokens = 4,
        encoder_depth = 1,
        decoder_depth = 1,
        decoder_flow_steps = decoder_flow_steps,
        separate_flow_decoder = separate_flow_decoder,
        flow_decoder_train_prob = 0.5,
        latent_receive_grad_frac = override_grad_frac
    )
    video = torch.randn(1, 3, 2, 32, 32)

    # test parameter separation
    if separate_flow_decoder:
        decoder_params = list(tokenizer.decoder.parameters())
        flow_decoder_params = list(tokenizer.flow_decoder.parameters())

        assert len(decoder_params) == len(flow_decoder_params)
        for p1, p2 in zip(decoder_params, flow_decoder_params):
            assert p1 is not p2, 'parameters must not be shared between base decoder and flow decoder'

    # test forward
    loss = tokenizer(video)
    assert loss.numel() == 1
    loss.backward()

    # test inference
    latents = tokenizer(video, return_latents = True)
    decoded_video = tokenizer.decode(latents, height=32, width=32)
    assert decoded_video.shape == video.shape

@param('space_attention_use_pope', (False, True))
def test_space_pope(space_attention_use_pope):
    from dreamer4.dreamer4 import VideoTokenizer, special_token_mask, block_mask_special_tokens_right
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        from torch.nn.attention.flex_attention import create_block_mask
        has_flex = True
    except ImportError:
        has_flex = False

    tokenizer = VideoTokenizer(
        dim = 16,
        dim_latent = 16,
        patch_size = 4,
        num_latent_tokens = 4,
        channels = 3,
        image_height = 16,
        image_width = 16,
        encoder_depth = 1,
        decoder_depth = 1,
        time_block_every = 1,
        attn_dim_head = 16,
        attn_heads = 2,
        lpips_loss_weight = 0.,
        space_attention_use_pope = space_attention_use_pope
    ).to(device)

    # test forward
    video = torch.randn(1, 3, 2, 16, 16, device = device)
    loss = tokenizer(video)
    assert loss.numel() == 1

    # if checking pope mask
    if not space_attention_use_pope or not has_flex:
        return

    space_seq_len = (16 // 4) ** 2 + 4
    num_latents = 4

    q_seq = torch.arange(space_seq_len, device = device)
    k_seq = torch.arange(space_seq_len, device = device)
    dense_mask = special_token_mask(q_seq, k_seq, space_seq_len, num_latents, False)

    block_mask_fn = block_mask_special_tokens_right(space_seq_len, num_latents, False)
    flex_block_mask = create_block_mask(block_mask_fn, 1, 1, space_seq_len, space_seq_len, device = device)

    from einops import rearrange
    flex_dense = flex_block_mask.to_dense()
    assert torch.all(dense_mask == rearrange(flex_dense, '1 1 q k -> q k'))

def test_actor_spr_wrapper():
    from dreamer4.dreamer4 import ActionEmbedder, ActorSPRWrapper

    # parameters
    batch_size = 2
    seq_len = 10
    dim = 64
    num_discrete_actions = 4
    num_continuous_actions = 2

    action_embedder = ActionEmbedder(
        dim = dim,
        num_discrete_actions = num_discrete_actions,
        num_continuous_actions = num_continuous_actions,
        can_unembed = True,
        unembed_dim = dim * 4,
        num_unembed_preds = 1
    )

    wrapper = ActorSPRWrapper(
        action_embedder = action_embedder,
        dim = dim * 4,
        num_rollouts = 3,
        spr_loss_weight = 1.0,
        kl_loss_weight = 1.0,
        sigreg_loss_weight = 0.05
    )

    policy_embed = torch.randn(batch_size, seq_len, dim * 4)
    discrete_actions = torch.randint(0, num_discrete_actions, (batch_size, seq_len, 1))
    continuous_actions = torch.randn(batch_size, seq_len, num_continuous_actions)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    total_loss, losses = wrapper(
        policy_embed=policy_embed,
        discrete_actions=discrete_actions,
        continuous_actions=continuous_actions,
        mask=mask
    )

    assert total_loss.ndim == 0
    assert total_loss.item() >= 0

    spr_loss, kl_loss, sigreg_loss = losses
    assert spr_loss.item() >= 0
    assert kl_loss.item() >= 0
    assert sigreg_loss.item() >= 0

@param('encode_temporal_diff', (False, True))
def test_temporal_diff(encode_temporal_diff):
    from dreamer4.dreamer4 import VideoTokenizer

    tokenizer = VideoTokenizer(
        16,
        encoder_depth = 1,
        decoder_depth = 1,
        time_block_every = 1,
        dim_latent = 16,
        patch_size = 16,
        attn_dim_head = 8,
        num_latent_tokens = 4,
        lpips_loss_weight = 0.,
        encode_temporal_diff = encode_temporal_diff
    )

    video = torch.randn(2, 3, 4, 32, 32)

    loss, (losses, recon_video, *rest) = tokenizer(video, return_intermediates = True)
    loss.backward()

    latents = tokenizer.tokenize(video)
    out = tokenizer.decode(latents, height = 32, width = 32)

    assert out.shape == video.shape

@param('custom_activation', (False, True))
def test_custom_activations(custom_activation):
    from dreamer4.dreamer4 import VideoTokenizer, DynamicsWorldModel

    act = 'relu_squared' if custom_activation else 'silu'
    ff_kwargs = dict(activation=act)

    tokenizer = VideoTokenizer(
        dim=16,
        encoder_depth=1,
        decoder_depth=1,
        time_block_every=1,
        dim_latent=16,
        patch_size=16,
        num_latent_tokens=4,
        decoder_pos_emb_mlp_activation=act,
        ff_kwargs=ff_kwargs
    )

    world_model = DynamicsWorldModel(
        dim=16,
        video_tokenizer=tokenizer,
        dim_latent=16,
        max_steps=16,
        num_tasks=2,
        num_latent_tokens=4,
        depth=1,
        num_spatial_tokens=2,
        num_discrete_actions=4,
        spatial_pre_encoder_depth=1,
        action_pre_encoder_depth=1,
        ssl_lapo=True,
        ssl_tem=True,
        lapo_kwargs=dict(
            latent_action_embed_mlp_activation=act,
            pred_next_state_embed_mlp_activation=act,
            pred_raw_latent_mlp_activation=act
        ),
        tem_kwargs=dict(
            learned_relative_encode_mlp_activation=act,
            init_hiddens_mlp_activation=act,
            sensory_encoder_mlp_activation=act,
            sensory_decoder_mlp_activation=act
        ),
        actor_spr=True,
        actor_nlp_kwargs=dict(
            dynamics_mlp_activation=act
        ),
        latent_ar=True,
        latent_ar_layer=0,
        latent_ar_kwargs=dict(
            mlp_activation=act
        ),
        policy_head_mlp_activation=act,
        value_head_mlp_activation=act,
        agent_state_pred_mlp_activation=act,
        state_terminal_pred_mlp_activation=act,
        pred_rewards_mlp_activation=act,
        pred_terminals_mlp_activation=act,
        pred_is_truncated_mlp_activation=act,
        ff_kwargs=ff_kwargs
    )

    assert exists(tokenizer)
    assert exists(world_model)

    video = torch.randn(1, 3, 2, 32, 32)
    actions = torch.randint(0, 4, (1, 2, 1))

    latents = tokenizer(video, return_latents=True)
    assert exists(latents)

    loss = world_model(
        video=video,
        discrete_actions=actions
    )
    assert exists(loss)

    loss.backward()

def test_memmap_replay(tmp_path):
    from torch.utils._pytree import tree_flatten
    from dreamer4.dreamer4 import Actions, Experience, is_tensor, DynamicsWorldModel, combine_experiences

    time = 10

    def make_exp(batch):
        return Experience(
            latents = torch.randn(batch, time, 32),
            actions = Actions(discrete = torch.randint(0, 2, (batch, time, 1)), continuous = None),
            rewards = torch.randn(batch, time),
            terminals = torch.randint(0, 2, (batch,), dtype = torch.bool),
            lens = torch.full((batch,), time),
            is_truncated = torch.zeros((batch,), dtype = torch.bool),
            episode_return = torch.randn(batch)
        )

    # combine multiple experiences and write to buffer

    exp = combine_experiences([make_exp(2), make_exp(3)])
    batch = 5

    buffer = Experience.create_memmap_replay_buffer(
        exp,
        str(tmp_path / 'buffer'),
        max_episodes = 10,
        max_timesteps = time,
        overwrite = True,
        circular = True
    )

    exp.add_to_memmap_buffer(buffer)
    assert len(buffer) == batch

    # rehydrate from buffer and verify round-trip

    rehydrated = Experience.from_buffer_dict(next(iter(buffer.dataloader(batch_size = batch))))

    exp_flat, _ = tree_flatten(vars(exp))
    re_flat, _ = tree_flatten(vars(rehydrated))

    for orig, recon in zip(exp_flat, re_flat):
        if is_tensor(recon):
            orig_t = orig if is_tensor(orig) else torch.tensor([orig] * batch)
            assert torch.allclose(orig_t.float(), recon.float())
        else:
            assert orig == recon

    # verify rehydrated experience can be fed through world model

    world_model = DynamicsWorldModel(
        dim = 64,
        dim_latent = 32,
        num_discrete_actions = 2,
        num_spatial_tokens = 4,
        num_latent_tokens = 1
    )

    loss = world_model(
        latents = rehydrated.latents,
        discrete_actions = rehydrated.actions.discrete,
        rewards = rehydrated.rewards,
        terminals = rehydrated.terminals
    )

    assert loss.ndim == 0
    loss.backward()

    # clear and verify empty

    buffer.clear()
    assert len(buffer) == 0

def test_video_tokenizer_trainer():
    from dreamer4.dreamer4 import VideoTokenizer
    from dreamer4.trainers import VideoDataset, VideoTokenizerTrainer, video_tensor_collate_fn

    tokenizer = VideoTokenizer(
        dim = 16,
        patch_size = 8,
        dim_latent = 32,
        num_latent_tokens = 1,
        image_height = 64,
        image_width = 64,
        lpips_loss_weight = 0.
    )

    dataset = VideoDataset(
        folder = 'tests/mock_videos',
        image_size = 64,
        exts = ['mp4']
    )

    trainer = VideoTokenizerTrainer(
        model = tokenizer,
        dataset = dataset,
        batch_size = 2,
        num_train_steps = 1,
        collate_fn = video_tensor_collate_fn
    )

    trainer()

def test_dynamics_save_load(tmp_path):
    import os
    import torch
    from pathlib import Path
    from torch_einops_utils.nn import count_parameters
    from dreamer4.dreamer4 import VideoTokenizer, DynamicsWorldModel

    tokenizer = VideoTokenizer(
        dim = 128,
        dim_latent = 16,
        patch_size = 8,
        image_height = 64,
        image_width = 64,
        encoder_depth = 2,
        decoder_depth = 2,
    )

    model = DynamicsWorldModel(
        video_tokenizer = tokenizer,
        dim_latent = tokenizer.dim_latent,
        dim = 256,
        depth = 2,
        attn_dim_head = 32,
        attn_heads = 4,
        num_continuous_actions = 0,
        num_discrete_actions = 0,
    )

    assert model.video_tokenizer is not tokenizer, "Tokenizer was not deepcopied!"
    for param in model.video_tokenizer.parameters():
        assert not param.requires_grad, "Tokenizer parameters should not require gradients"
    assert not model.video_tokenizer.training, "Tokenizer should be in eval mode"

    save_path = tmp_path / 'test_dyn.pt'
    model.save(save_path)

    # Using init_and_load provided by @save_load from torch_einops_utils

    reconstituted_model = DynamicsWorldModel.init_and_load(save_path, strict=True)

    assert exists(reconstituted_model.video_tokenizer), "Tokenizer was not reconstituted!"
    assert reconstituted_model.video_tokenizer.dim_latent == tokenizer.dim_latent, "Tokenizer config mismatch!"

    # Assert parameters match
    model_params = count_parameters(model)
    recon_params = count_parameters(reconstituted_model)
    assert model_params == recon_params, f"Weights mismatch: {model_params} != {recon_params}"

@pytest.mark.parametrize('action_type', ['discrete', 'continuous', 'multi_discrete'])
def test_actions_e2e(tmp_path, action_type):
    import torch
    import gymnasium as gym
    import numpy as np
    from dreamer4.dreamer4 import VideoTokenizer, DynamicsWorldModel
    from dreamer4.trainers import VideoTrajectoryDataset, BehaviorCloneTrainer
    from dreamer4.env import RecordToFolderEnvWrapper

    data_folder = tmp_path / 'mock_videos'
    data_folder.mkdir(exist_ok=True, parents=True)

    class RenderObsWrapper(gym.ObservationWrapper):
        def observation(self, obs):
            img = self.env.render()
            import cv2
            return cv2.resize(img, (64, 64))

    if action_type == 'discrete':
        raw_env = RenderObsWrapper(gym.make('CartPole-v1', render_mode = 'rgb_array'))
    elif action_type == 'continuous':
        raw_env = RenderObsWrapper(gym.make('LunarLanderContinuous-v3', render_mode = 'rgb_array'))
    elif action_type == 'multi_discrete':
        class MockMultiDiscreteEnv(gym.Env):
            def __init__(self):
                self.action_space = gym.spaces.MultiDiscrete([3, 4])
                self.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
            def reset(self, **kwargs):
                return self.observation_space.sample(), {}
            def step(self, action):
                return self.observation_space.sample(), 1.0, False, False, {}
            def render(self):
                return self.observation_space.sample()
        raw_env = MockMultiDiscreteEnv()

    env = RecordToFolderEnvWrapper(raw_env, folder = str(data_folder))
    env.reset()
    for _ in range(8):
        if action_type == 'continuous':
            action = np.random.uniform(0.0, 1.0, size=env.action_space.shape).astype(np.float32)
            env.step(action)
        else:
            env.step(env.action_space.sample())
    env.close()

    tokenizer = VideoTokenizer(
        dim = 16,
        patch_size = 8,
        dim_latent = 32,
        num_latent_tokens = 1,
        image_height = 64,
        image_width = 64
    )
    tokenizer.eval().requires_grad_(False)

    dataset = VideoTrajectoryDataset(
        folder = str(data_folder),
        image_size = 64,
        channels = 3,
        max_num_frames = 8,
        exts = ['mp4']
    )

    assert len(dataset) > 0, "No datasets found!"
    sample = dataset[0]

    num_discrete_actions = 0
    num_continuous_actions = 0

    if action_type == 'discrete':
        num_discrete_actions = int(raw_env.action_space.n)
    elif action_type == 'continuous':
        num_continuous_actions = int(raw_env.action_space.shape[0])
    elif action_type == 'multi_discrete':
        num_discrete_actions = tuple(raw_env.action_space.nvec)

    model = DynamicsWorldModel(
        video_tokenizer = tokenizer,
        dim_latent = tokenizer.dim_latent,
        dim = 16,
        depth = 1,
        attn_dim_head = 8,
        attn_heads = 2,
        num_continuous_actions = num_continuous_actions,
        num_discrete_actions = num_discrete_actions
    )

    trainer = BehaviorCloneTrainer(
        model = model,
        dataset = dataset,
        batch_size = 2,
        learning_rate = 3e-4,
        num_train_steps = 2,
        checkpoint_every = 0,
        log_video = False,
        log_video_every = 0,
        use_tensorboard = False,
        use_wandb = False,
        log_dir = str(tmp_path / 'logs_test')
    )

    trainer()

@pytest.mark.parametrize('byol_use_sem', (False, True))
def test_video_byol(byol_use_sem):
    from dreamer4.trainers import VideoTokenizerTrainer
    from dreamer4.dreamer4 import VideoTokenizer
    from torch.utils.data import Dataset

    class MockDataset(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return dict(
                video = torch.randn(3, 2, 64, 64),
                time_lens = 2
            )

    dataset = MockDataset()

    tokenizer = VideoTokenizer(
        16,
        encoder_depth = 1,
        decoder_depth = 1,
        time_block_every = 1,
        dim_latent = 16,
        patch_size = 32,
        attn_dim_head = 16,
        num_latent_tokens = 4,
        has_byol = True,
        byol_use_sem = byol_use_sem
    )

    trainer = VideoTokenizerTrainer(
        tokenizer,
        dataset = dataset,
        num_train_steps = 1,
        batch_size = 1,
        cpu = True,
        max_grad_norm = 0.5,
        use_ema = True
    )

    trainer()

def test_state_entropy_bonus_e2e():
    import torch
    from dreamer4.dreamer4 import DynamicsWorldModel, VideoTokenizer
    from dreamer4.mocks import MockEnv

    tokenizer = VideoTokenizer(
        dim = 16, dim_latent = 16, patch_size = 32, image_height = 64, image_width = 64,
        num_latent_tokens = 2, encoder_depth = 1, decoder_depth = 1,
        time_block_every = 1, attn_heads = 1, attn_dim_head = 8
    )

    world_model = DynamicsWorldModel(
        dim = 16, dim_latent = 16,
        num_latent_tokens = 2, num_spatial_tokens = 2,
        video_tokenizer = tokenizer,
        num_discrete_actions = 2,
        depth = 1, time_block_every = 1, attn_heads = 1, attn_dim_head = 8,
        add_state_pred_head = True,
        state_entropy_bonus_weight = 0.05
    )

    mock_env = MockEnv((64, 64), vectorized = False, num_envs = 1, terminate_after_step = 2)

    experience = world_model.interact_with_env(mock_env, max_timesteps = 2, env_is_vectorized = False)
    assert exists(experience)

@param('clip_decoded', (True, False))
def test_hallucination_metric(clip_decoded):
    import torch
    from dreamer4.dreamer4 import DynamicsWorldModel, VideoTokenizer

    tokenizer = VideoTokenizer(
        dim = 16,
        dim_latent = 16,
        patch_size = 32,
        image_size = 64,
        num_latent_tokens = 2,
        encoder_depth = 1,
        decoder_depth = 1,
        time_block_every = 1,
        attn_heads = 1,
        attn_dim_head = 8
    )

    dynamics = DynamicsWorldModel(
        dim = 16,
        dim_latent = 16,
        num_latent_tokens = 2,
        num_spatial_tokens = 2,
        video_tokenizer = tokenizer,
        num_continuous_actions = 16,
        depth = 1,
        time_block_every = 1,
        attn_heads = 1,
        attn_dim_head = 8
    )

    batch = 2
    time = 4

    video = torch.randn(batch, 3, time, 64, 64)
    actions = torch.randn(batch, time, 16)

    metric = dynamics(
        video = video,
        continuous_actions = actions,
        return_latent_disagreement = True,
        return_latent_disagreement_clip_decoded = clip_decoded
    )

    assert metric.shape == (batch, time)

@param('attn_every', [1, None])
@param('var_len', [False, True])
def test_state_tokenizer(attn_every, var_len):
    from dreamer4.dreamer4 import StateTokenizer

    tokenizer = StateTokenizer(
        dim_state = 17,
        num_latent_tokens = 4,
        dim_latent = 16,
        dim = 32,
        depth = 1,
        attn_every = attn_every,
        attn_heads = 1,
        attn_dim_head = 8,
        mae_fraction = 0.5 if exists(attn_every) else 0.
    )

    batch = 2
    time = 4

    obs = torch.randn(batch, time, 17)
    time_lens = torch.tensor([2, 4]) if var_len else None

    # test training pass
    loss = tokenizer(obs, time_lens = time_lens)
    loss.backward()

    # test tokenize
    latents = tokenizer.tokenize(obs, time_lens = time_lens)
    assert latents.shape == (batch, time, 4, 16)

    # test sequential equivalence

    time_cache = None
    seq_latents = []
    for t in range(time):
        step_time_lens = None
        if exists(time_lens):
            step_time_lens = (time_lens - t).clamp(min = 0, max = 1)

        step_latents, time_cache = tokenizer.tokenize(obs[:, t:t+1], time_cache = time_cache, return_time_cache = True, time_lens = step_time_lens)
        seq_latents.append(step_latents)

    seq_latents = torch.cat(seq_latents, dim = 1)
    assert torch.allclose(latents, seq_latents, atol = 1e-5), "Sequential and parallel tokenization should yield equal latents"

def test_experience_combine_and_unbind():
    from dreamer4.dreamer4 import Experience, combine_experiences

    # Create two individual experiences of different lengths
    exp1 = Experience(
        latents=torch.randn(1, 10, 16),
        rewards=torch.randn(1, 10),
        terminals=torch.zeros(1, dtype=torch.bool),
        lens=torch.tensor([10]),
        is_truncated=torch.tensor([False]),
        episode_return=torch.tensor([5.0])
    )

    exp2 = Experience(
        latents=torch.randn(1, 5, 16),
        rewards=torch.randn(1, 5),
        terminals=torch.ones(1, dtype=torch.bool),
        lens=torch.tensor([5]),
        is_truncated=torch.tensor([True]),
        episode_return=torch.tensor([2.0])
    )

    # Combine them
    combined = combine_experiences([exp1, exp2])

    assert combined.latents.shape == (2, 10, 16)
    assert combined.rewards.shape == (2, 10)
    assert combined.terminals.shape == (2,)
    assert combined.lens.shape == (2,)
    assert combined.is_truncated.shape == (2,)
    assert combined.episode_return.shape == (2,)

    # Unbind them
    unbound_exps = list(combined.unbind())

    assert len(unbound_exps) == 2

    # Check first experience
    assert unbound_exps[0].latents.shape == (1, 10, 16)
    assert unbound_exps[0].rewards.shape == (1, 10)
    assert unbound_exps[0].terminals.shape == (1,)
    assert unbound_exps[0].lens.shape == (1,)
    assert unbound_exps[0].is_truncated.shape == (1,)
    assert unbound_exps[0].episode_return.shape == (1,)
    assert unbound_exps[0].lens.item() == 10

    # Check second experience (should be properly sliced to length 5)
    assert unbound_exps[1].latents.shape == (1, 5, 16)
    assert unbound_exps[1].rewards.shape == (1, 5)
    assert unbound_exps[1].terminals.shape == (1,)
    assert unbound_exps[1].lens.shape == (1,)
    assert unbound_exps[1].is_truncated.shape == (1,)
    assert unbound_exps[1].episode_return.shape == (1,)
    assert unbound_exps[1].lens.item() == 5

    # Check iteration via __iter__
    iter_exps = list(combined)
    assert len(iter_exps) == 2
    assert iter_exps[1].latents.shape == (1, 5, 16)

@param('has_views', [False, True])
@param('use_time_rnn', [False, True])
def test_dynamics_model_state_sequential_parallel_cache(has_views, use_time_rnn):
    from dreamer4.dreamer4 import StateTokenizer, DynamicsWorldModel, pack_one
    from einops import rearrange

    tokenizer = StateTokenizer(
        dim_state = 17,
        num_latent_tokens = 4,
        dim_latent = 16,
        dim = 32,
        depth = 1,
        attn_every = 1,
        attn_heads = 1,
        attn_dim_head = 8
    )

    model = DynamicsWorldModel(
        dim = 32,
        state_tokenizer = tokenizer,
        num_agents = 1,
        dim_latent = 16,
        num_latent_genes = 2,
        num_latent_tokens = 4,
        depth = 2,
        time_block_every = 1,
        attn_heads = 2,
        attn_dim_head = 16,
        use_time_rnn = use_time_rnn,
        num_video_views = 2 if has_views else 1
    )
    model.eval()

    batch = 2
    time = 4

    state_shape = (batch, *((2,) if has_views else tuple()), time, 17)
    state = torch.randn(*state_shape)

    # parallel pass
    _, (intermediates_parallel, _) = model(
        state = state,
        signal_levels = 1,
        step_sizes = 1,
        return_pred_only = True,
        return_intermediates = True,
        latent_is_noised = True
    )

    agent_embed_parallel = intermediates_parallel.agent

    # sequential pass
    tokenizer_cache = None
    dynamics_cache = None
    agent_embed_sequential = []

    if not has_views:
        state = rearrange(state, 'b ... -> b 1 ...')

    state_packed, unpack_views = pack_one(state, '* t d')

    for t in range(time):
        step_latents_packed, tokenizer_cache = tokenizer(
            state_packed[:, t:t+1, :],
            return_latents = True,
            time_cache = tokenizer_cache,
            return_time_cache = True
        )
        step_latents = unpack_views(step_latents_packed, '* t n d')
        step_latents = rearrange(step_latents, 'b v t n d -> b t v n d')

        _, (intermediates_seq, dynamics_cache) = model(
            latents = step_latents,
            signal_levels = 1,
            step_sizes = 1,
            time_cache = dynamics_cache,
            return_pred_only = True,
            return_intermediates = True,
            latent_is_noised = True,
            latent_has_view_dim = True
        )

        agent_embed_sequential.append(intermediates_seq.agent)

    agent_embed_sequential = torch.cat(agent_embed_sequential, dim = 1)

    assert torch.allclose(agent_embed_parallel, agent_embed_sequential, atol = 1e-4), "Sequential and parallel passes through DynamicsWorldModel with StateTokenizer should yield equal agent embeds"

@param('has_views', [False, True])
@param('use_time_rnn', [False, True])
def test_dynamics_model_concurrent_tokenizers_sequential_parallel_cache(has_views, use_time_rnn):
    from dreamer4.dreamer4 import StateTokenizer, VideoTokenizer, DynamicsWorldModel, pack_one

    state_tokenizer = StateTokenizer(
        dim_state = 17,
        num_latent_tokens = 4,
        dim_latent = 16,
        dim = 32,
        depth = 1,
        attn_every = 1,
        attn_heads = 1,
        attn_dim_head = 8
    )

    video_tokenizer = VideoTokenizer(
        dim = 32,
        dim_latent = 16,
        patch_size = 4,
        image_height = 16,
        image_width = 16,
        num_latent_tokens = 8,
        encoder_depth = 1,
        decoder_depth = 1,
        attn_heads = 1,
        attn_dim_head = 8,
        time_block_every = 1
    )

    model = DynamicsWorldModel(
        dim = 32,
        state_tokenizer = state_tokenizer,
        video_tokenizer = video_tokenizer,
        num_agents = 1,
        dim_latent = 16,
        num_latent_genes = 2,
        num_latent_tokens = 12, # 8 from video + 4 from state
        depth = 2,
        time_block_every = 1,
        attn_heads = 2,
        attn_dim_head = 16,
        use_time_rnn = use_time_rnn,
        num_video_views = 2 if has_views else 1
    )
    model.eval()

    batch = 2
    time = 4

    state_shape = (batch, *((2,) if has_views else tuple()), time, 17)
    state = torch.randn(*state_shape)

    video_shape = (batch, *((2,) if has_views else tuple()), 3, time, 16, 16)
    video = torch.randn(*video_shape)

    # parallel pass
    _, (intermediates_parallel, _) = model(
        state = state,
        video = video,
        signal_levels = 1,
        step_sizes = 1,
        return_pred_only = True,
        return_intermediates = True,
        latent_is_noised = True
    )

    agent_embed_parallel = intermediates_parallel.agent

    video_tokenizer_cache = None
    state_tokenizer_cache = None
    dynamics_cache = None
    agent_embed_sequential = []

    if not has_views:
        state = rearrange(state, 'b ... -> b 1 ...')
        video = rearrange(video, 'b ... -> b 1 ...')

    state_packed, state_unpack_views = pack_one(state, '* t d')
    video_packed, video_unpack_views = pack_one(video, '* c t vh vw')

    for t in range(time):
        state_step = state_packed[:, t:t+1, :]
        video_step = video_packed[:, :, t:t+1, :, :]

        state_latents_packed, state_tokenizer_cache = state_tokenizer(
            state_step,
            return_latents = True,
            time_cache = state_tokenizer_cache,
            return_time_cache = True
        )
        state_latents = state_unpack_views(state_latents_packed, '* t n d')
        state_latents = rearrange(state_latents, 'b v t n d -> b t v n d')

        video_latents_packed, video_tokenizer_cache = video_tokenizer(
            video_step,
            return_latents = True,
            time_cache = video_tokenizer_cache,
            return_time_cache = True
        )
        video_latents = video_unpack_views(video_latents_packed, '* t n d')
        video_latents = rearrange(video_latents, 'b v t n d -> b t v n d')

        step_latents = torch.cat((video_latents, state_latents), dim = -2)

        _, (intermediates_seq, dynamics_cache) = model(
            latents = step_latents,
            signal_levels = 1,
            step_sizes = 1,
            time_cache = dynamics_cache,
            return_pred_only = True,
            return_intermediates = True,
            latent_is_noised = True
        )

        agent_embed_sequential.append(intermediates_seq.agent)

    agent_embed_sequential = torch.cat(agent_embed_sequential, dim = 1)

    assert torch.allclose(agent_embed_parallel, agent_embed_sequential, atol = 1e-4), "Sequential and parallel passes through DynamicsWorldModel with Concurrent Tokenizers should yield equal agent embeds"

@param('use_time_rnn', [False, True])
def test_dynamics_model_all_three_tokenizers_sequential_parallel_cache(use_time_rnn):
    from dreamer4.dreamer4 import AudioTokenizer, StateTokenizer, VideoTokenizer, DynamicsWorldModel, pack_one

    has_views = True

    video_tokenizer = VideoTokenizer(
        dim = 32,
        dim_latent = 16,
        image_size = 16,
        patch_size = 4,
        num_latent_tokens = 4,
        encoder_depth = 1,
        decoder_depth = 1,
        attn_heads = 1,
        attn_dim_head = 8,
        time_block_every = 1
    )

    state_tokenizer = StateTokenizer(
        dim_state = 8,
        dim_latent = 16,
        num_latent_tokens = 1, # as requested
        depth = 1,
        attn_heads = 1,
        attn_dim_head = 8
    )

    audio_tokenizer = AudioTokenizer(
        channels = 2,
        sample_rate = 16000,
        n_fft = 400,
        n_mels = 16,
        hop_length = 160,
        dim = 32,
        dim_latent = 16,
        patch_size = 4,
        num_latent_tokens = 2,
        encoder_depth = 1,
        decoder_depth = 1,
        attn_heads = 1,
        attn_dim_head = 8,
        time_block_every = 1
    )

    model = DynamicsWorldModel(
        dim = 32,
        video_tokenizer = video_tokenizer,
        state_tokenizer = state_tokenizer,
        audio_tokenizer = audio_tokenizer,
        num_agents = 1,
        dim_latent = 16,
        num_latent_genes = 2,
        depth = 2,
        time_block_every = 1,
        attn_heads = 2,
        attn_dim_head = 16,
        use_time_rnn = use_time_rnn,
        num_discrete_actions = 4,
        num_video_views = 2 if has_views else 1
    )
    model.eval()

    batch = 2
    time = 4

    # Video
    video_shape = (batch, *((2,) if has_views else tuple()), 3, time, 16, 16)
    video = torch.randn(*video_shape)

    # State
    state_shape = (batch, *((2,) if has_views else tuple()), time, 8)
    state = torch.randn(*state_shape)

    # Audio
    time_frames = 16
    samples = 2400
    audio_shape = (batch, *((2,) if has_views else tuple()), 2, time, samples)
    audio = torch.randn(*audio_shape)

    # parallel pass
    _, (intermediates_parallel, _) = model(
        video = video,
        state = state,
        audio = audio,
        signal_levels = 1,
        step_sizes = 1,
        return_pred_only = True,
        return_intermediates = True,
        latent_is_noised = True
    )

    agent_embed_parallel = intermediates_parallel.agent

    # sequential pass
    video_tokenizer_cache = None
    state_tokenizer_cache = None
    audio_tokenizer_cache = None
    dynamics_cache = None
    agent_embed_sequential = []

    video_packed, video_unpack_views = pack_one(video, '* c t vh vw')
    state_packed, state_unpack_views = pack_one(state, '* t d')
    audio_packed, audio_unpack_views = pack_one(audio, '* c t s')

    for t in range(time):
        video_step = video_packed[:, :, t:t+1, :, :]
        state_step = state_packed[:, t:t+1, :]
        audio_step = audio_packed[:, :, t:t+1, :]

        # Video latents
        vid_latents_packed, video_tokenizer_cache = video_tokenizer(
            video_step,
            return_latents = True,
            time_cache = video_tokenizer_cache,
            return_time_cache = True
        )
        vid_latents = video_unpack_views(vid_latents_packed, '* t n d')
        vid_latents = rearrange(vid_latents, 'b v t n d -> b t v n d')

        # State latents
        state_latents_packed, state_tokenizer_cache = state_tokenizer(
            state_step,
            return_latents = True,
            time_cache = state_tokenizer_cache,
            return_time_cache = True
        )
        state_latents = state_unpack_views(state_latents_packed, '* t n d')
        state_latents = rearrange(state_latents, 'b v t n d -> b t v n d')

        # Audio latents
        audio_latents_packed, audio_tokenizer_cache = audio_tokenizer(
            audio_step,
            return_latents = True,
            time_cache = audio_tokenizer_cache,
            return_time_cache = True
        )
        audio_latents = audio_unpack_views(audio_latents_packed, '* t n d')
        audio_latents = rearrange(audio_latents, 'b v t n d -> b t v n d')

        # Concat latents
        latents = torch.cat((vid_latents, state_latents, audio_latents), dim = -2)

        _, (intermediates_seq, dynamics_cache) = model(
            latents = latents,
            signal_levels = 1,
            step_sizes = 1,
            time_cache = dynamics_cache,
            return_pred_only = True,
            return_intermediates = True,
            latent_is_noised = True,
            latent_has_view_dim = True
        )

        agent_embed_sequential.append(intermediates_seq.agent)

    agent_embed_sequential = torch.cat(agent_embed_sequential, dim = 1)

    assert torch.allclose(agent_embed_parallel, agent_embed_sequential, atol = 1e-4), "Sequential and parallel passes through DynamicsWorldModel with all 3 Tokenizers should yield equal agent embeds"

    # test generative rl with prompts

    prompt_time = 2
    prompt_video = video[:, :, :, :prompt_time]
    prompt_state = state[:, :, :prompt_time]
    prompt_audio = audio[:, :, :, :prompt_time]

    generated_outputs = model.generate(
        time_steps = 4,
        batch_size = batch,
        num_steps = 2,
        image_height = 16,
        image_width = 16,
        prompt = prompt_video,
        prompt_state = prompt_state,
        prompt_audio = prompt_audio,
        return_for_policy_optimization = True
    )

    assert exists(generated_outputs)

def test_attention_pool_einsum_attend_matches_the_sdpa_default():
    from dreamer4.dreamer4 import AttentionPool

    torch.manual_seed(0)
    pool = AttentionPool(16, heads = 2, dim_head = 8)
    x = torch.randn(2, 5, 16)
    hiddens = [torch.randn(2, 5, 16) for _ in range(3)]

    sdpa_out = pool(x, hiddens = hiddens)
    pool.einsum_attend = True
    einsum_out = pool(x, hiddens = hiddens)

    assert torch.allclose(einsum_out, sdpa_out, atol = 1e-6)


@pytest.mark.parametrize('mode, block_size, expected_sources', [
    # sublayers 0..5 (attention, feedforward for each of 3 layers), then the final output
    ('full', 1, [1, 2, 3, 4, 5, 6, 7]),
    ('block', 2, [1, 2, 2, 3, 3, 4, 4]),
    ('block', 3, [1, 2, 2, 2, 3, 3, 3]),
])
def test_attention_residuals_follow_the_block_source_layout(monkeypatch, mode, block_size, expected_sources):
    from dreamer4.dreamer4 import AxialSpaceTimeTransformer, AttentionResiduals

    source_counts = []
    original_mix = AttentionResiduals.mix

    def recording_mix(self, index, sources):
        source_counts.append(len(sources))
        return original_mix(self, index, sources)

    monkeypatch.setattr(AttentionResiduals, 'mix', recording_mix)

    torch.manual_seed(0)
    transformer = AxialSpaceTimeTransformer(
        dim = 16, depth = 3, attn_heads = 2, attn_dim_head = 8, time_block_every = 2,
        use_attn_pool = False, attn_res = mode, attn_res_block_size = block_size,
    )
    tokens = torch.randn(2, 3, 5, 16)
    out = transformer(tokens)
    out.sum().backward()

    assert source_counts == expected_sources
    assert out.shape == tokens.shape
    assert transformer.attn_res.queries.grad[1:].abs().sum() > 0


def test_attention_residuals_start_as_an_equal_weight_average():
    from dreamer4.dreamer4 import AttentionResiduals

    attn_res = AttentionResiduals(8, num_sublayers = 2)
    sources = [torch.randn(2, 4, 8) * scale for scale in (1., 10., 100.)]

    assert torch.allclose(attn_res.mix(1, sources), sum(sources) / 3, atol = 1e-5)


def test_attention_residuals_require_the_attention_pools_off():
    from dreamer4.dreamer4 import AxialSpaceTimeTransformer

    with pytest.raises(ValueError, match = 'use_attn_pool'):
        AxialSpaceTimeTransformer(dim = 16, depth = 3, attn_heads = 2, attn_dim_head = 8, attn_res = 'full')


@pytest.mark.parametrize('mode', ['full', 'block'])
def test_dynamics_world_model_trains_and_generates_with_attention_residuals(mode):
    from dreamer4.dreamer4 import DynamicsWorldModel

    torch.manual_seed(0)
    dynamics = DynamicsWorldModel(
        dim = 16,
        dim_latent = 8,
        max_steps = 64,
        num_latent_tokens = 2,
        num_spatial_tokens = 2,
        depth = 3,
        time_block_every = 2,
        attn_heads = 2,
        attn_dim_head = 8,
        num_discrete_actions = 0,
        num_continuous_actions = 3,
        transformer_kwargs = dict(use_attn_pool = False, attn_res = mode),
    )

    loss = dynamics(
        latents = torch.randn(2, 4, 2, 8),
        continuous_actions = torch.rand(2, 3, 3),
        rewards = torch.randn(2, 4),
    )
    loss.backward()
    assert torch.isfinite(loss)

    generations = dynamics.generate(time_steps = 4, batch_size = 2, use_time_cache = True, return_rewards_per_frame = True)
    assert generations.latents.shape[:2] == (2, 4)

@param('shortcut', (False, True))
def test_clean_context_frames_have_no_latent_loss(shortcut):
    import torch
    from dreamer4.dreamer4 import DynamicsWorldModel

    torch.manual_seed(0)
    world_model = DynamicsWorldModel(
        dim = 16, dim_latent = 4, max_steps = 8, num_latent_tokens = 1, num_spatial_tokens = 1,
        depth = 1, time_block_every = 1, attn_heads = 1, attn_dim_head = 8, num_continuous_actions = 2,
        prob_shortcut_train = float(shortcut), clean_context_frame_prob = 0.5,
    )
    latents = torch.randn(3, 6, 1, 1, 4)
    actions = torch.rand(3, 6, 2)
    lens = torch.tensor((6, 6, 4))

    seen_levels = []
    get_times = world_model.get_times_from_signal_level

    def record_levels(signal_levels, *args, **kwargs):
        seen_levels.append(signal_levels.clone())
        return get_times(signal_levels, *args, **kwargs)

    world_model.get_times_from_signal_level = record_levels
    loss, losses = world_model(latents = latents, continuous_actions = actions, lens = lens, latent_has_view_dim = True, return_all_losses = True)
    loss.backward()

    assert torch.isfinite(loss)
    levels = seen_levels[0]
    # Clean frames are at the top signal level, which shortcut samples never reach on their grid.
    assert (levels == world_model.max_steps - 1).any()
    assert all(torch.isfinite(p.grad).all() for p in world_model.parameters() if p.grad is not None)

    if shortcut:
        # The three predictions of a shortcut step: the main one, then the two half steps. The clean frames keep their
        # input latents in the second half step (they are context, not targets).
        world_model.get_times_from_signal_level = get_times
        inputs = []
        hook = world_model.latents_to_spatial_tokens.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach().clone()))
        torch.manual_seed(1)
        world_model(latents = latents, continuous_actions = actions, lens = lens, latent_has_view_dim = True, return_all_losses = True)
        hook.remove()
        main, first, second = inputs
        assert torch.equal(main, first)
        clean = (first - second).flatten(2).abs().amax(-1) == 0
        assert clean.any() and not clean.all()

    # A model where every frame is clean context has no latent loss to learn from: the flow loss is 0.
    world_model.clean_context_frame_prob = 0.999999
    _, losses = world_model(latents = latents, continuous_actions = actions, lens = lens, latent_has_view_dim = True, return_all_losses = True)
    assert losses.flow.abs().item() < 1e-6

def test_clean_context_frame_noise_matches_dream_context():
    import torch
    from dreamer4.dreamer4 import DynamicsWorldModel

    torch.manual_seed(0)
    world_model = DynamicsWorldModel(
        dim = 16, dim_latent = 4, max_steps = 8, num_latent_tokens = 1, num_spatial_tokens = 1,
        depth = 1, time_block_every = 1, attn_heads = 1, attn_dim_head = 8, num_continuous_actions = 2,
        prob_shortcut_train = 0., clean_context_frame_prob = 0.5,
    )
    latents = torch.randn(4, 6, 1, 1, 4)
    actions = torch.rand(4, 6, 2)

    def inputs_and_levels(noise):
        # the same seed gives the same levels, clean frames and noise; only the noise mix of the clean frames changes
        world_model.clean_context_frame_noise = noise
        inputs, levels = [], []
        get_times = world_model.get_times_from_signal_level
        world_model.get_times_from_signal_level = lambda signal_levels, *args, **kwargs: levels.append(signal_levels.clone()) or get_times(signal_levels, *args, **kwargs)
        hook = world_model.latents_to_spatial_tokens.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach().clone()))
        torch.manual_seed(1)
        world_model(latents = latents, continuous_actions = actions, latent_has_view_dim = True, return_all_losses = True)
        hook.remove()
        del world_model.get_times_from_signal_level
        return inputs[0].reshape(latents.shape), levels[0]

    clean_input, levels = inputs_and_levels(0.)
    dream_input, dream_levels = inputs_and_levels(0.3)
    assert torch.equal(levels, dream_levels)

    # A clean frame keeps the label max_steps - 1 and gets the dream mix 0.7 x + 0.3 noise in place of 7 / 8 x + 1 / 8 noise.
    noise = (clean_input - latents * 7 / 8) * 8
    expected = latents * 0.7 + noise * 0.3
    top = levels == world_model.max_steps - 1
    same = (dream_input - clean_input).flatten(2).abs().amax(-1) == 0
    moved = (dream_input - expected).flatten(2).abs().amax(-1) < 1e-5
    assert torch.all(same | (top & moved))
    assert (top & moved & ~same).any()
    assert same[~top].all()
