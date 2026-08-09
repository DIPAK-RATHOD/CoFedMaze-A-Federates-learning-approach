from evaluation.robustness import FaultInjector, FaultProfile


def test_fault_injector_applies_configured_message_drop_rate():
    injector = FaultInjector(FaultProfile(message_drop_rate=1.0, seed=7))

    assert injector.should_drop_message()
    assert injector.sent == 1
    assert injector.dropped == 1


def test_fault_injectors_with_same_seed_are_reproducible():
    profile = FaultProfile(validation_noise_std=0.5, seed=12)
    assert FaultInjector(profile).perturb_validation_reward(2.0) == FaultInjector(profile).perturb_validation_reward(2.0)
