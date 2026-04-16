# Final Report 4.0 Test Inventory (Integration, System, Acceptance)

Generated from `# commit:` tags and in-file TS/TC annotations under `tests/`.

## 1) Integration tests (`# commit: integration test`)

### 1.1 TS/TC (or TI) explicitly annotated

> Note: `tests/integration/test_encoder_htm_integration.py` uses **TI-001..TI-014** IDs instead of TC IDs.

| File | Test Function | TS # | TC # | TI # |
|---|---|---:|---:|---:|
| `tests/integration/test_encoder_htm_integration.py` | `test_input_field_initialization_with_rdse` | TS-11 | — | TI-001 |
| `tests/integration/test_encoder_htm_integration.py` | `test_input_field_encode_scalar_values` | TS-11 | — | TI-002 |
| `tests/integration/test_encoder_htm_integration.py` | `test_input_field_encode_sequence` | TS-11 | — | TI-003 |
| `tests/integration/test_encoder_htm_integration.py` | `test_input_field_decode_active_state` | TS-11 | — | TI-004 |
| `tests/integration/test_encoder_htm_integration.py` | `test_input_field_with_category_encoder` | TS-11 | — | TI-005 |
| `tests/integration/test_encoder_htm_integration.py` | `test_category_encoding_through_input_field` | TS-11 | — | TI-006 |
| `tests/integration/test_encoder_htm_integration.py` | `test_category_unknown_value_handling` | TS-11 | — | TI-007 |
| `tests/integration/test_encoder_htm_integration.py` | `test_input_field_with_date_encoder` | TS-11 | — | TI-008 |
| `tests/integration/test_encoder_htm_integration.py` | `test_date_encoding_through_input_field` | TS-11 | — | TI-009 |
| `tests/integration/test_encoder_htm_integration.py` | `test_single_input_field_to_column_field` | TS-11 | — | TI-010 |
| `tests/integration/test_encoder_htm_integration.py` | `test_non_spatial_column_field` | TS-11 | — | TI-011 |
| `tests/integration/test_encoder_htm_integration.py` | `test_temporal_learning_with_sequence` | TS-11 | — | TI-012 |
| `tests/integration/test_encoder_htm_integration.py` | `test_column_field_bursting_behavior` | TS-11 | — | TI-013 |
| `tests/integration/test_encoder_htm_integration.py` | `test_two_input_fields_to_column_field` | TS-11 | — | TI-014 |

### 1.2 Integration tests missing explicit TS/TC in test body

| File | Test Function | TS # | TC # |
|---|---|---:|---:|
| `tests/integration/test_encoder_to_htm.py` | `test_encoder_to_htm_receives_sdr_object` | — | — |
| `tests/integration/test_input_to_encoder.py` | `test_input_to_encoder_passes_records_into_encoder` | — | — |
| `tests/integration/test_encoder_htm_integration.py` | `test_multiple_fields_temporal_sequence` | TS-11 (suite-level) | — |
| `tests/integration/test_encoder_htm_integration.py` | `test_encode_compute_decode_cycle` | TS-11 (suite-level) | — |
| `tests/integration/test_encoder_htm_integration.py` | `test_predictive_state_decoding` | TS-11 (suite-level) | — |
| `tests/integration/test_encoder_htm_integration.py` | `test_small_encoder_size` | TS-11 (suite-level) | — |
| `tests/integration/test_encoder_htm_integration.py` | `test_large_encoder_size` | TS-11 (suite-level) | — |
| `tests/integration/test_encoder_htm_integration.py` | `test_varying_sparsity_levels` | TS-11 (suite-level) | — |
| `tests/integration/test_encoder_htm_integration.py` | `test_invalid_decode_state` | TS-11 (suite-level) | — |
| `tests/integration/test_encoder_htm_integration.py` | `test_clear_states_resets_properly` | TS-11 (suite-level) | — |
| `tests/integration/test_encoder_htm_integration.py` | `test_advance_states_preserves_history` | TS-11 (suite-level) | — |
| `tests/integration/test_encoder_htm_integration.py` | `test_duty_cycle_updates` | TS-11 (suite-level) | — |
| `tests/integration/test_encoder_htm_integration.py` | `test_simple_branching_sequence` | TS-11 (suite-level) | — |
| `tests/integration/test_encoder_htm_integration.py` | `test_branching_with_different_contexts` | TS-11 (suite-level) | — |
| `tests/integration/test_encoder_htm_integration.py` | `test_triple_branching` | TS-11 (suite-level) | — |
| `tests/integration/test_encoder_htm_integration.py` | `test_similar_encoder_outputs_activate_similar_columns` | TS-11 (suite-level) | — |
| `tests/integration/test_encoder_htm_integration.py` | `test_encoder_sparsity_affects_column_activation` | TS-11 (suite-level) | — |
| `tests/integration/test_encoder_htm_integration.py` | `test_proximal_synapses_strengthen_for_active_input` | TS-11 (suite-level) | — |
| `tests/integration/test_encoder_htm_integration.py` | `test_predictive_state_propagates_to_input_field` | TS-11 (suite-level) | — |

## 2) System tests (`# commit: system/integration test`)

| File | Test Function | TS # | TC # |
|---|---|---:|---:|
| `tests/test_cartpole_brain_training.py` | `test_train_env_policy_returns_metrics` | TS-21 | TC-167 |
| `tests/test_env_adapter.py` | `test_env_adapter_accepts_instantiated_fingym` | TS-20 | TC-170 |
| `tests/test_env_adapter.py` | `test_env_adapter_accepts_make_kwargs` | TS-20 | TC-171 |
| `tests/test_env_adapter.py` | `test_env_adapter_rejects_kwargs_with_env_instance` | TS-20 | TC-172 |
| `tests/test_fin_gym.py` | `test_fingym_builds_observation_from_dataframe` | — | — |
| `tests/test_fin_gym.py` | `test_fingym_step_progression_and_reward` | — | — |
| `tests/test_fin_gym.py` | `test_fingym_rejects_invalid_action` | — | — |
| `tests/test_fin_gym.py` | `test_fingym_loads_csv` | — | — |
| `tests/integration/test_input_to_encoder.py` | `test_sine_wave_through_input_handler` | — | — |

## 3) Acceptance tests (`# commit: system/acceptance test`)

| File | Test Function | TS # | TC # |
|---|---|---:|---:|
| `tests/test_demo_driver_full_demo.py` | `test_run_full_demo_generates_visual_artifacts` | TS-10 | TC-097 |

## 4) Quick totals

- Integration tests tagged by commit annotation: **33**
- System/integration tests tagged by commit annotation: **9**
- System/acceptance tests tagged by commit annotation: **1**

## 5) Recommended TS table update actions

1. Keep TS-10 / TC-097 as-is for acceptance.
2. Keep TS-20 and TS-21 mappings as-is for system tests.
3. For TS-11 integration suite, decide whether TI-001..TI-014 should be mirrored as TC IDs in your TS tables (for consistency).
4. Assign TS/TC IDs to currently unnumbered FinGym and pipeline tests so your final report has full traceability.
