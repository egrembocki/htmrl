# System Functional (SF) Test Coverage Report

This table maps each Software Feature (SF) from Team 20 SWENG 481 4.0 to the most relevant test file(s) in the codebase. Gaps are noted for missing or partial coverage.

| SF ID     | Description (abridged)                                                                 | Test File(s) / Status                                   |
|-----------|----------------------------------------------------------------------------------------|---------------------------------------------------------|
| SF-A-1    | Input tool: input_data() method                                                        | test_input_handler_load_data.py                         |
| SF-B-1    | Input tool: ensure no NaN                                                              | test_input_handler_validate_data.py                     |
| SF-B-2    | Input tool: no missing values                                                          | test_input_handler_validate_data.py                     |
| SF-B-3    | Input tool: validate file path                                                         | test_input_handler_load_data.py                         |
| SF-B-4    | Input tool: validate file type                                                         | test_input_handler_load_data.py                         |
| SF-B-5    | Input tool: detect/normalize date info                                                 | test_input_handler_validate_data.py (partial)           |
| SF-B-6    | Input tool: detect duplicate columns                                                   | test_input_handler_validate_data.py                     |
| SF-C-1    | Encoder: convert dataset to SDR                                                        | test_encoder_handler_suite.py, test_base_encoder.py      |
| SF-C-2    | Encoder: SDR structure by size param                                                   | test_base_encoder.py, test_encoder_scalar.py, ...        |
| SF-C-3    | Scalar Encoder: encode value                                                           | test_encoder_scalar.py                                  |
| SF-C-4    | RDSE: encode value                                                                     | test_encoder_rdse.py                                    |
| SF-C-5    | Date Encoder: season params                                                            | test_encoder_date.py                                    |
| SF-C-6    | Date Encoder: day-of-week                                                              | test_encoder_date.py                                    |
| SF-C-7    | Date Encoder: weekend                                                                  | test_encoder_date.py                                    |
| SF-C-8    | Date Encoder: custom-days                                                              | test_encoder_date.py                                    |
| SF-C-9    | Date Encoder: holiday                                                                  | test_encoder_date.py                                    |
| SF-C-10   | Date Encoder: time-of-day                                                              | test_encoder_date.py                                    |
| SF-C-11   | Category Encoder: category set                                                         | test_encoder_category.py, test_encoder_category_new.py   |
| SF-C-12   | Fourier Encoder: wave                                                                  | test_encoder_fourier.py                                 |
| SF-C-13   | Geospatial Encoder: coordinate                                                         | test_encoder_geospatial.py                              |
| SF-D-1    | Agent config: model parameters                                                         | test_agent.py, test_trainer.py                          |
| SF-D-2    | Agent config: cells per column                                                         | test_agent.py, test_trainer.py                          |
| SF-D-3    | Agent config: number of columns                                                        | test_agent.py, test_trainer.py                          |
| SF-E-1    | Trainer: start training                                                                | test_trainer.py, test_cartpole_brain_training.py        |
| SF-E-2    | Trainer: build_brain method                                                            | test_trainer.py                                         |
| SF-E-3    | Trainer: add input_fields                                                              | test_trainer.py                                         |
| SF-E-4    | Trainer: add output_fields                                                             | test_trainer.py                                         |
| SF-E-5    | Trainer: add column_fields                                                             | test_trainer.py                                         |
| SF-E-6    | Trainer: maintain brain model ref                                                      | test_trainer.py                                         |
| SF-E-7    | Trainer: test method for logger                                                        | test_trainer.py, test_demo_driver_full_demo.py          |
| SF-F-1    | Logger: display training progress                                                      | test_trainer.py, test_demo_driver_full_demo.py          |
| SF-F-2    | Logger: store RMSE in text file                                                        | test_trainer.py (partial/manual check)                  |
| SF-F-3    | Logger: store shape in JSON                                                            | test_trainer.py (partial/manual check)                  |
| SF-F-4    | Logger: output final performance to console                                            | test_trainer.py, test_demo_driver_full_demo.py          |
| SF-F-5    | Logger: store final performance in text file                                           | test_trainer.py (partial/manual check)                  |
| SF-F-6    | Logger: set report artifact path                                                       | test_trainer.py (partial/manual check)                  |
| SF-F-7    | Logger: retrieve validated dataset                                                     | test_trainer.py (partial/manual check)                  |
| SF-F-8    | Logger: retrieve last trainer params                                                   | test_trainer.py (partial/manual check)                  |
| SF-F-9    | Logger: retrieve latest agent prediction reports                                       | test_trainer.py (partial/manual check)                  |
| SF-F-10   | Logger: report method calls for brain                                                  | test_trainer.py (partial/manual check)                  |
| SF-F-11   | Logger: report avg reward per step                                                     | test_trainer.py (partial/manual check)                  |
| SF-F-12   | Logger: get_logger method                                                             | test_trainer.py (partial/manual check)                  |
| SF-F-13   | Logger: show log origin by class name                                                  | test_trainer.py (partial/manual check)                  |
| SF-F-14   | Logger: Enum {INFO, WARNING, ERROR, DEBUG}                                             | test_trainer.py (partial/manual check)                  |
| SF-G-1    | Env: gymnasium setup with custom brain                                                 | test_env_adapter.py, test_fin_gym.py, test_cartpole_brain_training.py |
| SF-G-2    | Agent: brain with HTM/RL hierarchy                                                     | test_htm_brain.py, test_agent.py                        |
| SF-G-3    | Env: agent reads observations                                                          | test_env_adapter.py, test_fin_gym.py                    |
| SF-G-4    | InputFields: base encoder delegation                                                   | test_htm_input_field.py, test_encoder_handler_suite.py   |
| SF-G-5    | Env: reward tuple {action, state, reward}                                              | test_env_adapter.py, test_fin_gym.py                    |
| SF-G-6    | Env: reward algorithm                                                                  | test_env_adapter.py, test_fin_gym.py                    |
| SF-G-7    | Env: update observation state                                                          | test_env_adapter.py, test_fin_gym.py                    |
| SF-G-8    | Env: get/set action space                                                              | test_env_adapter.py, test_fin_gym.py                    |
| SF-G-9    | Env: get/set observation space                                                         | test_env_adapter.py, test_fin_gym.py                    |
| SF-G-10   | Env: interface for decoded action                                                      | test_env_adapter.py, test_fin_gym.py                    |
| SF-H-1    | Model: compute column states                                                           | test_htm_column.py, test_htm_cell.py                    |
| SF-H-2    | Model: compute cell states                                                             | test_htm_cell.py                                        |
| SF-H-3    | Model: compute segment states                                                          | test_htm_segments.py                                    |
| SF-H-4    | Model: learn from active state, predict                                                | test_htm.py, test_htm_brain.py                          |
| SF-H-5    | HTM: initialize input fields                                                           | test_htm_input_field.py                                 |
| SF-H-6    | HTM: initialize column fields                                                          | test_htm_column.py                                      |
| SF-H-7    | HTM: initialize output fields                                                          | test_htm_output_field.py                                |
| SF-H-8    | HTM input fields: set cells/active cells                                               | test_htm_input_field.py                                 |
| SF-H-9    | HTM input fields: encode/decode via delegation                                         | test_htm_input_field.py, test_encoder_handler_suite.py   |
| SF-H-10   | HTM column fields: manage input fields/cell states                                     | test_htm_column.py                                      |
| SF-I-1    | Trainer: save model with pytorch                                                       | test_trainer.py (partial/manual check)                  |
| SF-I-2    | Trainer: load model with pytorch                                                       | test_trainer.py (partial/manual check)                  |
| SF-J-1    | Scalar Encoder: decode SDR                                                             | test_encoder_scalar.py                                  |
| SF-J-2    | RDSE: decode SDR                                                                       | test_encoder_rdse.py                                    |
| SF-J-3    | Date Encoder: decode SDR                                                               | test_encoder_date.py                                    |
| SF-J-4    | Category Encoder: decode SDR                                                           | test_encoder_category.py, test_encoder_category_new.py   |
| SF-J-5    | Fourier Encoder: decode SDR                                                            | test_encoder_fourier.py                                 |
| SF-J-6    | Geospatial Encoder: decode SDR                                                         | test_encoder_geospatial.py                              |
| SF-K-1    | Trainer: grapher method for 2D matplotlib plot                                         | test_sdr_visual.py, test_demo_driver_full_demo.py       |

*Partial/manual check* means the feature may be exercised by the code but not explicitly asserted in a test. See test files for details.
# Team 20 Doc 3.5 SF Completion Report

Source analyzed: `docs/reports/Team 20 SWENG 481 3.5.docx` (Table mapping UF->SF).

> Note: the table contains **72 SF IDs** from `SF-A-1` through `SF-K-1`, while section numbering in the TOC says 71. This report tracks all 72 listed SF IDs.

| SF ID | Status | Requirement (Doc 3.5) | Evidence in codebase |
|---|---|---|---|
| SF-A-1 | Completed | The input tool shall have a method that allows the user to input their data with a data tool in the InputLayer through an input_data() method.. | InputHandler.input_data() plus file/type/date/missing-value validation and normalization in input_layer/input_handler.py; tests in tests/test_input_handler_*.py. |
| SF-B-1 | Completed | The input tool shall ensure that Nan (not a number) is not present in the input data. | InputHandler.input_data() plus file/type/date/missing-value validation and normalization in input_layer/input_handler.py; tests in tests/test_input_handler_*.py. |
| SF-B-2 | Completed | The input tool shall ensure the input data has no missing values. | InputHandler.input_data() plus file/type/date/missing-value validation and normalization in input_layer/input_handler.py; tests in tests/test_input_handler_*.py. |
| SF-B-3 | Completed | The input tool shall be validated for a proper file path. | InputHandler.input_data() plus file/type/date/missing-value validation and normalization in input_layer/input_handler.py; tests in tests/test_input_handler_*.py. |
| SF-B-4 | Completed | The input tool shall be validated for file type. | InputHandler.input_data() plus file/type/date/missing-value validation and normalization in input_layer/input_handler.py; tests in tests/test_input_handler_*.py. |
| SF-B-5 | Completed | The input tool shall detect date info and normalize if present. | InputHandler.input_data() plus file/type/date/missing-value validation and normalization in input_layer/input_handler.py; tests in tests/test_input_handler_*.py. |
| SF-C-1 | Completed | The encoder tools will convert the pre-validated dataset into an SDR. | Encoder implementations exist for scalar/RDSE/date/category/fourier/geospatial and composite SDR creation in encoder_layer/* and encoder_handler.py; decoder/encoder tests exist in tests/test_encoder_* and tests/test_decoder_*. |
| SF-C-2 | Completed | The encoder tools shall generate an SDR data structure based on the provided size parameter. | Encoder implementations exist for scalar/RDSE/date/category/fourier/geospatial and composite SDR creation in encoder_layer/* and encoder_handler.py; decoder/encoder tests exist in tests/test_encoder_* and tests/test_decoder_*. |
| SF-C-3 | Completed | The encoder tools shall encode an input value into a valid SDR using a Scalar Encoder. | Encoder implementations exist for scalar/RDSE/date/category/fourier/geospatial and composite SDR creation in encoder_layer/* and encoder_handler.py; decoder/encoder tests exist in tests/test_encoder_* and tests/test_decoder_*. |
| SF-C-4 | Completed | The encoder tools shall encode an input value into a valid SDR using a Random Distributed Scalar Encoder. | Encoder implementations exist for scalar/RDSE/date/category/fourier/geospatial and composite SDR creation in encoder_layer/* and encoder_handler.py; decoder/encoder tests exist in tests/test_encoder_* and tests/test_decoder_*. |
| SF-C-5 | Completed | The encoder tools shall encode a date and time input into a valid SDR using the Date Encoder with season parameters. | Encoder implementations exist for scalar/RDSE/date/category/fourier/geospatial and composite SDR creation in encoder_layer/* and encoder_handler.py; decoder/encoder tests exist in tests/test_encoder_* and tests/test_decoder_*. |
| SF-C-6 | Completed | The encoder tools shall encode a date and time input into a valid SDR using the Date Encoder with day-of-week parameters. | Encoder implementations exist for scalar/RDSE/date/category/fourier/geospatial and composite SDR creation in encoder_layer/* and encoder_handler.py; decoder/encoder tests exist in tests/test_encoder_* and tests/test_decoder_*. |
| SF-C-7 | Completed | The encoder tools shall encode a date and time input into a valid SDR using the Date Encoder with weekend parameters. | Encoder implementations exist for scalar/RDSE/date/category/fourier/geospatial and composite SDR creation in encoder_layer/* and encoder_handler.py; decoder/encoder tests exist in tests/test_encoder_* and tests/test_decoder_*. |
| SF-C-8 | Completed | The encoder tools shall encode a date and time input into a valid SDR using the Date Encoder with custom-days parameters. | Encoder implementations exist for scalar/RDSE/date/category/fourier/geospatial and composite SDR creation in encoder_layer/* and encoder_handler.py; decoder/encoder tests exist in tests/test_encoder_* and tests/test_decoder_*. |
| SF-C-9 | Completed | The encoder tools shall encode a date and time input into a valid SDR using the Date Encoder with holiday parameters. | Encoder implementations exist for scalar/RDSE/date/category/fourier/geospatial and composite SDR creation in encoder_layer/* and encoder_handler.py; decoder/encoder tests exist in tests/test_encoder_* and tests/test_decoder_*. |
| SF-C-10 | Completed | The encoder tools shall encode a date and time input into a valid SDR using the Date Encoder with time-of-day parameters. | Encoder implementations exist for scalar/RDSE/date/category/fourier/geospatial and composite SDR creation in encoder_layer/* and encoder_handler.py; decoder/encoder tests exist in tests/test_encoder_* and tests/test_decoder_*. |
| SF-C-11 | Completed | The encoder tools shall encode a category set into a valid SDR using the Category Encoder with width parameters and category list. | Encoder implementations exist for scalar/RDSE/date/category/fourier/geospatial and composite SDR creation in encoder_layer/* and encoder_handler.py; decoder/encoder tests exist in tests/test_encoder_* and tests/test_decoder_*. |
| SF-C-12 | Completed | The encoder tools shall encode a wave into a valid SDR using the Fourier Encoder with given frequency ranges, sparsity for these ranges or active bits for these ranges and size. | Encoder implementations exist for scalar/RDSE/date/category/fourier/geospatial and composite SDR creation in encoder_layer/* and encoder_handler.py; decoder/encoder tests exist in tests/test_encoder_* and tests/test_decoder_*. |
| SF-C-13 | Completed | The encoder tools shall encode a geospatial coordinate into a valid SDR using the Geospatial Encoder with scale, timestep, max radius, and altitude. | Encoder implementations exist for scalar/RDSE/date/category/fourier/geospatial and composite SDR creation in encoder_layer/* and encoder_handler.py; decoder/encoder tests exist in tests/test_encoder_* and tests/test_decoder_*. |
| SF-D-1 | Completed | The agent tools shall allow for agent model configuration through parameters. | Brain/ColumnField configuration through parameters in Trainer.build_brain(), _setup_column_fields(), and HTM column settings. |
| SF-D-2 | Completed | The agent tools shall allow for setting the cells per column in the agent model through parameters. | Brain/ColumnField configuration through parameters in Trainer.build_brain(), _setup_column_fields(), and HTM column settings. |
| SF-D-3 | Completed | The agent tools show allow for setting the number of columns in the agent model through parameters. | Brain/ColumnField configuration through parameters in Trainer.build_brain(), _setup_column_fields(), and HTM column settings. |
| SF-E-1 | Completed | The trainer tool shall use a method call to start training the HTM brain model. | Training entry points available via Trainer.train_column() / train_full_brain(). |
| SF-E-2 | Completed | The trainer tool shall have a build_brain method to act as a facade to create a brain-HTM/RL. | Facade method Trainer.build_brain() builds Brain/HTM field hierarchy. |
| SF-E-3 | Partial | The trainer tool shall have an add input_fields method for manual input_field creation. | Trainer has add_input_field() (singular) rather than add_input_fields(). |
| SF-E-4 | Partial | The trainer tool shall provide an add output_fields method for manual output_field creation. | Trainer has add_output_field() (singular) rather than add_output_fields(). |
| SF-E-5 | Partial | The trainer tool shall provide an add column_fields method for manual column_field creation. | Trainer has add_column_field() (singular) rather than add_column_fields(). |
| SF-E-6 | Completed | The trainer tool shall maintain a brain model reference. | Trainer maintains _main_brain and brains list references. |
| SF-E-7 | Completed | The trainer shall have a test method to perform agent model evaluations for the logger. | Trainer.test() provides evaluation metrics for logger/reporting. |
| SF-F-1 | Partial | The logger tool shall display the training progress in the console when the trainer performs a step. | Trainer emits per-step logger.info() messages during training. |
| SF-F-2 | Not Found | The logger shall store the root mean squared error for agent-brain model predictions in a text file. | No dedicated RMSE-to-text-file persistence method found. |
| SF-F-3 | Not Found | The logger shall store the shape of the agent-brain used during training steps into a JSON file. | No JSON shape artifact writer found. |
| SF-F-4 | Partial | The logger shall output to the console the final performance of all training steps from the agent-brain. | Trainer.test()/print_train_stats() can print summary metrics. |
| SF-F-5 | Partial | The logger shall store in a text file the final performance of all training steps from the agent-brain. | print_train_stats(save_path=...) can append summary text report. |
| SF-F-6 | Not Found | The logger tool shall have a method to set the desired path to save all report artifacts. | No logger method to set artifact path found. |
| SF-F-7 | Not Found | The logger tool shall be able to retrieve the dataset that was previously validated and saved through a method call. | No logger API to retrieve validated dataset found. |
| SF-F-8 | Not Found | The logger tool shall be able to retrieve the parameters used in the last trainer evaluation through a method call. | No logger API to retrieve last trainer parameter set found. |
| SF-F-9 | Not Found | The logger tool shall be able to retrieve the latest agent prediction reports through a method call. | No logger API to retrieve latest prediction reports found. |
| SF-F-10 | Not Found | The logger report method calls shall take in a brain to determine the correct log files to retrieve. | No logger report methods taking Brain to select log files found. |
| SF-F-11 | Partial | The logger shall generate a report with the average reward per step through a method call. | Average reward reporting path is not explicit; prediction/MSE reporting exists. |
| SF-F-12 | Completed | The logger tool shall have a get_logger method to set the current Class logger. | Global get_logger() helper exists in psu_capstone/log.py. |
| SF-F-13 | Completed | The logger tool shall always show where a log’s origin through Class name. | Logger naming includes class origin via getChild(ClassName). |
| SF-F-14 | Not Found | The logger shall have an Enum {INFO, WARNING, ERROR, DEBUG} structure. | No custom Enum {INFO,WARNING,ERROR,DEBUG}; standard logging levels are used. |
| SF-G-1 | Not Found | The environment tool shall set up an environment gymnasium that uses the custom HTM/RL brain as an agent. | No gymnasium environment implementation found; only EnvInterface protocol exists. |
| SF-G-2 | Partial | The agent shall have a brain composed of an HTM/RL hierarchy of Fields that can encode data into an SDR data shape from the environment. | Agent/Brain/HTM field hierarchy exists (HTM.py, brain.py), but gym environment integration is not present. |
| SF-G-3 | Not Found | The environment shall allow the agent to read observations into the agent's brain model InputFields. | No gymnasium environment implementation found; only EnvInterface protocol exists. |
| SF-G-4 | Not Found | The InputFields of the agent using the environment shall have a base encoder object that can encode input data into SDR data shape through delegation. | No gymnasium environment implementation found; only EnvInterface protocol exists. |
| SF-G-5 | Not Found | Every time the agent (brain) takes a step, the environment tool shall compute the current reward as tuple return of {action, state, reward} | No gymnasium environment implementation found; only EnvInterface protocol exists. |
| SF-G-6 | Not Found | The environment tool shall use a reward algorithm to compute the final reward returned in the tuple in SF-G-5. | No gymnasium environment implementation found; only EnvInterface protocol exists. |
| SF-G-7 | Not Found | Every time the agent (brain) takes a step, the environment tool shall update the observation state based on the current step or action taken by the brain through an update method. | No gymnasium environment implementation found; only EnvInterface protocol exists. |
| SF-G-8 | Not Found | The environment tool shall have a method to get and set the action space vector as a tuple. | No gymnasium environment implementation found; only EnvInterface protocol exists. |
| SF-G-9 | Not Found | The environment tool shall have a method to get and set the observation space vector as a tuple. | No gymnasium environment implementation found; only EnvInterface protocol exists. |
| SF-G-10 | Not Found | The environment tool shall have an interface to receive a decoded action from the HTM/RL model. (encoded or decoded) | No gymnasium environment implementation found; only EnvInterface protocol exists. |
| SF-H-1 | Completed | The model shall compute the states of Columns as {active, predictive, bursting}. | HTM field/state logic implemented in agent_layer/HTM.py and legacy_htm modules; dedicated tests test_htm_*.py. |
| SF-H-2 | Completed | The model should compute the states of cells as {active, winner, predictive}. | HTM field/state logic implemented in agent_layer/HTM.py and legacy_htm modules; dedicated tests test_htm_*.py. |
| SF-H-3 | Completed | The model should compute the states of Segments as {active, learning, matching}. | HTM field/state logic implemented in agent_layer/HTM.py and legacy_htm modules; dedicated tests test_htm_*.py. |
| SF-H-4 | Completed | The model should learn from the active list and active state, and return a prediction. | HTM field/state logic implemented in agent_layer/HTM.py and legacy_htm modules; dedicated tests test_htm_*.py. |
| SF-H-5 | Completed | The HTM shall be able to initialize input fields. | HTM field/state logic implemented in agent_layer/HTM.py and legacy_htm modules; dedicated tests test_htm_*.py. |
| SF-H-6 | Completed | The HTM shall be able to initialize column fields. | HTM field/state logic implemented in agent_layer/HTM.py and legacy_htm modules; dedicated tests test_htm_*.py. |
| SF-H-7 | Completed | The HTM shall be able to initialize output fields. | HTM field/state logic implemented in agent_layer/HTM.py and legacy_htm modules; dedicated tests test_htm_*.py. |
| SF-H-8 | Completed | The HTM input fields should be able to use a list of bits to set the number of cells and active cells inside of the field. | HTM field/state logic implemented in agent_layer/HTM.py and legacy_htm modules; dedicated tests test_htm_*.py. |
| SF-H-9 | Completed | The HTM input fields should be able to use delegation to encode and decode the set of cells. | HTM field/state logic implemented in agent_layer/HTM.py and legacy_htm modules; dedicated tests test_htm_*.py. |
| SF-H-10 | Completed | The HTM column fields should be able to manage input fields such as adding them to columns and managing their cell states. | HTM field/state logic implemented in agent_layer/HTM.py and legacy_htm modules; dedicated tests test_htm_*.py. |
| SF-I-1 | Not Found | The trainer tool shall use pytorch library to save a model for the user through a method call. | No PyTorch save/load model methods found in trainer/agent modules. |
| SF-I-2 | Not Found | The trainer tool shall use pytorch library to load a model for the user through a method call. | No PyTorch save/load model methods found in trainer/agent modules. |
| SF-J-1 | Completed | The encoder tool shall decode an SDR made from the Scalar Encoder and return a valid value. | decode() methods implemented across encoders (scalar/rdse/date/category/fourier/geospatial) with decoder tests. |
| SF-J-2 | Completed | The encoder tool shall decode an SDR made from the Random Distributed Scalar Encoder and return a valid value. | decode() methods implemented across encoders (scalar/rdse/date/category/fourier/geospatial) with decoder tests. |
| SF-J-3 | Completed | The encoder tool shall decode an SDR made from the Date Encoder and return a valid value. | decode() methods implemented across encoders (scalar/rdse/date/category/fourier/geospatial) with decoder tests. |
| SF-J-4 | Completed | The encoder tool shall decode an SDR made from the Category Encoder and return a valid value. | decode() methods implemented across encoders (scalar/rdse/date/category/fourier/geospatial) with decoder tests. |
| SF-J-5 | Completed | The encoder tool shall decode an SDR made from the Fourier Encoder and return a valid value. | decode() methods implemented across encoders (scalar/rdse/date/category/fourier/geospatial) with decoder tests. |
| SF-J-6 | Completed | The encoder tool shall decode an SDR made from the Geospatial Encoder and return a valid value. | decode() methods implemented across encoders (scalar/rdse/date/category/fourier/geospatial) with decoder tests. |
| SF-K-1 | Completed | The trainer tool shall use a grapher method to display a 2D matplotlib plot of the on bits in the bit vector. | grapher.plot_sdr() provides 2D matplotlib visualization of on-bit vectors. |

## Totals
- **Completed**: 45
- **Partial**: 8
- **Not Found**: 19
