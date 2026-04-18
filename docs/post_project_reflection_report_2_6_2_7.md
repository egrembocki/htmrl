# Post-Project Reflection Report (Rubric 7.c and 7.b)

## Project Context
This capstone is primarily a **research-driven ML/HTM pipeline project**, not just an app-delivery project. The current codebase reflects that focus through:
- multiple encoder experiments (RDSE, scalar, date, category, coordinate, geospatial, fourier, delta),
- HTM pipeline integration from input ingestion through brain compute and prediction,
- environment adapters for controlled experimentation,
- and research artifacts (plots, UML, generated reports) used to evaluate behavior and learning quality.

---

## 7.c — Post-Project Reflection: Lessons Learned (4 points / ~1m40s)

### Suggested spoken summary (about 1 minute 40 seconds)
Our biggest lesson learned is that **research software lives or dies by pipeline clarity**. In this project, we learned that strong separation of concerns—input normalization, encoding, environment adaptation, and HTM compute—made experiments reproducible and findings defendable. The `InputHandler` now accepts multiple file and in-memory formats and normalizes them before model processing, which prevented data-shape errors from contaminating HTM conclusions.

A second lesson is that **encoder choice dominates downstream HTM behavior**. Through our overlap and activation-distribution experiments, we observed that RDSE and scalar encoders produce very different sparsity and noise profiles. The codebase now supports multiple encoder types through a factory, so we can run controlled comparisons instead of rewriting plumbing for each test.

Third, we learned that **bridging RL-style environments with HTM requires explicit translation layers**. The environment adapter converts Gym spaces and observations into flat, serializable HTM inputs. Without that adapter boundary, experimentation across environments would have remained brittle and inconsistent.

Fourth, we learned that **observability is part of research quality**. Structured logging, sequence diagrams, and generated test/research reports were not “extra docs”—they were necessary to trace why a behavior occurred and whether a result was trustworthy.

Finally, the project reinforced that in research capstones, success is not only “did the model win,” but also “can we explain and reproduce what happened.” This codebase now supports that scientific workflow significantly better than at project start.

### Evidence from current codebase
- Data ingestion/normalization and validation are centralized in `InputHandler`, including file-type handling and dataframe processing.
- Encoder experimentation is structured via `EncoderFactory` with explicit mappings for multiple encoder families.
- HTM flow orchestration is encapsulated in `Brain.step()` and related encode/compute methods.
- Gym-to-HTM translation is handled by `EnvAdapter` through schema extraction and flattened bridge inputs.
- End-to-end architecture and experiment communication are preserved with sequence diagrams and test/research reports in `docs/`.

---

## 7.b — Post-Project Reflection: Skills We Would NOT Have Learned Otherwise (4 points / ~1m40s)

### Suggested spoken summary (about 1 minute 40 seconds)
If we had not done this capstone, we likely would not have learned how to **engineer a full experimental HTM pipeline** that spans data interfaces, encoding theory, and adaptive control loops. This was different from typical coursework where models are used as fixed black boxes.

First, we developed practical skill in **sparse distributed representation engineering**—including encoder parameterization, tradeoffs in sparsity/noise, and how those design decisions propagate into spatial pooling and prediction quality.

Second, we gained experience designing **extensible architecture for comparative research**, especially with factory-based encoder creation and modular field composition in the brain/trainer layers. That gave us a repeatable way to test hypotheses rather than one-off scripts.

Third, we learned **cross-domain integration**: adapting Gymnasium observations/actions into HTM-compatible inputs, managing episode flow, and exposing a live control channel via WebSocket for visualization and interaction.

Fourth, we built skill in **research-grade software communication**: UML sequence/class modeling, experiment artifact generation, and explicit documentation of training/evaluation outputs so another team member can reproduce results.

Lastly, this project taught us an important capstone-level mindset: combine software engineering discipline with scientific method. We now know how to move from “interesting idea” to “testable, instrumented, and explainable ML system,” which we would not have learned deeply without this research-centered capstone scope.

### Evidence from current codebase
- Encoder breadth and extensibility are visible in the encoder layer and factory mapping.
- Modular HTM control flow appears in brain and trainer components.
- Environment integration and flattening logic are in the Gym adapter layer.
- Real-time interaction patterns are demonstrated by the agent WebSocket server.
- Reproducibility and communication artifacts exist across UML, reports, and experiment visualizations under `docs/` and `test_images/`.

---

## Optional presenter tip
To reliably hit the timing target, deliver each section as:
1. One-sentence thesis,
2. Four concise points,
3. One closing sentence tied to research reproducibility.
