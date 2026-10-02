## Mission
This workspace is the persistent local study environment for PGSR. Its primary purpose is to build an independent mental
model of the method by tracing source code, inspecting tensors, running short controlled experiments, and maintaining
durable notes that remain available even when remote servers are offline.
The local workspace is not the default place for full-paper benchmark reproduction. It is the place for understanding,
instrumentation, toy experiments, and synthesis.
## Primary Goals
1. Trace the end-to-end data path from dataset loading to camera sampling, Gaussian state, rendering outputs, losses,
backpropagation, optimizer updates, densification, rendering, depth output, and mesh extraction.
2. Maintain a paper-to-code map for both the official PGSR repository and the 2026 refactored Python package.
3. Use short experiments to verify hypotheses about code behavior rather than relying on static reading alone.
4. Produce durable artifacts: tensor notes, call graphs, data-flow diagrams, experiment records, and oral-explanation
outlines.
5. Preserve the official baseline and clearly separate upstream code from study instrumentation.
## Workspace Roles
- `upstream/PGSR-official/`: clean reference clone of the official `zju3dv/PGSR` repository. Do not make casual edits here.
- `upstream/PGSR-python/`: clean reference clone of the refactored Python package repository.
- `worktrees/`: study branches or editable copies used for breakpoints, logging, probes, and temporary instrumentation.
- `data/`: only the small benchmark scenes and toy inputs needed for learning.
- `notes/`: human-readable understanding of the system. Notes are first-class outputs, not scratch material.
- `outputs/`: local short-run artifacts only. Large benchmark outputs belong on the 4090 server.
## Local Compute Policy
The local GPU is an RTX 3050. Treat GPU memory and local disk as constrained resources.
Default local execution policy:
1. Prefer import tests, dataset inspection, single-forward-pass tests, and short runs.
2. Prefer the refactored Python package for debugger-driven learning because it exposes modular preparation APIs and a
`base` mode without densification.
3. Do not start a full 30k-iteration benchmark run locally unless the user explicitly requests it.
4. Before any GPU run, inspect available memory with `nvidia-smi`.
5. If an out-of-memory error occurs, reduce image resolution, iteration count, active features, or scene size. Do not hide
the failure by silently changing algorithmic settings that affect the learning objective.
6. Never assume that simply deleting images from a COLMAP dataset creates a valid subset. Camera/image metadata must remain
consistent.
## Baseline Integrity
- Keep a clean upstream clone for each implementation.
- Put instrumentation on a dedicated branch such as `study/local-trace` or in a separate worktree.
- Before changing code, record the current commit with `git rev-parse HEAD`.
- Do not mix algorithmic changes with instrumentation changes in the same commit.
- If a code modification changes outputs, label it as an experiment rather than a baseline reproduction.
- Never report a benchmark number as a PGSR result unless the exact command, commit, environment, dataset, and evaluation
path are recorded.
## Learning Protocol
For every important module, answer all of the following in `notes/`:
1. What enters this module?
2. What leaves this module?
3. What are the important tensor shapes, coordinate frames, and units?
4. Which PGSR paper concept does it implement?
5. What behavior is inherited from vanilla 3DGS or the reusable Gaussian-splatting package?
6. What changes during backpropagation?
7. What changes only during densification, trimming, pruning, or opacity reset?
8. How can one short experiment falsify an incorrect interpretation?
When tracing code, prefer this order:
1. CLI / entry point.
2. Dataset and camera preparation.
3. Gaussian initialization and parameter containers.
4. One rendering forward pass.
5. Rendered RGB / normal / distance / depth outputs.
6. Loss construction.
7. `backward()` and optimizer update.
8. Densification and pruning.
9. Rendering for evaluation.
10. Mesh extraction.
## Experiment Discipline
Each local experiment should have a compact record containing:
- date and experiment ID;
- purpose / hypothesis;
- repository and commit;
- environment name;
- dataset and scene;
- exact command;
- iteration range;
- key configuration overrides;
- GPU and peak memory if relevant;
- observed outputs;
- conclusion;
- next question.
Short runs are allowed to be intentionally non-converged. Their purpose must be stated explicitly, for example: "trace one
training iteration" or "verify whether plane-depth participates in the geometry loss."
## Data and Disk Policy
Local storage is the durable source of truth for study-critical material, not for every generated artifact.
Keep locally:
- both source repositories;
- the Truck scene used by the refactored package quick start;
- DTU scan24 and its evaluation assets once needed;
- small toy datasets;
- notes, diagrams, scripts, configs, logs, and selected checkpoints;
- server experiment manifests and final meshes needed for comparison.
Do not keep locally by default:
- redundant full benchmark suites;
- every checkpoint from every run;
- large temporary render dumps;
- duplicate environment caches that can be recreated.
Maintain at least 20-30 GB of free disk headroom so local experiments and package builds do not fail unpredictably.
## Evidence and Reporting Rules
- Distinguish observed behavior from inferred behavior.
- When explaining code, cite file paths, function/class names, and relevant call sites.
- When explaining runtime behavior, cite the command and log or tensor inspection that supports the claim.
- Do not invent experiment results. If a run was not executed, label the statement as an expectation or proposed test.
- Prefer one verified diagram over multiple speculative diagrams.
## 文档语言
- 实验报告、张量与数据通路笔记、论文—代码映射、源码审计及其他学习产出文档一律用中文撰写。
- 代码标识符、命令、路径、数学符号和必要的论文原词保留原文，并在中文正文中解释其含义，以便核对源码。
- 修改已有英文学习文档时，将所修改的整份文档转为中文，不只在末尾追加中文附录。
## Multi-Agent Policy
Use subagents by default for non-trivial tasks. The root agent is the coordinator and must route work intentionally rather
than spawning agents indiscriminately.
Routing policy:
- Top-level design, research strategy, architecture decisions, experiment design, and final global audit: use GPT-6 Sol with
`xhigh` reasoning.
- Execution work, code tracing, implementation, experiment preparation, and detailed evidence collection: use GPT-6 Luna
with `xhigh` reasoning.
- Simple file I/O, mechanical transformations, repetitive formatting, log collation, and final text emission when the
structure is already fixed: use GPT-5.6 Sol with `xhigh` reasoning.
Example for an experiment report or data-flow report:
1. GPT-6 Sol defines the outline, claims to verify, and acceptance criteria.
2. GPT-6 Luna fills the outline with code evidence, commands, tensors, experiment observations, and implementation details.
3. GPT-5.6 Sol performs mechanical consolidation, formatting, cross-reference cleanup, and final text emission without
changing technical conclusions.
4. GPT-6 Sol performs a final global consistency audit when the report is technically consequential.
Do not use multiple agents when the task is trivial enough that delegation would cost more context than it saves.
## User Override: Single-Agent Mode
If the user says "do not use subagents", "single model only", "no delegation", or any equivalent instruction, do not spawn
any subagent for that task. Complete the task with the model and reasoning level selected by the user in the current
interface. This user instruction overrides the default multi-agent policy.
## Safety and System Hygiene
- Do not delete datasets, checkpoints, or notes without explicit user approval.
- Do not run destructive Git commands such as `reset --hard`, `clean -fdx`, or history rewrites unless explicitly requested.
- Do not modify system-wide CUDA, drivers, or global Python installations for a local study task.
- Prefer isolated environments.
- Before a large download, check free disk space.
## Definition of Done for Local Learning Tasks
A local learning task is complete only when the result can be explained without relying on the agent conversation. The
workspace should contain enough evidence that the user can later answer:
- what the module does;
- where it is implemented;
- what data enters and leaves it;
- how it connects to adjacent modules;
- which observation or experiment verified the interpretation.
