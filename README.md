# Week 7: Build and defend an end-to-end project

**One module, one week, one repository.** Independent learning material mapped to the published NUS School of Computing SCTP AI Engineering objectives. Not issued, endorsed or accredited by NUS.

## What is ready

- `course.py`: reactive Marimo mechanism lab, responsive SVG charts and controllable motion graphics.
- `build_lab.py`: deeper executable project, not just a reading outline.
- `docs/notebook.html`: browser-runtime version of the core. It downloads Python/packages and needs internet.
- `docs/read.html`: executed read-only core snapshot for phones and browsers that cannot load the runtime.
- `docs/build.html`: executed read-only deeper-project snapshot. Run locally to change its inputs.
- `CURRICULUM.md`, `ASSESSMENT.md`, `LAB_LOG.md`, `SOURCES.md` and `learning_tests/`.
- `TEST_RESULTS.txt`: recorded local tests. This is evidence about this teaching build, not the entire inherited upstream application.

## Run locally

Use Python 3.11 or later. Create a dedicated environment, then install the learning dependencies, not any inherited upstream application's default environment:

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-build.txt
marimo edit course.py
# For the deeper project, open build_lab.py instead.
python -m unittest discover -s learning_tests -v
```

For CPU PyTorch weeks (3 and 4), install a CPU-compatible PyTorch wheel for your operating system. No GPU or model downloads are needed for the supplied synthetic tasks. Browser code is read-only in run mode; use local Marimo edit mode to change Python.

## Week plan

Estimated **20 active hours**, an original self-study estimate. The official module lists 3 classroom day(s); these are different quantities.

- Day 1: scope, dataset card and evaluation plan (3h)
- Day 2: baseline pipeline and user interface (4h)
- Day 3: predictor plus retrieval integration (4h)
- Day 4: evaluation, incident drill and fairness caveats (4h)
- Day 5: clean rerun, report and five-minute demo (5h)

Submit: **Study-support demonstrator, using only synthetic records and a tiny authored corpus**. See ASSESSMENT.md for the original self-study rubric and answer hints.

## Important boundary

Assessment weights, expected capstone size, official datasets and marking rubrics are unpublished in the evidence inspected. The rubric supplied here is an original self-study rubric, not NUS's.

Default lessons make no external API calls, buy no compute and use no private data. Synthetic educational examples must not be used to rank real learners. Upstream live integrations can need paid provider accounts; they are not run automatically.

## Provenance

Published objective anchor: https://www.comp.nus.edu.sg/executive-education/course/sctp-capstone-project-for-ai-engineering/

All newly written lessons and kernels are original. Public student work is linked as evidence, not copied without permission. No NUS logos, private teaching materials, student IDs, emails, credentials or datasets are included in the original curriculum additions.

## Seven-week sequence

- Week 1: [Python Programming for AI Engineering](https://github.com/azjhf-dev/nus-aie-week-01-python)
- Week 2: [Applied Machine Learning](https://github.com/azjhf-dev/nus-aie-week-02-machine-learning)
- Week 3: [Deep Learning and Neural Networks](https://github.com/azjhf-dev/nus-aie-week-03-deep-learning)
- Week 4: [Generative AI and its Applications](https://github.com/azjhf-dev/nus-aie-week-04-generative-ai)
- Week 5: [Large Language Models and Agentic AI](https://github.com/azjhf-dev/nus-aie-week-05-llm-agents)
- Week 6: [Model Deployment and MLOps Essentials](https://github.com/azjhf-dev/nus-aie-week-06-mlops)
- Week 7: [Capstone Project for AI Engineering](https://github.com/azjhf-dev/nus-aie-week-07-capstone)

Checked 19 November 2026.
