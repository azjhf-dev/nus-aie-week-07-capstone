# Verification of the independent learning build

Checked 19 November 2026.

- Seven separate repositories; week 5 is a genuine JobAId fork with the MIT notice preserved.
- Fourteen Marimo notebooks executed successfully to HTML: seven mechanism labs and seven deeper project labs.
- 133 deterministic tests passed across the learning_tests directories. These are 17 shared numerical/validation tests per repo plus module-specific tests, not 133 distinct topic objectives.
- Actual CNN, RNN and Transformer training checked on synthetic held-out tasks. Actual VAE, 2D DDPM and masked/causal objective training checked for finite outputs and learning. These are not pretrained-model benchmarks.
- Actual LangGraph review rejection/resume checked. Actual local HTTP health/prediction/invalid-input checks passed.
- Every published core browser notebook loaded, a slider value was changed, output text changed, and its conceptual animation paused. No browser script errors were observed in those checks.
- All fourteen read-only snapshots were visually inspected at 390px width. Core live views were also captured at 320px and 1280px. Tables can scroll horizontally. The static export warns that its controls cannot rerun Python; use the interactive core link or run locally.
- Landing-page links and publication state were checked without sign-in.

## Limits

This is not a full audit of inherited JobAId code, upstream dependency security, all devices, private NUS assessments or a production deployment. Static pages do not provide a server-side inference API. Browser runtime downloads need internet and can be slow or memory-limited on phones. No keys, private data, paid APIs or compute purchases are needed for the supplied default lessons.
