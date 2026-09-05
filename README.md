<p align="center">
  <img src="wildginger.png" alt="WildGinger logo" width="520">
</p>

# WildGinger

**LLM-assisted source-code security review, with evidence you can inspect.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org)

WildGinger reviews a repository or a single file using configurable language models,
local security rules, and read-only code navigation. It tracks analysis coverage,
validates candidates with a judge quorum, checks whether findings are reachable from
an external entry point, and writes reports for human review or CI.

## Highlights

- **Multiple providers:** AWS Bedrock, OpenAI-compatible APIs, OpenRouter, and a configurable GPT gateway adapter — with per-role model routing.
- **Evidence-driven review:** source citations, symbol-aware chunking, reachability analysis, and deterministic + semantic deduplication.
- **Controlled runs:** cost estimates, hard budget ceilings, checkpoint/resume, and incremental scanning.
- **Practical output:** JSON, SARIF 2.1.0, Markdown, executive summaries, and reproduction sketches (never executed).
- **Review workflow:** baselines, suppressions, and human verdicts.

## Quick Start

Use **Python 3.11**. Keep the checkout directory named `WildGinger` and run the
commands below from its **parent directory**.

```bash
python3.11 -m venv WildGinger/.venv
source WildGinger/.venv/bin/activate
python -m pip install -r WildGinger/requirements.txt
```

On Windows, activate with `WildGinger\.venv\Scripts\Activate.ps1` instead.

Configure [config.yaml](config.yaml) before scanning. It contains **placeholders,
not working credentials or model IDs**. For machine-specific settings, use an
ignored `config.local.yaml` and select it with `--config`.

1. Replace the `YOUR_*` model IDs with models you can access (Bedrock model IDs/ARNs, gateway deployments, or OpenAI-compatible model names).
2. Supply credentials through your environment. Bedrock uses `aws.profile` or the default credential chain (`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`, an instance role, SSO, or `AWS_BEARER_TOKEN_BEDROCK`); set `aws.profile: null` for the chain.
3. For HTTP providers, set `base_url` and a matching entry in `governance.allowed_model_endpoints`. The `openai_gpt` gateway adapter reads `GPT_BASE_URL`, `APIGEE_KEY`, `FM_END_APP_ID`, and `FM_END_USER_ID`; all are required.
4. Store keys in environment variables only. `api_key_env` holds the variable **name**, never a key.
5. Verify model prices, scope, and budgets. Shipped prices are illustrative, not a live pricing feed.

Check configuration, then estimate work — neither command calls a model:

```bash
python -m WildGinger validate --config WildGinger/config.yaml
python -m WildGinger dry-run --config WildGinger/config.yaml --path ./target_repo
```

Run a scan:

```bash
python -m WildGinger scan --config WildGinger/config.yaml \
  --path ./target_repo --output ./wildginger_output --max-cost 5
```

**Model requests cost money.** Enabled preflight makes one small call per model
**before** the confirmation prompt. Budgets use configured prices, and in-flight
requests can overshoot the ceiling.

## Commands And Arguments

Put the subcommand first; options belong to the subcommand. Every subcommand
accepts `-h/--help`; the top level also accepts `--version`.

Config precedence: `--config` > `$WILDGINGER_CONFIG` > `./config.yaml` > the
config shipped with the package. Relative paths resolve from the working directory.

| Command | Arguments |
|---|---|
| `validate` | `--config PATH` |
| `dry-run` | `--config`, `--path`, `--output`, `--hunters N`, `--max-cost USD`, `--limit N`, `--yes`, `--experiment NAME` |
| `scan` | all of `dry-run`'s options, plus `--ci`, `--resume`, `--resume-force` |
| `baseline` | `--config`, `--output DIR`, `--report PATH`, `--out PATH` |
| `mark FINDING_ID` | `--config`, `--result {confirmed,false_positive,accepted_risk,fixed}`, `--notes TEXT` |
| `purge` | `--config`, `--confirm` |

- `--path` accepts a directory **or a single file**. A single-file scan indexes its surrounding directory for context.
- `--hunters N` overrides `concurrency.hunters`; `--max-cost USD` overrides the hard spend ceiling.
- `--limit N` claims at most N cells **per pass** — a cheap smoke test, not a whole-run cap. It also suppresses the incomplete-coverage exit code.
- `--yes` skips the cost confirmation; `--ci` implies `--yes`. Neither disables model calls.
- `--experiment NAME` overrides the MLflow experiment (monitoring stays off unless enabled in config).
- `--resume` continues an interrupted scan from the same output directory; `--resume-force` additionally bypasses the config-hash mismatch refusal.
- `baseline` merges fingerprints from an existing `report.json`; configure `baseline.file` so later scans read it.
- `mark` records a human verdict; `purge` previews deletions and only removes state, audit logs, reports, and PoCs with `--confirm`.

### Exit Codes

| Code | Meaning |
|---|---|
| `0` | Clean — no reportable findings at/above `output.severity_threshold` |
| `1` | Reportable findings (after baseline/suppression filtering) |
| `2` | No reportable findings but coverage incomplete (`output.fail_on_incomplete_coverage: true`) |
| `3` | Fatal — config, preflight, or runtime failure |

Findings outrank coverage failure: a run with findings **and** incomplete coverage
exits `1`. Aborted runs (cost ceiling, interrupt) keep a `1`/`2` result if reports
were finalized; otherwise they exit `3`.

## Results

| Artifact | Contents |
|---|---|
| `report.json` / `report.sarif` | Structured findings for automation and security tooling |
| `report.md` / `executive_summary.md` | Technical review and summary |
| `pocs/` | Reproduction sketches for manual review, never executed by the scanner |
| `scan_state.db` / `audit/` | Progress, resumable state, usage, and audit events |

## Security By Design

WildGinger is deliberately built to operate within tight boundaries. The following
controls are in place by design:

| Control | Detail |
|---|---|
| No general internet access | The only outbound connections are to the model endpoints **you** explicitly approve in `governance.allowed_model_endpoints` (AWS Bedrock uses the AWS SDK and needs no entry). Endpoint matching is strict — scheme, host, port, and path boundary — and plaintext `http://` is refused for public addresses. The tool does not browse the web, download anything, or contact any other service. |
| No arbitrary/unknown software | The tool relies only on a small, pre-reviewable set of dependencies ([requirements.txt](requirements.txt)); it never installs or pulls in software at runtime. |
| Read-only on your code | The scanner only **reads** the target. Review agents get read-only navigation tools — no shell, no write access — and cannot modify or delete the code being reviewed. The tool's only writes are its own state, audit logs, and reports in the output directory. |
| No code execution | The tool never runs, builds, or executes the code it reviews — it analyzes text. PoC artifacts are inert Markdown sketches and are never executed. |
| Runs on your infrastructure | It runs entirely on machines you control, using your own cloud/API credentials. Source code stays under your control except for the model calls you configure and approve. |
| Sensitive-data masking | Obvious secrets (keys, passwords, tokens) are detected and masked before content is sent to a model (`sanitization.mode: redact`, or `block` to refuse outright), and high-risk file types (`.env`, `.pem`, `.key`, `.p12`) are quarantined and never sent at all. |
| Untrusted-input handling | Scanned content is treated as data, never instructions: injection defenses in the system prompt, untrusted-content wrapping of tool output, and an injection guard reduce prompt-injection risk. |
| Full auditability | Every model call, tool call, and result is written to a local audit log (`audit/events.jsonl`), including token usage and cost, so any run can be reviewed after the fact. Output files can be locked to your account (`governance.data_protection.restrict_permissions`). |
| Human-in-the-loop | No automated changes or actions. Findings are suggestions; a person reviews, records verdicts (`mark`), and decides. |
| Bounded usage | Hard ceilings on wall-clock time, USD cost, passes, and per-role turn budgets; the cost ceiling aborts a run mid-flight when reached. |
| Model scope is a setting | Providers and models are pure configuration (Bedrock, OpenAI-compatible, OpenRouter, gateway adapters) — swap in approved models as guidance evolves, with no code changes. |

## Security And Limits

Only scan code you own or are authorized to assess. Hosted models receive selected
source content; review each provider's retention policy before sending private code.
Redaction and prompt-injection defenses reduce risk but cannot guarantee detection.
For a local-only run, route **every active role** to a local model and keep
monitoring disabled.

Findings can be incorrect or incomplete. Coverage measures configured review work,
not proof that a project is secure — review findings and reproduction sketches
before acting on them. Reports, audit logs, and optional telemetry can contain
sensitive data; treat the output directory accordingly.

## License

[Apache License 2.0](LICENSE). Dependencies retain their respective licenses.
