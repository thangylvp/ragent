# Boundary contracts

This directory will contain versioned, language-neutral contracts shared with
the future agent harness and execution layer:

```text
contracts/
├── robot_tools.v1.json          # OpenAI-compatible function definitions
├── required_fields.v1.json      # Harness-owned elicitation requirements
├── execution_constraints.v1.json# Execution-owned ranges and preconditions
└── examples.v1.json             # Valid complete and partial outputs
```

These files are intentionally not fabricated in the scaffold. The existing
gateway contains action and mission names, but not enough typed information to
define production function parameters safely.

Important separation:

- Function descriptions and argument schemas are model inputs.
- Required-field policy belongs to the harness and may be stricter than the
  model-facing JSON schema so the SLM can emit partial calls.
- Live validation, defaults and safety constraints belong to execution.
