# Privacy and analytical reporting contract

- Local only: retinal image bytes, image metadata/file names, previews, image tensors, heatmaps and PDF image insertion.
- External only with explicit per-request consent: a strictly validated summary of three core scores, normalized weights, ensemble score, bounded quality measures, model versions and fixed limitation indicators.
- Reject arbitrary nested values, unknown keys, unsupported model names, nonfinite values, invalid probabilities or weights, raw images, URLs, identifiers and unbounded text. Consent must be a true boolean; its absence must fail closed.
- Keep the Gemini key on the Flask server. Environment secrets must never enter Vite bundles, logs, browser storage or report files.
- Do not log request bodies, store images/reports on the server or echo rejected sensitive fields. Define rate limits, authentication where appropriate, timeouts and safe error responses before deployment.
- Consent does not itself guarantee privacy: disclose that approved analytics leave the browser and review the provider's current retention policy before enabling the integration.
- Generated text must remain grounded in supplied scores and quality checks. Include limitations, uncertainty and the research-only, non-diagnostic status. Provider failure must not prevent a local structured PDF.

The current report route returns 501; validators and Gemini integration are placeholders. This document is a future implementation contract, not a claim of completed enforcement.
