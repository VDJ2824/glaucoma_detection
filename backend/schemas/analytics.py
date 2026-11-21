"""Future strict request allowlist: consent, three core scores, weights,
ensemble score, quality summary, model versions, and experimental limitations.
Reject unknown fields, images/base64, free-text identifiers, nonfinite numbers,
invalid scores/weights, unsupported models, and oversized requests.
"""

def validate_analytics(payload):
    raise NotImplementedError("No external requests until the privacy schema is implemented.")
