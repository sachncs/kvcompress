# Error policy

Library code separates caller-correctable contract failures from runtime
failures:

- `KVCompressError` subclasses are structured, actionable library errors. They
  carry a stable `code`, message, and optional hint and should be handled at an
  application boundary.
- Invalid shapes, dtypes, devices, method configuration, cache state, and
  compatibility are fatal for the current operation and raise.
- Optional acceleration and integration probes may return unavailable/skip
  results before activation, but an explicitly requested backend raises a typed
  availability or compatibility error.
- Recoverable benchmark/integration failures may be reported by the command
  runner with context; library compression paths must not silently downgrade or
  swallow exceptions.

When adding an error, include the method/layer where available, tensor shape,
dtype, device, and the corrective action. Never catch `BaseException` in
rollback or request handling; process-control exceptions must propagate.
