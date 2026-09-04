"""Fail-closed helpers for OpenAI-compatible endpoint contracts.

Prospective CATBench runs must bind a runner to the intended local model, not
merely to any HTTP service that happens to answer a health check.  This module
keeps the identity, context-window, and loopback checks consistent across the
five local C1 runners.
"""

from __future__ import annotations

import ipaddress
import json
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from collections.abc import Callable
from urllib import parse


class EndpointContractError(RuntimeError):
  """The endpoint is reachable but violates the frozen runtime contract."""


def models_url(endpoint_url: str) -> str:
  """Returns the OpenAI-compatible model-list URL for ``endpoint_url``."""
  base = endpoint_url.rstrip("/")
  if base.endswith("/models"):
    return base
  if base.endswith("/v1"):
    return f"{base}/models"
  return f"{base}/v1/models"


def require_loopback(endpoint_url: str) -> None:
  """Raises unless an HTTP(S) endpoint is explicitly bound via loopback."""
  parsed = parse.urlparse(endpoint_url)
  if parsed.scheme not in {"http", "https"}:
    raise EndpointContractError(
        "Endpoint URL must use http or https; "
        f"received scheme={parsed.scheme!r}."
    )
  if parsed.username is not None or parsed.password is not None:
    raise EndpointContractError("Endpoint URL must not contain user info.")
  if parsed.query or parsed.fragment:
    raise EndpointContractError(
        "Endpoint URL must not contain a query string or fragment."
    )
  host = parsed.hostname
  if not host:
    raise EndpointContractError("Endpoint URL has no hostname.")
  normalized_host = host.lower().rstrip(".")
  if normalized_host == "localhost":
    return
  try:
    address = ipaddress.ip_address(normalized_host)
  except ValueError as exc:
    raise EndpointContractError(
        "Prospective local-model endpoints must use an explicit loopback host; "
        f"received {host!r}."
    ) from exc
  if not address.is_loopback:
    raise EndpointContractError(
        "Prospective local-model endpoints must be loopback-only; "
        f"received {host!r}."
    )


def parse_model_records(payload: object) -> dict[str, dict[str, object]]:
  """Parses a ``/v1/models`` response without accepting ambiguous records."""
  if not isinstance(payload, Mapping):
    raise EndpointContractError("Endpoint /v1/models response is not an object.")
  data = payload.get("data")
  if not isinstance(data, list):
    raise EndpointContractError(
        "Endpoint /v1/models response does not contain a data list."
    )
  records: dict[str, dict[str, object]] = {}
  for index, item in enumerate(data):
    if not isinstance(item, Mapping):
      raise EndpointContractError(
          f"Endpoint model record {index} is not an object."
      )
    model_id = item.get("id")
    if not isinstance(model_id, str) or not model_id:
      raise EndpointContractError(
          f"Endpoint model record {index} has no non-empty string id."
      )
    if model_id in records:
      raise EndpointContractError(
          f"Endpoint reports duplicate model id {model_id!r}."
      )
    records[model_id] = dict(item)
  return records


def fetch_model_records(
    endpoint_url: str, timeout_sec: float = 5.0
) -> dict[str, dict[str, object]]:
  """Fetches and parses the endpoint's advertised model records."""
  request = urllib.request.Request(
      models_url(endpoint_url), headers={"Accept": "application/json"}
  )
  with urllib.request.urlopen(request, timeout=timeout_sec) as response:
    payload = json.loads(response.read().decode("utf-8"))
  return parse_model_records(payload)


def context_compatible(record: Mapping[str, object], minimum_context: int) -> bool:
  """Whether a ``/v1/models`` record satisfies the pinned context floor."""
  if minimum_context <= 0:
    return True
  reported_context = record.get("max_model_len")
  return (
      isinstance(reported_context, int)
      and not isinstance(reported_context, bool)
      and reported_context >= minimum_context
  )


def validate_model_record(
    records: Mapping[str, Mapping[str, object]],
    expected_model: str,
    minimum_context: int,
) -> Mapping[str, object]:
  """Returns the exact model record or raises on identity/context mismatch."""
  if not expected_model:
    raise EndpointContractError("Expected endpoint model id must be non-empty.")
  record = records.get(expected_model)
  if record is None:
    raise EndpointContractError(
        f"Endpoint does not serve exact model id {expected_model!r}; "
        f"advertised ids={sorted(records)}."
    )
  if not context_compatible(record, minimum_context):
    raise EndpointContractError(
        f"Endpoint model {expected_model!r} reports max_model_len="
        f"{record.get('max_model_len')!r}; requires an integer >= "
        f"{minimum_context}."
    )
  return record


def wait_for_model(
    endpoint_url: str,
    expected_model: str,
    minimum_context: int,
    *,
    timeout_sec: float,
    poll_sec: float,
    loopback_only: bool,
    status_callback: Callable[[str], None] | None = None,
) -> Mapping[str, object]:
  """Waits for readiness, then validates exact identity and context.

  Connection failures and an empty model list are treated as transient startup
  states.  A non-empty but incompatible model list is a configuration error and
  fails immediately instead of silently waiting against the wrong server.
  """
  if minimum_context < 0:
    raise EndpointContractError("Minimum context length cannot be negative.")
  if loopback_only:
    require_loopback(endpoint_url)
  deadline = time.monotonic() + max(0.0, timeout_sec)
  last_status = "endpoint not reachable yet"
  while True:
    try:
      records = fetch_model_records(endpoint_url, timeout_sec=5.0)
      if records:
        return validate_model_record(records, expected_model, minimum_context)
      last_status = "endpoint responded but advertised no models"
    except EndpointContractError:
      raise
    except (OSError, urllib.error.URLError, ValueError) as exc:
      last_status = f"endpoint check error: {exc}"
    if time.monotonic() >= deadline:
      raise EndpointContractError(
          "Endpoint did not become ready before timeout. "
          f"Expected exact model={expected_model!r} at "
          f"{models_url(endpoint_url)} with minimum context="
          f"{minimum_context}. Last status: {last_status}"
      )
    if status_callback is not None:
      status_callback(last_status)
    time.sleep(max(0.05, poll_sec))
