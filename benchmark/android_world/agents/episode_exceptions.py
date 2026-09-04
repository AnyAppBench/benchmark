"""Typed exceptions and attribution for agent-output failures.

Only exceptions deriving from :class:`DeclaredAgentOutputError` are eligible
to become scored agent failures, and only when they escape during the agent
stage. Generic ``ValueError``/``KeyError``/JSON errors are intentionally not
included: those types are also raised by runner, environment, and evaluator
bugs and therefore cannot establish model responsibility.
"""

from __future__ import annotations

import dataclasses
from typing import Literal


EpisodeStage = Literal["initialize", "agent", "verifier", "teardown"]
AttributionKind = Literal[
    "agent_output_parse_or_malformed_action",
    "environment_or_evaluator",
    "unknown",
]


class DeclaredAgentOutputError(ValueError):
  """Base type for a model response that cannot form an executable action."""

  failure_code = "declared_agent_output_error"


class ActionParseError(DeclaredAgentOutputError):
  """The model response cannot be parsed into the declared action schema."""

  failure_code = "action_parse_error"


class MalformedActionError(DeclaredAgentOutputError):
  """The parsed response violates the declared executable-action schema."""

  failure_code = "malformed_action_error"


class AgentInfrastructureError(RuntimeError):
  """Base type for an agent-side service/runtime fault, never model output."""

  failure_code = "agent_infrastructure_error"


class ModelEndpointError(AgentInfrastructureError):
  """The configured inference endpoint failed to return a usable response."""

  failure_code = "model_endpoint_error"


class EnvironmentInfrastructureError(RuntimeError):
  """Base type for a device/runtime fault, never a behavioral outcome."""

  failure_code = "environment_infrastructure_error"


class EmulatorRuntimeHealthError(EnvironmentInfrastructureError):
  """The emulator became unhealthy while an episode was in progress."""

  failure_code = "emulator_runtime_health_error"


class StagedEpisodeError(Exception):
  """Carries a nested callback error across a stage-obscuring API boundary."""

  def __init__(self, stage: EpisodeStage, cause: BaseException):
    super().__init__(str(cause))
    self.stage = stage
    self.cause = cause


@dataclasses.dataclass(frozen=True)
class ExceptionAttribution:
  """Machine-readable attribution for one escaped episode exception."""

  stage: EpisodeStage
  attribution: AttributionKind
  valid_agent_failure: bool
  declared_agent_output_exception: bool
  exception_type: str
  failure_code: str | None

  def as_episode_fields(self) -> dict[str, object]:
    return {
        "catbench_exception_stage": self.stage,
        "catbench_exception_attribution": self.attribution,
        "catbench_exception_valid_agent_failure": self.valid_agent_failure,
        "catbench_exception_declared_agent_output": (
            self.declared_agent_output_exception
        ),
        "catbench_exception_type": self.exception_type,
        "catbench_exception_failure_code": self.failure_code,
    }


def attribute_exception(
    error: BaseException,
    stage: EpisodeStage,
    *,
    known_environment_or_evaluator: bool = False,
) -> ExceptionAttribution:
  """Classifies an escaped exception without message/name heuristics."""
  if stage not in {"initialize", "agent", "verifier", "teardown"}:
    raise ValueError(f"Unsupported episode exception stage: {stage!r}")

  declared = isinstance(error, DeclaredAgentOutputError)
  agent_infrastructure = isinstance(error, AgentInfrastructureError)
  environment_infrastructure = isinstance(
      error, EnvironmentInfrastructureError
  )
  valid_agent_failure = (
      stage == "agent" and declared and not known_environment_or_evaluator
  )
  if valid_agent_failure:
    attribution: AttributionKind = "agent_output_parse_or_malformed_action"
  elif (
      known_environment_or_evaluator
      or agent_infrastructure
      or environment_infrastructure
      or stage in {"initialize", "verifier", "teardown"}
  ):
    attribution = "environment_or_evaluator"
  else:
    attribution = "unknown"

  error_type = type(error)
  failure_code = None
  if declared or agent_infrastructure or environment_infrastructure:
    failure_code = str(getattr(error, "failure_code", "")) or None
  return ExceptionAttribution(
      stage=stage,
      attribution=attribution,
      valid_agent_failure=valid_agent_failure,
      declared_agent_output_exception=declared,
      exception_type=f"{error_type.__module__}.{error_type.__qualname__}",
      failure_code=failure_code,
  )
