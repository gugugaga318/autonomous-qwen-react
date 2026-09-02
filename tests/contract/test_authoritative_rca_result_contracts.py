from __future__ import annotations

import json
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core"))

from yield_rca_core.authoritative_result import (  # noqa: E402
    AuthoritativeRCAResult,
    ImpactPublicationResult,
)
from yield_rca_core.evidence_models import (  # noqa: E402
    Evidence,
    EvidenceSourceType,
    ModelValidationError,
)
from yield_rca_core.models import RCAJob, RCAState  # noqa: E402


def _authority(
    *,
    result_id: str = "RCA_RESULT_01",
    evidence_refs: tuple[str, ...] = ("EV_AUTHORITY_01",),
) -> AuthoritativeRCAResult:
    return AuthoritativeRCAResult(
        result_id=result_id,
        source_finding_id=None,
        source_hypothesis_id=None,
        conclusion_status="supported",
        root_cause_candidate_id="CANDIDATE_01",
        root_cause="Chamber pressure-control drift caused the observed defect excursion.",
        confirmation_status="supported",
        competition_status="complete_confirmed",
        terminal_reason="confirmation_gate_supported_unique_candidate",
        evidence_refs=evidence_refs,
    )


def _impact(
    *,
    rca_result_id: str = "RCA_RESULT_01",
) -> ImpactPublicationResult:
    return ImpactPublicationResult(
        rca_result_id=rca_result_id,
        publication_status="confirmed",
        confirmed_impact_lots=("LOT_IMPACT_01",),
        evidence_refs=("EV_AUTHORITY_01",),
    )


def _evidence() -> Evidence:
    return Evidence(
        evidence_id="EV_AUTHORITY_01",
        source_type=EvidenceSourceType.SYSTEM.value,
        source_id="SRC_AUTHORITY_01",
        summary="Deterministic evidence supporting the authority contract.",
    )


def test_authoritative_result_is_immutable_json_safe_and_round_trips() -> None:
    result = _authority()

    with pytest.raises(FrozenInstanceError):
        result.root_cause = "mutated"  # type: ignore[misc]

    payload = result.to_dict()
    assert payload["evidence_refs"] == ["EV_AUTHORITY_01"]
    json.dumps(payload)
    assert AuthoritativeRCAResult.from_dict(payload) == result


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"root_cause_candidate_id": None}, "root_cause_candidate_id"),
        ({"root_cause": None}, "root_cause"),
        ({"confirmation_status": "inconclusive"}, "confirmation_status"),
        ({"competition_status": "exhausted"}, "competition_status"),
        ({"evidence_refs": ()}, "evidence_refs"),
    ],
)
def test_supported_authority_requires_publishable_confirmed_content(
    overrides: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "result_id": "RCA_RESULT_01",
        "source_finding_id": None,
        "source_hypothesis_id": None,
        "conclusion_status": "supported",
        "root_cause_candidate_id": "CANDIDATE_01",
        "root_cause": "A supported evidence-bounded root cause.",
        "confirmation_status": "supported",
        "competition_status": "complete_confirmed",
        "terminal_reason": "confirmation_gate_supported_unique_candidate",
        "evidence_refs": ("EV_AUTHORITY_01",),
    }
    values.update(overrides)

    with pytest.raises(ModelValidationError, match=message):
        AuthoritativeRCAResult(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("conclusion_status", ["inconclusive", "insufficient_evidence"])
def test_non_supported_authority_cannot_publish_a_leading_candidate(
    conclusion_status: str,
) -> None:
    with pytest.raises(ModelValidationError, match="must not publish root_cause"):
        AuthoritativeRCAResult(
            result_id="RCA_RESULT_01",
            source_finding_id=None,
            source_hypothesis_id=None,
            conclusion_status=conclusion_status,  # type: ignore[arg-type]
            root_cause_candidate_id="CANDIDATE_01",
            root_cause="A leading but unconfirmed candidate.",
            confirmation_status=conclusion_status,
            competition_status="exhausted",
            terminal_reason="no_high_value_action_remains",
            evidence_refs=("EV_AUTHORITY_01",),
        )


def test_inconclusive_authority_preserves_evidence_but_withholds_impact_lots() -> None:
    authority = AuthoritativeRCAResult(
        result_id="RCA_RESULT_INCONCLUSIVE",
        source_finding_id=None,
        source_hypothesis_id=None,
        conclusion_status="inconclusive",
        root_cause_candidate_id=None,
        root_cause=None,
        confirmation_status="inconclusive",
        competition_status="exhausted",
        terminal_reason="no_high_value_action_remains",
        evidence_refs=("EV_AUTHORITY_01",),
    )
    publication = ImpactPublicationResult(
        rca_result_id=authority.result_id,
        publication_status="withheld",
        confirmed_impact_lots=(),
        evidence_refs=("EV_AUTHORITY_01",),
    )
    state = RCAState(
        job=RCAJob(job_id="JOB_INCONCLUSIVE", user_query="Preserve conservative state."),
        evidence=[_evidence()],
        authoritative_rca_result=authority,
        impact_publication_result=publication,
    )

    restored = RCAState.from_dict(state.to_dict())

    assert restored.authoritative_rca_result == authority
    assert restored.impact_publication_result == publication
    assert restored.impact_publication_result.confirmed_impact_lots == ()


def test_authority_rejects_unknown_status_blank_ids_and_duplicate_evidence_refs() -> None:
    with pytest.raises(ModelValidationError, match="conclusion_status"):
        AuthoritativeRCAResult(
            result_id="RCA_RESULT_01",
            source_finding_id=None,
            source_hypothesis_id=None,
            conclusion_status="candidate",  # type: ignore[arg-type]
            root_cause_candidate_id=None,
            root_cause=None,
            confirmation_status="not_evaluated",
            competition_status="not_evaluated",
            terminal_reason="investigation_incomplete",
            evidence_refs=(),
        )

    with pytest.raises(ModelValidationError, match="result_id"):
        AuthoritativeRCAResult(
            result_id=" ",
            source_finding_id=None,
            source_hypothesis_id=None,
            conclusion_status="inconclusive",
            root_cause_candidate_id=None,
            root_cause=None,
            confirmation_status="inconclusive",
            competition_status="exhausted",
            terminal_reason="no_high_value_action_remains",
            evidence_refs=(),
        )

    with pytest.raises(ModelValidationError, match="duplicates"):
        _authority(evidence_refs=("EV_AUTHORITY_01", "EV_AUTHORITY_01"))


def test_impact_publication_requires_confirmed_scope_and_evidence() -> None:
    publication = _impact()
    payload = publication.to_dict()

    assert payload["confirmed_impact_lots"] == ["LOT_IMPACT_01"]
    json.dumps(payload)
    assert ImpactPublicationResult.from_dict(payload) == publication

    with pytest.raises(ModelValidationError, match="confirmed_impact_lots"):
        ImpactPublicationResult(
            rca_result_id="RCA_RESULT_01",
            publication_status="confirmed",
            confirmed_impact_lots=(),
            evidence_refs=("EV_AUTHORITY_01",),
        )

    with pytest.raises(ModelValidationError, match="evidence_refs"):
        ImpactPublicationResult(
            rca_result_id="RCA_RESULT_01",
            publication_status="confirmed",
            confirmed_impact_lots=("LOT_IMPACT_01",),
            evidence_refs=(),
        )

    with pytest.raises(ModelValidationError, match="must not contain confirmed lots"):
        ImpactPublicationResult(
            rca_result_id="RCA_RESULT_01",
            publication_status="withheld",
            confirmed_impact_lots=("LOT_IMPACT_01",),
            evidence_refs=("EV_AUTHORITY_01",),
        )


def test_rca_state_round_trips_typed_authority_and_publication() -> None:
    state = RCAState(
        job=RCAJob(job_id="JOB_AUTHORITY_01", user_query="Audit the RCA authority."),
        evidence=[_evidence()],
        authoritative_rca_result=_authority(),
        impact_publication_result=_impact(),
    )

    payload = state.to_dict()
    json.dumps(payload)
    restored = RCAState.from_dict(payload)

    assert restored == state
    assert isinstance(restored.authoritative_rca_result, AuthoritativeRCAResult)
    assert isinstance(restored.impact_publication_result, ImpactPublicationResult)


def test_legacy_state_without_new_authority_fields_remains_compatible() -> None:
    payload = RCAState(
        job=RCAJob(job_id="JOB_LEGACY_01", user_query="Read an old State payload.")
    ).to_dict()
    payload.pop("authoritative_rca_result", None)
    payload.pop("impact_publication_result", None)

    restored = RCAState.from_dict(payload)

    assert restored.authoritative_rca_result is None
    assert restored.impact_publication_result is None
    serialized = restored.to_dict()
    assert "authoritative_rca_result" not in serialized
    assert "impact_publication_result" not in serialized
    assert RCAState.from_dict(serialized) == restored


def test_rca_state_rejects_untraceable_authority_or_detached_publication() -> None:
    with pytest.raises(ModelValidationError, match="unknown evidence_ids"):
        RCAState(
            job=RCAJob(job_id="JOB_UNKNOWN_EV", user_query="Reject unknown authority refs."),
            authoritative_rca_result=_authority(),
        )

    with pytest.raises(ModelValidationError, match="requires authoritative_rca_result"):
        RCAState(
            job=RCAJob(job_id="JOB_DETACHED_IMPACT", user_query="Reject detached impact."),
            evidence=[_evidence()],
            impact_publication_result=_impact(),
        )

    with pytest.raises(ModelValidationError, match="rca_result_id must match"):
        RCAState(
            job=RCAJob(job_id="JOB_WRONG_AUTHORITY", user_query="Reject wrong authority."),
            evidence=[_evidence()],
            authoritative_rca_result=_authority(),
            impact_publication_result=_impact(rca_result_id="RCA_RESULT_OTHER"),
        )

    with pytest.raises(ModelValidationError, match="source_finding_id references"):
        RCAState(
            job=RCAJob(job_id="JOB_UNKNOWN_FINDING", user_query="Reject unknown source."),
            evidence=[_evidence()],
            authoritative_rca_result=replace(
                _authority(),
                source_finding_id="FINDING_UNKNOWN",
            ),
        )


def test_rca_state_rejects_confirmed_impact_for_inconclusive_authority() -> None:
    authority = AuthoritativeRCAResult(
        result_id="RCA_RESULT_INCONCLUSIVE",
        source_finding_id=None,
        source_hypothesis_id=None,
        conclusion_status="inconclusive",
        root_cause_candidate_id=None,
        root_cause=None,
        confirmation_status="inconclusive",
        competition_status="exhausted",
        terminal_reason="no_high_value_action_remains",
        evidence_refs=("EV_AUTHORITY_01",),
    )

    with pytest.raises(ModelValidationError, match="requires a supported"):
        RCAState(
            job=RCAJob(job_id="JOB_UNSAFE_IMPACT", user_query="Reject unsafe impact."),
            evidence=[_evidence()],
            authoritative_rca_result=authority,
            impact_publication_result=_impact(rca_result_id=authority.result_id),
        )
