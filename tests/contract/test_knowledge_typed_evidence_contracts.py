from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core"))

from yield_rca_core import (  # noqa: E402
    AgentKind,
    EntityType,
    EvidenceType,
    ModelValidationError,
    ToolInput,
    ToolOutput,
)
from yield_rca_core.repositories import CsvFabRepository, Row  # noqa: E402
from yield_rca_core.tool_layer import RetrieveSimilarCaseTool  # noqa: E402

GOLDEN_SEED_DIR = ROOT / "data" / "seeds" / "golden_case"


def knowledge_input(
    request_id: str,
    *,
    query: str,
    module: str = "",
    equipment_type: str = "",
    requested_by: str = AgentKind.KNOWLEDGE.value,
) -> ToolInput:
    return ToolInput(
        tool_name="retrieve_similar_case",
        request_id=request_id,
        parameters={
            "query": query,
            "module": module,
            "equipment_type": equipment_type,
        },
        requested_by=requested_by,
    )


class UnconfirmedKnowledgeRepository:
    def rows(self, table_name: str) -> list[Row]:
        tables: dict[str, list[Row]] = {
            "rca_case": [
                {
                    "case_id": "RCA_DRAFT_001",
                    "title": "Unapproved CMP finding",
                    "module": "Cu CMP",
                    "equipment_type": "CMP",
                    "symptom": "slurry flow",
                    "root_cause": "draft root cause",
                    "solution": "draft action",
                    "confidence": "0.99",
                    "created_at": "2026-07-01T00:00:00+00:00",
                    "validation_status": "DRAFT",
                }
            ],
            "knowledge_document": [],
        }
        return [dict(row) for row in tables[table_name]]


class MixedDocumentApprovalRepository:
    def rows(self, table_name: str) -> list[Row]:
        tables: dict[str, list[Row]] = {
            "rca_case": [
                {
                    "case_id": "RCA_CONFIRMED_001",
                    "title": "Confirmed Cu CMP slurry issue",
                    "module": "Cu CMP",
                    "equipment_type": "CMP",
                    "symptom": "slurry flow decline and scratch",
                    "root_cause": "slurry delivery degradation",
                    "solution": "inspect slurry delivery",
                    "confidence": "0.90",
                    "created_at": "2026-06-01T00:00:00+00:00",
                    "validation_status": "CONFIRMED",
                }
            ],
            "knowledge_document": [
                {
                    "document_id": "DOC_CONFIRMED_001",
                    "case_id": "RCA_CONFIRMED_001",
                    "document_type": "RCA_CASE",
                    "title": "Approved RCA",
                    "content": "Approved engineering RCA record.",
                    "tags": "CMP;slurry",
                    "created_at": "2026-06-01T00:00:00+00:00",
                    "validation_status": "CONFIRMED",
                },
                {
                    "document_id": "DOC_DRAFT_001",
                    "case_id": "RCA_CONFIRMED_001",
                    "document_type": "ENGINEERING_NOTE",
                    "title": "Draft note",
                    "content": "Unapproved draft content.",
                    "tags": "draft",
                    "created_at": "2026-06-02T00:00:00+00:00",
                    "validation_status": "DRAFT",
                },
            ],
        }
        return [dict(row) for row in tables[table_name]]


class KnowledgeTypedEvidenceContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.golden_repository = CsvFabRepository(GOLDEN_SEED_DIR)

    def assert_typed_knowledge_output(self, output: ToolOutput) -> None:
        self.assertTrue(output.evidence)
        self.assertEqual(output.evidence_ids, [item.evidence_id for item in output.evidence])
        self.assertEqual(
            output.data["evidence"],
            [item.to_dict() for item in output.evidence],
        )
        for evidence in output.evidence:
            self.assertTrue(evidence.is_typed)
            self.assertEqual(evidence.source_agent, AgentKind.KNOWLEDGE.value)
            self.assertEqual(evidence.source_tool, "retrieve_similar_case")
            self.assertTrue(evidence.entities)
            self.assertIsNotNone(evidence.confidence)

    def test_confirmed_historical_case_returns_typed_match_evidence(self) -> None:
        output = RetrieveSimilarCaseTool(self.golden_repository).run(
            knowledge_input(
                "REQ_TYPED_KNOWLEDGE_MATCH",
                query="Cu CMP slurry flow scratch leakage",
                module="Cu CMP",
                equipment_type="CMP",
            )
        )

        self.assert_typed_knowledge_output(output)
        evidence = output.evidence[0]
        self.assertEqual(
            evidence.evidence_type,
            EvidenceType.HISTORICAL_CASE_MATCH.value,
        )
        self.assertEqual(evidence.confidence, output.data["top_case"]["similarity"])
        self.assertTrue(
            any(
                entity.entity_type == EntityType.KNOWLEDGE_ASSET.value
                and entity.entity_id == "RCA_CMP_2025_032"
                and entity.attributes["validation_status"] == "CONFIRMED"
                for entity in evidence.entities
            )
        )
        self.assertTrue(
            any(
                entity.entity_type == EntityType.KNOWLEDGE_ASSET.value
                and entity.entity_id == "DOC_RCA_CMP_2025_032"
                for entity in evidence.entities
            )
        )

    def test_available_source_without_confirmed_match_is_negative_signal(self) -> None:
        output = RetrieveSimilarCaseTool(UnconfirmedKnowledgeRepository()).run(
            knowledge_input(
                "REQ_TYPED_KNOWLEDGE_UNCONFIRMED",
                query="CMP slurry flow",
                module="Cu CMP",
            )
        )

        self.assert_typed_knowledge_output(output)
        evidence = output.evidence[0]
        self.assertEqual(evidence.evidence_type, EvidenceType.NEGATIVE_SIGNAL.value)
        self.assertEqual(evidence.metadata["retrieval_status"], "no_match")
        self.assertIs(evidence.metadata["source_available"], True)
        self.assertEqual(output.data["cases"], [])
        self.assertIsNone(output.data["top_case"])
        self.assertIn(
            "WARN_KNOWLEDGE_NO_CONFIRMED_CASE",
            {warning.warning_id for warning in output.warnings},
        )
        self.assertEqual(output.warnings[0].evidence_ids, [evidence.evidence_id])

    def test_unconfirmed_documents_are_excluded_from_output_and_entities(self) -> None:
        output = RetrieveSimilarCaseTool(MixedDocumentApprovalRepository()).run(
            knowledge_input(
                "REQ_TYPED_KNOWLEDGE_DOCUMENT_GOVERNANCE",
                query="Cu CMP slurry flow",
                module="Cu CMP",
                equipment_type="CMP",
            )
        )

        self.assert_typed_knowledge_output(output)
        self.assertEqual(
            [document["document_id"] for document in output.data["documents"]],
            ["DOC_CONFIRMED_001"],
        )
        entity_ids = {entity.entity_id for entity in output.evidence[0].entities}
        self.assertIn("DOC_CONFIRMED_001", entity_ids)
        self.assertNotIn("DOC_DRAFT_001", entity_ids)

    def test_output_round_trip_preserves_first_class_evidence(self) -> None:
        output = RetrieveSimilarCaseTool(self.golden_repository).run(
            knowledge_input(
                "REQ_TYPED_KNOWLEDGE_ROUND_TRIP",
                query="Cu CMP slurry flow",
                module="Cu CMP",
            )
        )

        restored = ToolOutput.from_dict(output.to_dict())
        self.assertEqual(restored.evidence_ids, output.evidence_ids)
        self.assertEqual(
            [evidence.to_dict() for evidence in restored.evidence],
            restored.data["evidence"],
        )
        self.assertTrue(all(evidence.is_typed for evidence in restored.evidence))

    def test_empty_query_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ModelValidationError,
            "query must be a non-empty string",
        ):
            RetrieveSimilarCaseTool(self.golden_repository).run(
                knowledge_input(
                    "REQ_TYPED_KNOWLEDGE_EMPTY_QUERY",
                    query="   ",
                )
            )

    def test_knowledge_tool_rejects_non_owner_agent(self) -> None:
        with self.assertRaisesRegex(
            ModelValidationError,
            "belongs to agent knowledge",
        ):
            RetrieveSimilarCaseTool(self.golden_repository).run(
                knowledge_input(
                    "REQ_TYPED_KNOWLEDGE_WRONG_OWNER",
                    query="Cu CMP slurry flow",
                    requested_by=AgentKind.FDC.value,
                )
            )


if __name__ == "__main__":
    unittest.main()
