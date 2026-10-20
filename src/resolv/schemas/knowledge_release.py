"""Immutable release identities and evidence receipts. No I/O or activation."""

from hashlib import sha256
from typing import ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from resolv.schemas.hybrid_knowledge import (
    Count,
    FrozenContract,
    FrozenFileConfig,
    Hash64,
    HybridIndexProfile,
    Name,
    Revision,
    Version,
)
from resolv.services.hybrid_fingerprints import canonical_hash, profile_hash


class KnowledgeContract(FrozenContract):
    model_config = ConfigDict(
        **(
            FrozenContract.model_config
            | {
                "validate_default": True,
                "revalidate_instances": "always",
            }
        )
    )

    @model_validator(mode="after")
    def check_nested_models(self) -> Self:
        def checked(item):
            if isinstance(item, BaseModel):
                type(item).model_validate(item.model_dump())
            elif isinstance(item, tuple):
                for value in item:
                    checked(value)

        for name in type(self).model_fields:
            checked(getattr(self, name))
        return self

    def model_copy(self, *, update=None, deep=False):
        # Pydantic's default model_copy deliberately skips validation of updates.
        return type(self).model_validate(self.model_dump() | (update or {}))

    @property
    def fingerprint(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class ReleaseSelection(KnowledgeContract):
    status: Literal["disabled", "invalid", "configured"]
    release_id: Version | None = None
    index_version: Version | None = None
    collection: Name | None = None
    profile_hash: Hash64 | None = None
    manifest_hash: Hash64 | None = None
    profile: HybridIndexProfile | None = None
    error_code: str | None = None
    config_version: Literal[1] = 1

    @model_validator(mode="after")
    def identity(self) -> Self:
        values = (
            self.release_id,
            self.index_version,
            self.collection,
            self.profile_hash,
            self.manifest_hash,
            self.profile,
        )
        if self.status == "configured":
            if any(v is None for v in values) or self.error_code is not None:
                raise ValueError("Configured selection requires full identity")
            if (
                self.profile.collection_name != self.collection
                or profile_hash(self.profile) != self.profile_hash
            ):
                raise ValueError("Selection profile identity mismatch")
        elif any(v is not None for v in values):
            raise ValueError("Inactive selection cannot carry a release identity")
        if (self.status == "invalid") != bool(self.error_code):
            raise ValueError("Only invalid selections carry an error code")
        return self


class ReleaseDocument(KnowledgeContract):
    original_path: str = Field(min_length=1)
    original_commit: Revision
    source: str = Field(min_length=1)
    document_id: Hash64
    raw_sha256: Hash64
    content_hash: Hash64
    file_config: FrozenFileConfig
    max_chars: int = Field(default=800, gt=0)
    overlap: Count = 100
    passage_max_length: Literal[512] = 512

    @model_validator(mode="after")
    def identity(self) -> Self:
        if self.document_id != sha256(self.source.encode("utf-8")).hexdigest():
            raise ValueError("Source/document identity mismatch")
        if self.overlap >= self.max_chars:
            raise ValueError("Invalid release chunk overlap")
        return self


class ReleaseManifest(KnowledgeContract):
    release_id: Version
    index_version: Version
    collection: Name
    policy_commit: Revision
    policy_tree: Revision
    file_config_sha256: Hash64
    index_profile: HybridIndexProfile
    documents: tuple[ReleaseDocument, ...] = Field(min_length=1)
    manifest_rule: Literal["git-knowledge-release-v1"] = "git-knowledge-release-v1"

    @model_validator(mode="after")
    def identity(self) -> Self:
        if self.index_profile.collection_name != self.collection:
            raise ValueError("Release collection/profile mismatch")
        sources = [d.source for d in self.documents]
        if len(set(sources)) != len(sources):
            raise ValueError("Duplicate release source")
        for d in self.documents:
            expected = f"knowledge/releases/{self.release_id}/{d.original_path.rsplit('/', 1)[-1]}"
            if d.original_commit != self.policy_commit or d.source != expected:
                raise ValueError("Release source/commit mismatch")
        return self


class ReleaseBuildReceipt(KnowledgeContract):
    release_id: Version
    index_version: Version
    collection: Name
    profile_hash: Hash64
    manifest_hash: Hash64
    release_manifest_hash: Hash64
    database: str = Field(min_length=1)
    server_fingerprint: Hash64
    state: Literal["building", "ready"]
    expected_count: Count
    done_count: Count
    pending_count: Count

    @model_validator(mode="after")
    def counts(self) -> Self:
        if self.done_count + self.pending_count != self.expected_count:
            raise ValueError("Build receipt counts mismatch")
        if self.state == "ready" and (self.pending_count or not self.expected_count):
            raise ValueError("Ready receipt requires a nonempty complete index")
        return self


GateName = Literal["K1", "K2", "K3", "snapshot_python", "snapshot_frontend", "review"]


class GateReceipt(KnowledgeContract):
    name: GateName
    passed: bool
    report_sha256: Hash64
    source_fingerprints: tuple[tuple[str, Hash64], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_sources(self) -> Self:
        names = [name for name, _ in self.source_fingerprints]
        if any(not name.strip() for name in names) or len(set(names)) != len(names):
            raise ValueError("Gate source fingerprints must be unique and named")
        return self


class ReleaseAcceptance(KnowledgeContract):
    required_gates: ClassVar[frozenset[str]] = frozenset(
        {
            "K1",
            "K2",
            "K3",
            "snapshot_python",
            "snapshot_frontend",
            "review",
        }
    )
    candidate_commit: Revision
    policy_commit: Revision
    source_trees: tuple[tuple[str, Revision], ...] = Field(min_length=1)
    release_id: Version
    index_version: Version
    profile_hash: Hash64
    manifest_hash: Hash64
    gate_receipts: tuple[GateReceipt, ...]
    review_important_count: Count

    @model_validator(mode="after")
    def unique_receipts(self) -> Self:
        names = [r.name for r in self.gate_receipts]
        trees = [name for name, _ in self.source_trees]
        if len(set(names)) != len(names) or len(set(trees)) != len(trees):
            raise ValueError("Duplicate acceptance receipt or source tree")
        return self

    def require_passed(self) -> None:
        if (
            {r.name for r in self.gate_receipts} != self.required_gates
            or not all(r.passed for r in self.gate_receipts)
            or self.review_important_count
        ):
            raise ValueError("Required acceptance evidence is incomplete")
