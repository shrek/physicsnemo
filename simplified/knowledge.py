"""Minimal, evidence-backed wiki services for instrumentation agents.

Published wiki pages are read-only to proposing agents. Agents may append
low-trust discoveries to run memory; only draft promotion changes the wiki.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import yaml

from simplified.types import (
    AgentContext,
    Critique,
    KnowledgeItem,
    KnowledgeQuery,
    MemoryItem,
    TrainingSpec,
)


_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)
_KIND_DIRECTORIES = {
    "contract": "contracts",
    "playbook": "playbooks",
    "failure_pattern": "failure-patterns",
    "recipe_profile": "recipe-profiles",
}


class KnowledgeStore(Protocol):
    def search(self, query: KnowledgeQuery) -> tuple[KnowledgeItem, ...]: ...

    def read(self, identifier: str) -> KnowledgeItem: ...

    def create_draft(self, item: KnowledgeItem) -> Path: ...


class RunMemory(Protocol):
    def record(
        self,
        kind: str,
        content: str,
        *,
        trust: str = "observed",
        citations: tuple[str, ...] = (),
    ) -> MemoryItem: ...

    def search(self, text: str, limit: int = 4) -> tuple[MemoryItem, ...]: ...


def default_wiki_root() -> Path:
    return Path(__file__).with_name("instrumentation_wiki")


def _parse_page(path: Path) -> KnowledgeItem:
    match = _FRONTMATTER.match(path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"{path}: missing YAML frontmatter")
    metadata = yaml.safe_load(match.group(1)) or {}
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: frontmatter must be a mapping")
    metadata["body"] = match.group(2).strip()
    return KnowledgeItem.model_validate(metadata)


def _render_page(item: KnowledgeItem) -> str:
    metadata = item.model_dump(exclude={"body"}, mode="json")
    return f"---\n{yaml.safe_dump(metadata, sort_keys=False).strip()}\n---\n\n{item.body.strip()}\n"


class FilesystemKnowledgeStore:
    """Small typed wiki with deterministic lexical retrieval and validation."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or default_wiki_root()).resolve()

    @property
    def wiki_directory(self) -> Path:
        return self.root / "wiki"

    @property
    def candidate_directory(self) -> Path:
        return self.root / "candidates"

    def _pages(self, directory: Path) -> tuple[tuple[Path, KnowledgeItem], ...]:
        if not directory.is_dir():
            return ()
        return tuple((path, _parse_page(path)) for path in sorted(directory.rglob("*.md")))

    def _aliases(self) -> dict[str, str]:
        path = self.root / "data" / "aliases.yaml"
        if not path.is_file():
            return {}
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("data/aliases.yaml must be a mapping")
        aliases: dict[str, str] = {}
        for canonical, values in raw.items():
            if not isinstance(canonical, str) or not isinstance(values, list):
                raise ValueError("aliases must map a string to a list of strings")
            for value in [canonical, *values]:
                if not isinstance(value, str):
                    raise ValueError("aliases must contain strings")
                key = value.lower()
                if key in aliases and aliases[key] != canonical:
                    raise ValueError(f"ambiguous alias: {value}")
                aliases[key] = canonical
        return aliases

    def _known_tags(self) -> set[str]:
        path = self.root / "data" / "tags.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        tags = raw.get("tags", []) if isinstance(raw, dict) else []
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError("data/tags.yaml must contain a string list at tags")
        return set(tags)

    def _known_evidence_ids(self) -> set[str]:
        evidence: set[str] = set()
        sources = self.root / "sources"
        for path in sorted(sources.rglob("*.json")) if sources.is_dir() else ():
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("id"), str):
                evidence.add(raw["id"])
            else:
                raise ValueError(f"{path}: source record must contain a string id")
        return evidence

    def search(self, query: KnowledgeQuery) -> tuple[KnowledgeItem, ...]:
        aliases = self._aliases()
        requested_tags = {aliases.get(tag.lower(), tag) for tag in query.tags}
        terms = [aliases.get(term.lower(), term).lower() for term in re.findall(r"[A-Za-z0-9_-]+", query.text)]
        scored: list[tuple[int, KnowledgeItem]] = []
        for _, item in self._pages(self.wiki_directory):
            if item.status != "published" or item.trust != "verified":
                continue
            if requested_tags and not requested_tags.issubset(set(item.tags)):
                continue
            if any(item.applies_when.get(key) != value for key, value in query.attributes.items()):
                continue
            title = item.title.lower()
            tags = " ".join(item.tags).lower()
            body = item.body.lower()
            score = sum(
                10 * title.count(term) + 5 * tags.count(term) + body.count(term)
                for term in terms
            )
            if terms and score == 0:
                continue
            scored.append((score, item))
        scored.sort(key=lambda value: (-value[0], value[1].id))
        return tuple(item for _, item in scored[: query.limit])

    def read(self, identifier: str) -> KnowledgeItem:
        for _, item in (*self._pages(self.wiki_directory), *self._pages(self.candidate_directory)):
            if item.id == identifier:
                return item
        raise KeyError(identifier)

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        try:
            self._aliases()
            known_tags = self._known_tags()
            known_evidence_ids = self._known_evidence_ids()
        except (OSError, ValueError, yaml.YAMLError) as error:
            return (str(error),)
        ids: set[str] = set()
        for directory, expected_status in ((self.wiki_directory, "published"), (self.candidate_directory, "draft")):
            for path in sorted(directory.rglob("*.md")) if directory.is_dir() else ():
                try:
                    item = _parse_page(path)
                except (OSError, ValueError, yaml.YAMLError) as error:
                    errors.append(str(error))
                    continue
                if item.id in ids:
                    errors.append(f"duplicate knowledge id: {item.id}")
                ids.add(item.id)
                if item.status != expected_status:
                    errors.append(f"{path}: expected status {expected_status}")
                unknown = set(item.tags) - known_tags
                if unknown:
                    errors.append(f"{path}: unknown tags {sorted(unknown)}")
                if item.trust == "verified" and not item.evidence_ids:
                    errors.append(f"{path}: verified pages require evidence_ids")
                missing_evidence = set(item.evidence_ids) - known_evidence_ids
                if missing_evidence:
                    errors.append(f"{path}: unknown evidence ids {sorted(missing_evidence)}")
        return tuple(errors)

    def create_draft(self, item: KnowledgeItem) -> Path:
        if item.status != "draft":
            raise ValueError("new knowledge must start as a draft")
        self.candidate_directory.mkdir(parents=True, exist_ok=True)
        path = self.candidate_directory / f"{item.id}.md"
        if path.exists():
            raise FileExistsError(path)
        path.write_text(_render_page(item), encoding="utf-8")
        return path

    def promote(self, identifier: str, reviewer: str) -> KnowledgeItem:
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        source = self.candidate_directory / f"{identifier}.md"
        item = _parse_page(source)
        if item.status != "draft":
            raise ValueError("only drafts may be promoted")
        errors = self.validate()
        if errors:
            raise ValueError("cannot promote invalid knowledge: " + "; ".join(errors))
        published = item.model_copy(update={"status": "published", "version": item.version + 1})
        target = self.wiki_directory / _KIND_DIRECTORIES[published.kind] / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        target.write_text(_render_page(published), encoding="utf-8")
        source.unlink()
        audit = self.candidate_directory / "promotion-log.jsonl"
        with audit.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"id": identifier, "reviewer": reviewer, "version": published.version}, sort_keys=True) + "\n")
        return published


class JsonlRunMemory:
    """Run-scoped append-only memory; agent claims never alter the wiki."""

    def __init__(self, path: str | Path | None = None, run_id: str = "interactive"):
        self.path = Path(path) if path is not None else None
        self.run_id = run_id
        self._items: list[MemoryItem] = []
        if self.path is not None and self.path.is_file():
            self._items = [MemoryItem.model_validate_json(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def record(self, kind: str, content: str, *, trust: str = "observed", citations: tuple[str, ...] = ()) -> MemoryItem:
        item = MemoryItem(id=f"memory-{uuid4().hex[:12]}", run_id=self.run_id, kind=kind, trust=trust, content=content, citations=citations)
        self._items.append(item)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(item.model_dump_json() + "\n")
        return item

    def search(self, text: str, limit: int = 4) -> tuple[MemoryItem, ...]:
        terms = set(re.findall(r"[A-Za-z0-9_-]+", text.lower()))
        matches = [item for item in reversed(self._items) if not terms or terms.intersection(item.content.lower().split())]
        return tuple(matches[:limit])


class ContextBuilder:
    """Build a small, deterministic context from published knowledge and memory."""

    def __init__(self, knowledge: KnowledgeStore, memory: RunMemory):
        self.knowledge = knowledge
        self.memory = memory

    def build(self, spec: TrainingSpec, required_ranges: tuple[str, ...], previous: Critique | None) -> AgentContext:
        payload = spec.model_dump_json() + "\n" + ",".join(required_ranges)
        fingerprint = hashlib.sha256(payload.encode()).hexdigest()[:16]
        pages = self.knowledge.search(KnowledgeQuery(text="instrumentation profiler training loop", tags=("instrumentation",), limit=6))
        feedback = previous.feedback if previous else ""
        memories = self.memory.search(feedback, limit=4)
        return AgentContext(
            source_fingerprint=fingerprint,
            knowledge_snapshot=pages,
            memory_snapshot=memories,
            retrieval_summary=f"{len(pages)} published knowledge page(s) and {len(memories)} run-memory item(s) selected.",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the instrumentation knowledge base")
    parser.add_argument("command", choices=("validate", "search", "show", "promote"))
    parser.add_argument("value", nargs="?")
    parser.add_argument("--root", type=Path, default=default_wiki_root())
    parser.add_argument("--reviewer", default="")
    args = parser.parse_args()
    store = FilesystemKnowledgeStore(args.root)
    if args.command == "validate":
        errors = store.validate()
        if errors:
            raise SystemExit("\n".join(errors))
        print("knowledge base is valid")
    elif args.command == "search":
        for item in store.search(KnowledgeQuery(text=args.value or "")):
            print(f"{item.id}\t{item.kind}\t{item.title}")
    elif args.command == "show":
        print(store.read(args.value or "").model_dump_json(indent=2))
    else:
        print(store.promote(args.value or "", args.reviewer).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
