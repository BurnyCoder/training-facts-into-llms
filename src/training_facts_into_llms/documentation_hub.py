"""Expose only metadata reads, document downloads, and additive Hub commits.

Model bytes are never downloaded or loaded. Git blob and LFS identities prove
that untouched objects persist while documents receive direct byte checks.
Sources: https://huggingface.co/docs/huggingface_hub/package_reference/hf_api
https://huggingface.co/docs/huggingface_hub/guides/upload
"""

from pathlib import Path
from typing import Any


class DocumentationHub:
    """Keep repository, Collection, model-load, and deletion APIs out of this surface."""

    def __init__(self, root: Path, secret: str) -> None:
        from huggingface_hub import HfApi

        self.authenticated = HfApi(token=secret)
        self.anonymous = HfApi(token=False)
        self.cache = root / ".cache/huggingface-docs"

    def inspect(
        self, repo_id: str, repo_type: str, *, anonymous: bool,
        revision: str | None = None,
    ) -> dict[str, Any]:
        """Pin a complete paginated tree to the resolved repository commit."""
        from huggingface_hub import RepoFile

        api = self.anonymous if anonymous else self.authenticated
        info = api.repo_info(repo_id, repo_type=repo_type, revision=revision)
        if info.id != repo_id:
            raise ValueError("documentation repository identity changed")
        tree = {}
        for item in api.list_repo_tree(
            repo_id, repo_type=repo_type, revision=info.sha, recursive=True,
        ):
            if isinstance(item, RepoFile):
                if item.path in tree:
                    raise ValueError("duplicate remote documentation path")
                tree[item.path] = {
                    "blob_id": item.blob_id, "size": item.size,
                    "lfs_sha256": item.lfs.sha256 if item.lfs is not None else None,
                }
        return {
            "revision": info.sha, "private": bool(info.private),
            "gated": bool(info.gated), "tree": tree,
        }

    def read(
        self, repo_id: str, repo_type: str, path: str, *, revision: str, anonymous: bool,
    ) -> bytes:
        """Download only one allowlisted document at an immutable revision."""
        api = self.anonymous if anonymous else self.authenticated
        local = api.hf_hub_download(
            repo_id, path, repo_type=repo_type, revision=revision,
            cache_dir=self.cache, force_download=True,
        )
        return Path(local).read_bytes()

    def commit(self, repository: Any) -> str:
        """Send one explicit add/replace operation per reviewed document, atomically."""
        from huggingface_hub import CommitOperationAdd

        result = self.authenticated.create_commit(
            repo_id=repository.repo_id, repo_type=repository.repo_type, revision="main",
            parent_commit=repository.parent_commit,
            operations=[
                CommitOperationAdd(path_in_repo=path, path_or_fileobj=raw)
                for path, raw in sorted(repository.files.items())
            ],
            commit_message="Add reviewed documentation edition audit-2026-10-29",
        )
        return result.oid
