"""Where repos come from: Hugging Face (default), a hub directory (offline: DIR/<org>/<name>/<rev>/...), or a local dir."""
import os
from typing import Optional

from .hf import HFClient, HFSource, LocalSource


class Hub:
    def __init__(self, client: Optional[HFClient] = None, hub_dir: Optional[str] = None):
        self.hub_dir = hub_dir or os.environ.get("REAX_LINEAGE_HUB_DIR")
        self.client = client if client is not None else (None if self.hub_dir else HFClient())

    def _client(self):
        if self.client is None:
            self.client = HFClient()
        return self.client

    def source(self, repo, rev):
        if self.hub_dir:
            p = os.path.join(self.hub_dir, repo, rev)
            if os.path.isdir(p):
                s = LocalSource(p)
                s.repo, s.rev = repo, rev
                s.name = "%s@%s" % (repo, rev)
                return s
            if self.client is None and not os.environ.get("HF_TOKEN"):
                raise FileNotFoundError("%s@%s not in hub dir %s" % (repo, rev, self.hub_dir))
        return HFSource(repo, rev, self._client())

    def resolve_rev(self, repo, rev):
        """Full sha. Hub-dir revisions are used as given."""
        if self.hub_dir and os.path.isdir(os.path.join(self.hub_dir, repo, rev)):
            return rev
        if len(rev) == 40:
            return rev
        return self._client().resolve_revision(repo, rev)["sha"]
