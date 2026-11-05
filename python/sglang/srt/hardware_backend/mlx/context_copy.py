# SPDX-License-Identifier: Apache-2.0
"""Request-owned context copy proposals with immutable prefix snapshots."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CopyTokenPage:
    previous: "CopyTokenPage | None"
    tokens: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CopyTokenSnapshot:
    page: CopyTokenPage | None = None
    tail: tuple[int, ...] = ()


class ContextCopyProposer:
    """Index only consumed tokens; all proposed tokens still require verification.

    An eight-token suffix must match an earlier span with a complete continuation.
    Prefix snapshots retain immutable pages, so later appends, rejected proposals,
    and request removal cannot alter another request's history. The mutable index
    belongs to one request and is rebuilt only when restoring a prefix.
    """

    def __init__(self, snapshot=None, *, ngram_size=8):
        if ngram_size < 1:
            raise ValueError("The matching suffix must contain at least one token")
        self.ngram_size = ngram_size
        self.snapshot = snapshot or CopyTokenSnapshot()
        pages = []
        page = self.snapshot.page
        while page is not None:
            pages.append(page.tokens)
            page = page.previous
        self.tokens = [token for values in reversed(pages) for token in values]
        self.tokens.extend(self.snapshot.tail)
        self.positions = {}
        self.prefixes = set()
        for end in range(ngram_size, len(self.tokens) + 1):
            self._index(end)

    def _index(self, end):
        start = end - self.ngram_size
        key = tuple(self.tokens[start:end])
        self.positions.setdefault(key, []).append(start)
        self.prefixes.add(key[:-1])

    def extend(self, tokens):
        snapshot = self.snapshot
        for token in tokens:
            token = int(token)
            self.tokens.append(token)
            tail = snapshot.tail + (token,)
            if len(tail) == 64:
                snapshot = CopyTokenSnapshot(CopyTokenPage(snapshot.page, tail))
            else:
                snapshot = CopyTokenSnapshot(snapshot.page, tail)
            if len(self.tokens) >= self.ngram_size:
                self._index(len(self.tokens))
        self.snapshot = snapshot

    def can_match_next(self):
        """Avoid reading a GPU anchor unless its known suffix has a source."""
        if len(self.tokens) + 1 < self.ngram_size:
            return False
        prefix = (
            tuple(self.tokens[-(self.ngram_size - 1) :]) if self.ngram_size > 1 else ()
        )
        return prefix in self.prefixes

    def propose(self, anchor, count):
        if count <= 0 or len(self.tokens) + 1 < self.ngram_size:
            return ()
        key = (
            tuple(self.tokens[-(self.ngram_size - 1) :]) + (int(anchor),)
            if self.ngram_size > 1
            else (int(anchor),)
        )
        for start in reversed(self.positions.get(key, ())):
            end = start + self.ngram_size
            if end + count <= len(self.tokens):
                return tuple(self.tokens[end : end + count])
        return ()
