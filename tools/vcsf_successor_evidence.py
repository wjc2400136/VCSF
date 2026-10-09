"""Bounded, transaction-local read reuse for successor operational checks.

First reads authenticate bytes. Reuse checks file identity and timestamps;
call full_unchanged at the transaction boundary to rehash all evidence.
Historical auditors retain their original Evidence implementation.
"""
from collections import OrderedDict

from tools.audit_vcsf_cpu_analysis import Evidence, check, is_sha, plain, signature


class SuccessorEvidence(Evidence):
    def __init__(self, cache_bytes=64 << 20):
        super().__init__()
        self._cache = OrderedDict()
        self._cache_size = 0
        self._cache_limit = cache_bytes

    def bytes(self, path, expected, cap=None):
        path = plain(path)
        name = str(path)
        check(is_sha(expected) and (name not in self.checked or self.checked[name] == expected),
            'Conflicting or invalid expected evidence hash')
        current = signature(path.stat())
        check(cap is None or current[2] <= cap, 'Evidence exceeds its byte limit')
        if name in self.signatures:
            check(current == self.signatures[name], 'Previously read evidence changed')
            if cap is None:
                return None
            if name in self._cache:
                self._cache.move_to_end(name)
                return self._cache[name]
        raw = super().bytes(path, expected, cap)
        if raw is not None and len(raw) <= self._cache_limit:
            while self._cache and self._cache_size + len(raw) > self._cache_limit:
                _, old = self._cache.popitem(last=False)
                self._cache_size -= len(old)
            self._cache[name] = raw
            self._cache_size += len(raw)
        return raw

    def unchanged(self):
        check(all(signature(plain(name).stat()) == expected
            for name, expected in self.signatures.items()),
            'An audited input changed before completion')

    def full_unchanged(self):
        super().unchanged()
