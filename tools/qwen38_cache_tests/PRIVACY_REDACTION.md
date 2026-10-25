# Diagnostic artifact privacy

Private original logs, profiler traces, request/response bodies and replay token
IDs remain at their existing local paths and are ignored by Git. Original Git
history is retained in a private bundle outside the repository for local review.
Do not force-add these files.

Publishable copies of affected JSON summaries live under `redacted/`, preserving
performance numbers while removing message bodies, replay token arrays, request
identifiers, private network addresses and personal home-directory paths. Raw
binary profiler captures and logs are excluded from publishable branch history.

The cleanup reviewed 216 non-upstream commits across experiment branches and
local checkpoints. 71 commits directly touched artifacts selected for cleanup;
other commit IDs changed through ancestry. Author and committer identity lines
were verified unchanged for all 216 commits. Runtime code and registered tests
remain unchanged. Existing modified code in other worktrees was preserved.

Credential scanning found no confirmed live API token or private key. Entropy
findings were content/commit hashes, request IDs, kernel symbols, environment
settings and an intentional fake-token test fixture. This is a scoped audit of
project history, not a scan of unrelated private files or external services.

Historical provenance hashes describe artifacts before redaction and should not
be used to validate the redacted copies. Local originals and the private Git
bundle retain the original material for authorized investigation.

This local cleanup does not rewrite the remote repository until the sanitized
branches are explicitly published. Original private material must stay outside
Git when creating new diagnostics or sharing an archive.
