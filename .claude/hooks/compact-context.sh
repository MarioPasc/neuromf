#!/usr/bin/env bash
# Compaction recovery hook — re-injects critical context after /compact
# Called by Claude Code on SessionStart[compact] events

cat <<'CONTEXT'


=== END COMPACTION RECOVERY ===
CONTEXT
