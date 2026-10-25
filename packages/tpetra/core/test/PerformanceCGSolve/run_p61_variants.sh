#!/usr/bin/env bash
# P6.1: test the SAME P6 direct/phase benchmark with all three mutually
# exclusive GPU PUT execution granularities. No variant D / on-stream PUT.
# Each invocation allocates the packet plan and symmetric heap once, before
# warmup and timed calls. Runs are sequential to avoid resource contention.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOPOLOGY="${1:-inter16}"
SIZE="${2:-100}"
MODE="${3:-both}"
PASSES="${P61_PASSES:-2}"  # rotate order to reduce GPU clock drift bias
if ! [[ "$PASSES" =~ ^[1-9][0-9]*$ ]]; then
  echo 'P61_PASSES must be a positive integer' >&2; exit 2
fi
for ((pass=1; pass<=PASSES; pass++)); do
  # Alternate ordering. Every variant is measured with the same options.
  if ((pass % 2)); then variants=(thread warp block);
  else variants=(block warp thread); fi
  for variant in "${variants[@]}"; do
    label="${P6_LABEL:-p61}_pass${pass}_${variant}"
    printf 'P6.1 pass=%d/%d variant=%s topology=%s size=%s mode=%s\n' \
       "$pass" "$PASSES" "$variant" "$TOPOLOGY" "$SIZE" "$MODE"
    P6_PUT_VARIANT="$variant" P6_LABEL="$label" \
       "$HERE/run_p6_comparison.sh" "$TOPOLOGY" "$SIZE" "$MODE"
  done
done
