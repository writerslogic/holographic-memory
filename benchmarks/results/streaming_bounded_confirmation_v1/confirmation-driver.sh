#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/hms-streaming-bounded"
binary=$(cat binary.txt)
[[ $(sha256sum "$binary" | cut -d' ' -f1) == 6c3401a81769323dd32796ea75ae45e9d6edccb22a7bc50af9aea3d60a10e8d2 ]]
[[ $(sha256sum pair.sh | cut -d' ' -f1) == 2868b63206286c25668c285c298428615b3d44477ddfaa13f477b8a874f194a7 ]]
[[ $(sha256sum run.sh | cut -d' ' -f1) == 61f5f59abb33cf1107e495fd2e87bce4a0e0f0f6e0ffbd16aa13c3cab12dfae3 ]]
[[ $(sha256sum confirm.py | cut -d' ' -f1) == 4c02185d23a2265c01bad7dca75a26919a06be24a701d40be12d3a4121ac85ac ]]
mkdir -p confirmation-receipts
test ! -e confirmation-complete
for rows in 1000000 3000000; do
    for round in 1 2 3; do
        for arm in global segments; do
            name="$rows-250000-$round-$arm-640"
            test ! -e "work-$name"
            test ! -e "evidence/$name"
            test ! -e "evidence/$name.log"
        done
        printf 'START rows=%s round=%s utc=%s\n' "$rows" "$round" "$(date -u +%FT%TZ)"
        timeout --signal=TERM --kill-after=5s 820s bash pair.sh "$rows" 250000 "$round" 640 128 256
        for arm in global segments; do
            name="$rows-250000-$round-$arm-640"
            uv run --no-project python confirm.py --evidence evidence --arm "$name" \
                --output "confirmation-receipts/$name.json" > "confirmation-receipts/$name.log" 2>&1
        done
        printf 'PASS rows=%s round=%s utc=%s\n' "$rows" "$round" "$(date -u +%FT%TZ)"
    done
done
date -u +%FT%TZ > confirmation-complete
