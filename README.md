# Ember Forecast GTA6 — Twitch viewers, hour by hour, on GTA VI launch day

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/pranatakripto95/ember-forecast-gta6/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/pranatakripto95/ember-forecast-gta6/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

An entry in [make.fun's "Forecast GTA VI's launch-day Twitch viewers hour by hour" bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) ($250 in USDC).

`main.ts` (Node 22, standard library only, no network) reads one case file
(`/input/<id>.in`: hour 1's Twitch viewer print for the "Grand Theft Auto VI"
category plus the case's comparable past launches) and writes
`/output/<id>.out`: five quantiles (5%, 25%, 50%, 75%, 95%) of concurrent
viewers for each of hours 2–24 of release day, 2026-11-19 UTC.

## Model — v2 (current)

Per-hour local log-ratio regression on hour-1 scale. For each forecast hour
h, the model fits `ln(viewers_h / viewers_1) = a_h + b_h · (ln h1 − m)` by
kernel-weighted least squares over the comparables (Gaussian weights on
log-h1 distance, bandwidth 1.5; slope shrunk toward 0 by
`n_eff / (n_eff + 5)`). Spread is the kernel-weighted residual std per hour,
shrunk toward the pooled sigma and scaled by 0.8. Hours with fewer than 8
usable comparables fall back to the house persistence forecast.

Why the change: v1's neighbour median misses the scale gradient — small
hour-1 launches explode in the evening (Silksong 2000× from 164 viewers;
Wolverine 9× from 23k) while mega-launches peak in hour 1 (Elden Ring,
Cyberpunk 2077). The regression captures it.

Validated on the **exact practice grader** (the four practice launches'
true SullyGnome curves, which reproduce v1's server practice 0.251 and the
house baseline's 0.387 to three decimals): **v2 practice pinball 0.178**
(Wolverine 0.190, Red Dead 2 0.196, God of War Ragnarök 0.138, Spider-Man 2
0.187), median log error 0.28, 90% coverage 0.99. Leave-one-out over the
comparables agrees: 0.390 vs v1's 0.495.

## Model — v1 (superseded)

A launch's day is governed by its *regime*: whether hour 1 already holds a
big live audience (rolling midnight unlock reached the big timezones before
00:00 UTC) or hour 1 is still a pre-unlock trickle whose audience then
explodes by the evening. Two numbers measured at the checkpoint separate the
regimes, and both are known for the target case too:

- **lift** = hour-1 viewers / median hourly viewers in the 7 days before
  release day (from the target's own `history_hourly`);
- **h1** = hour-1 viewers (absolute scale: mega-launches behave differently
  from small ones).

For each comparable launch the program computes the same two numbers, takes
the 12 nearest neighbours in (log lift, log h1) distance, and reads each
forecast hour's five quantiles straight off the neighbours' empirical
distribution of log(viewers_h / viewers_1). No curve shape is assumed: the
evening peak (hours 19–21 UTC), the 01:00–02:00 UTC dip and the widening
spread all come from the neighbours themselves. A light 3-hour median smooth
removes single-hour jitter from the small neighbour pool. If too few
neighbours cover an hour, it falls back to all comparables, then to the house
persistence forecast, so every case stays valid.

Chosen by leave-one-out pinball scoring over the kit's 37 usable comparable
launches (each treated as a target, the other 36 as its pool): **mean pinball
loss 0.495 vs 0.985 for the house persistence baseline**. The shipped program
reproduces that score end to end (see `out_practice/` runs during development;
the check is deterministic — two runs write byte-identical outputs).

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

With the kit's practice cases in `input/` this writes `p00.out` … `p03.out`.
Typical forecasts: hour 2 median ≈ 0.7× hour 1, evening peak ≈ 1.4–2.3×
hour 1 around hours 19–21 UTC.

## Sources

Comparable launch curves: SullyGnome.com hourly Twitch category averages, as
bundled in the competition kit (`input/NOTICE.txt`). The final case's hour-1
print comes from make.fun's own minute-read recorder of Twitch's public
directory count for the category (game id 1867279146).

## Files

- `main.ts` — the forecaster (the only entrypoint)
- `package.json` — `{"type": "module"}` so Node runs `main.ts` as ESM
- `.makefun/owner.json` — claim file naming the entry's Solana wallet
