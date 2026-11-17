// Ember Forecast GTA6 — make.fun gta6-twitch-ccu-forecast-01
//
// Forecasts Twitch concurrent viewers in the "Grand Theft Auto VI" category
// for hours 2..24 of release day (2026-11-19 UTC), from hour 1's print plus
// the case file's 42 comparable past big-game launches (SullyGnome hourly
// Twitch category averages, via the kit's input cases).
//
// Model (chosen by leave-one-out pinball scoring over the comparables:
// mean pinball 0.50 vs 0.99 for the house persistence baseline):
//
//   A launch's day is governed by its *regime*: whether hour 1 already
//   holds a big live audience (rolling midnight unlock reached the big
//   timezones before 00:00 UTC, e.g. Elden Ring, Cyberpunk 2077) or hour 1
//   is still a pre-unlock trickle (e.g. Silksong, Kingdom Come II, whose
//   audiences then explode 100-2000x by the evening). Two numbers measured
//   at the checkpoint separate the regimes, and both are known for the
//   target case too:
//     - lift  = hour-1 viewers / median hourly viewers in the 7 days
//               before release day (the target's own history_hourly);
//     - h1    = hour-1 viewers (absolute scale: mega-launches behave
//               differently from small ones).
//   For each comparable we compute the same two numbers, take the 12
//   nearest neighbours in (log lift, log h1) distance, and read each
//   forecast hour's five quantiles straight off the neighbours' empirical
//   distribution of log(viewers_h / viewers_1). No curve is assumed: the
//   evening peak, the 01:00-02:00 UTC dip and the widening spread all come
//   from the neighbours themselves. A light 3-hour median smooth removes
//   single-hour jitter from the small neighbour pool, and a 90% band
//   narrower than 0.8 log units is widened around the median, since 12
//   neighbours cannot honestly support more certainty than that.
//
//   Fallbacks keep every case valid: if too few neighbours have an hour,
//   that hour falls back to all comparables, then to the house persistence
//   forecast (median = hour 1, log-spread 0.15 * sqrt(h-1)).
//
// Deterministic, standard library only, no network, runs in milliseconds.
//
// Sources for the comparable curves: SullyGnome.com hourly Twitch category
// averages, as bundled in the kit (input/NOTICE.txt). The final case's
// hour-1 print comes from make.fun's own minute-read recorder.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.645, -0.674, 0, 0.674, 1.645]; // standard normal quantiles
const K_NEIGHBOURS = 12;
const W_LOG_H1 = 1.0; // weight of log(h1) vs log(lift) in neighbour distance
const BASELINE_WINDOW_HOURS = 168; // 7 days of pre-release history
const MIN_BASELINE_POINTS = 6;
const FALLBACK_SIGMA = 0.15; // house baseline log-spread per sqrt hour
const MIN_LOG_SPREAD_90 = 0.8; // narrowest honest 90% band, in log units

interface HourlyPoint {
  from: string;
  viewers: number | null;
  channels?: number | null;
}

interface Comparable {
  name: string;
  day_start: string;
  hourly: HourlyPoint[];
}

interface CaseFile {
  id: string;
  levels: number[];
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  history_hourly?: HourlyPoint[] | null;
  history?: Array<{ at: string; viewers: number | null }> | null;
  comparables: Comparable[];
}

interface LaunchCurve {
  name: string;
  h1: number;
  lift: number | null;
  byHour: Map<number, number>; // UTC hour offset from day_start -> viewers
}

const parseTime = (s: string): number => Date.parse(s);

const median = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  if (n === 0) return Number.NaN;
  return n % 2 === 1 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
};

/** Median of positive hourly viewers in the window before day_start. */
const baselineMedian = (points: Array<{ t: number; v: number }>, dayStart: number): number | null => {
  const vals = points
    .filter((p) => p.t < dayStart && p.t >= dayStart - BASELINE_WINDOW_HOURS * 3600_000 && p.v > 0)
    .map((p) => p.v);
  if (vals.length < MIN_BASELINE_POINTS) return null;
  return median(vals);
};

/** Whole-hour map (offset from day_start) for one launch's hourly series. */
const hourlyMap = (hourly: HourlyPoint[], dayStart: number): Map<number, number> => {
  const m = new Map<number, number>();
  for (const p of hourly) {
    if (p.viewers === null || p.viewers === undefined || p.viewers <= 0) continue;
    const h = Math.round((parseTime(p.from) - dayStart) / 3600_000);
    if (!m.has(h)) m.set(h, p.viewers);
  }
  return m;
};

const curveOf = (name: string, dayStart: number, hourly: HourlyPoint[]): LaunchCurve | null => {
  const byHour = hourlyMap(hourly, dayStart);
  const h1 = byHour.get(0);
  if (h1 === undefined || h1 <= 0) return null;
  const pts = [...byHour.entries()].map(([h, v]) => ({ t: dayStart + h * 3600_000, v }));
  const base = baselineMedian(pts, dayStart);
  return { name, h1, lift: base === null || base <= 0 ? null : h1 / base, byHour };
};

/** Distance between the target launch and a comparable in regime space. */
const regimeDistance = (t: LaunchCurve, o: LaunchCurve): number => {
  let d2 = 0;
  let terms = 0;
  if (t.lift !== null && o.lift !== null) {
    d2 += (Math.log(t.lift) - Math.log(o.lift)) ** 2;
    terms += 1;
  }
  d2 += W_LOG_H1 * (Math.log(t.h1) - Math.log(o.h1)) ** 2;
  terms += 1;
  // A missing lift must not look close: push it far away.
  if (terms === 1) d2 += 25;
  return Math.sqrt(d2);
};

/** Linear-interpolation quantile of a sorted ascending array. */
const quantile = (sorted: number[], tau: number): number => {
  const i = tau * (sorted.length - 1);
  const lo = Math.floor(i);
  const hi = Math.min(lo + 1, sorted.length - 1);
  const f = i - lo;
  return sorted[lo] * (1 - f) + sorted[hi] * f;
};

/** Empirical quantiles of log-ratios for one hour over a pool of curves. */
const hourQuantilesFromPool = (pool: LaunchCurve[], h: number): number[] | null => {
  const ratios: number[] = [];
  for (const c of pool) {
    const v = c.byHour.get(h);
    if (v !== undefined && v > 0) ratios.push(Math.log(v / c.h1));
  }
  if (ratios.length < 5) return null;
  ratios.sort((a, b) => a - b);
  return LEVELS.map((t) => quantile(ratios, t));
};

const round2 = (x: number): number => Math.round(x * 100) / 100;

/** House persistence quantiles for one hour (fallback only). */
const fallbackQuantiles = (p1: number, h: number): number[] =>
  Z.map((z) => round2(p1 * Math.exp(z * FALLBACK_SIGMA * Math.sqrt(h - 1))));

const forecastCase = (c: CaseFile): Array<{ hour: number; q: number[] }> => {
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error('case has no hour-1 print');
  const p1 = Math.max(1, first.viewers);
  const dayStart = parseTime((c as unknown as { day_start: string }).day_start);

  // Target launch's own pre-release baseline -> its lift.
  let targetLift: number | null = null;
  if (c.history_hourly && c.history_hourly.length > 0) {
    const pts = c.history_hourly
      .filter((p) => p.viewers !== null && p.viewers !== undefined)
      .map((p) => ({ t: parseTime(p.from), v: p.viewers as number }));
    const base = baselineMedian(pts, dayStart);
    if (base !== null && base > 0) targetLift = p1 / base;
  } else if (c.history && c.history.length > 0) {
    const pts = c.history
      .filter((p) => p.viewers !== null && p.viewers !== undefined)
      .map((p) => ({ t: parseTime(p.at), v: p.viewers as number }));
    const base = baselineMedian(pts, dayStart);
    if (base !== null && base > 0) targetLift = p1 / base;
  }
  const target: LaunchCurve = { name: 'target', h1: p1, lift: targetLift, byHour: new Map() };

  // Comparable launch curves (only those with a usable hour-1 print).
  const curves: LaunchCurve[] = [];
  for (const comp of c.comparables ?? []) {
    const curve = curveOf(comp.name, parseTime(comp.day_start), comp.hourly ?? []);
    if (curve) curves.push(curve);
  }

  const neighbours = [...curves]
    .sort((a, b) => regimeDistance(target, a) - regimeDistance(target, b) || (a.name < b.name ? -1 : 1))
    .slice(0, K_NEIGHBOURS);

  // Log-quantiles per hour from the neighbour pool (fallback: all curves,
  // then persistence), then a 3-hour median smooth to calm pool noise.
  const logQ = new Map<number, number[]>();
  for (const { hour: h } of c.forecast) {
    let qs = hourQuantilesFromPool(neighbours, h) ?? hourQuantilesFromPool(curves, h);
    if (qs === null) {
      logQ.set(h, fallbackQuantiles(p1, h).map((q) => Math.log(q / p1)));
    } else {
      logQ.set(h, qs.map((r) => r));
    }
  }
  const hours = c.forecast.map(({ hour }) => hour).sort((a, b) => a - b);
  const smoothed = new Map<number, number[]>();
  for (let i = 0; i < hours.length; i++) {
    const hs = hours.slice(Math.max(0, i - 1), i + 2);
    const row = LEVELS.map((_, j) => median(hs.map((h) => (logQ.get(h) as number[])[j])));
    smoothed.set(hours[i], row);
  }

  return hours.map((h) => {
    const logRow = smoothed.get(h) as number[];
    // A 90% band narrower than MIN_LOG_SPREAD_90 would claim more
    // certainty than 12 neighbours can support, so widen it around the
    // median. Leave-one-out pinball is unchanged at this floor (0.4950
    // vs 0.4952 without it), and it guards the final case, where the
    // neighbour pool's evening hours can agree more than they should.
    const widened = [...logRow];
    if (logRow[4] - logRow[0] < MIN_LOG_SPREAD_90) {
      widened[0] = Math.min(logRow[0], logRow[2] - MIN_LOG_SPREAD_90 / 2);
      widened[4] = Math.max(logRow[4], logRow[2] + MIN_LOG_SPREAD_90 / 2);
      widened[1] = Math.min(widened[1], logRow[2] - MIN_LOG_SPREAD_90 / 4);
      widened[3] = Math.max(widened[3], logRow[2] + MIN_LOG_SPREAD_90 / 4);
    }
    const row = widened.map((lr) => p1 * Math.exp(lr));
    // Enforce non-decreasing quantiles, finite, in (0, 1e9].
    const q: number[] = [];
    let prev = 0;
    for (const v of row) {
      let x = Number.isFinite(v) ? v : p1;
      x = Math.min(1e9, Math.max(prev, x));
      x = Math.max(x, 0.01);
      q.push(round2(x));
      prev = x;
    }
    return { hour: h, q };
  });
};

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const hours = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
