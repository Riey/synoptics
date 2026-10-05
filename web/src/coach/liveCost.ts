/**
 * Conservative cost accounting for the paid calls this page makes, over the LIFETIME OF THIS PAGE: the
 * startup target selection and the guide lane's plan/follow/confirm calls. Restarting a run does not reset
 * it, so a reply that lands just after a restart is still counted and nothing already spent is un-spent.
 *
 * The only priced model is the one this app is deployed with, deepseek-flash (DeepSeek-V4.1-Flash), at
 * the OFFICIAL published peak rates (USD per 1M tokens): cached input $0.006, uncached input $0.30,
 * output $1.20 — i.e. exactly double the off-peak rates, which makes "peak" the conservative estimate
 * for a known token count. Any other model, or a response that returns no usage at all, is UNKNOWN:
 * it is counted as an unpriced attempt and never folded in as $0. The local follower is compute on the
 * user's own machine with no API charge (its energy cost is real and deliberately not counted here).
 *
 * This is an estimate built from the provider's own reported usage, not an invoice, and not a spending
 * cap: the enforceable limits are the request caps and the dispatch floors, not this number.
 */

import type { Usage } from '../generated/api.generated';

/** The only model this app has official pricing for. */
export const PRICED_COST_MODEL = 'deepseek-flash';

/** The provider that serves the priced model. */
export const PRICED_COST_PROVIDER = 'deepseek';

/** Official peak rates, USD per 1M tokens (https://api-docs.deepseek.com/quick_start/pricing/). */
export const DEEPSEEK_FLASH_PEAK_USD_PER_MILLION = {
  cachedInput: 0.006,
  missInput: 0.3,
  output: 1.2,
} as const;

export interface LiveUsageTotals {
  /** Prompt tokens reported by the provider, cached and uncached together. */
  input: number;
  /** The subset of `input` the provider reported as a cache hit. */
  cachedInput: number;
  output: number;
  /** Attempts whose usage never came back (failed, aborted, or unpriced model): unknown, not zero. */
  unknownAttempts: number;
  /**
   * Calls whose tokens this client has NO published rate for. Their tokens are kept out of
   * `input`/`output` (which `peakCostUsd` prices at the DeepSeek rates) so they can never be reported as
   * if every token had cost a DeepSeek token.
   */
  unpricedStages: number;
  unpricedInput: number;
  unpricedOutput: number;
  /**
   * Local inference stages. There is no API charge for them at all, so they are neither priced nor
   * "unknown spend"; the machine's energy cost is real but is deliberately NOT counted here.
   */
  localStages: number;
  localInput: number;
  localOutput: number;
}

export const EMPTY_USAGE_TOTALS: LiveUsageTotals = {
  input: 0,
  cachedInput: 0,
  output: 0,
  unknownAttempts: 0,
  unpricedStages: 0,
  unpricedInput: 0,
  unpricedOutput: 0,
  localStages: 0,
  localInput: 0,
  localOutput: 0,
};

/**
 * Fold one provider-reported usage into the running totals.
 *
 * A report missing EITHER half is incomplete: whatever came back is still real, reported spend and is
 * counted, but the attempt is flagged so the totals are never read as a complete account of the run. A
 * missing cache-hit split is not a missing half — it is folded in as misses, the expensive side.
 */
export function addUsage(totals: LiveUsageTotals, usage: Usage | null | undefined): LiveUsageTotals {
  const input = typeof usage?.input_tokens === 'number' ? usage.input_tokens : 0;
  const output = typeof usage?.output_tokens === 'number' ? usage.output_tokens : 0;
  const cachedInput = typeof usage?.cached_input_tokens === 'number' ? usage.cached_input_tokens : 0;
  const incomplete = typeof usage?.input_tokens !== 'number' || typeof usage?.output_tokens !== 'number';
  return {
    ...totals,
    input: totals.input + input,
    cachedInput: totals.cachedInput + cachedInput,
    output: totals.output + output,
    unknownAttempts: totals.unknownAttempts + (incomplete ? 1 : 0),
  };
}

/**
 * Fold one provider-reported usage into the running totals for one paid call
 * (the startup target selection, a paid guide call): priced only when this client has published rates for exactly that
 * provider+model, otherwise counted as reported-but-unpriced so its tokens can never be priced as
 * DeepSeek tokens. A `null`/incomplete usage is an unknown attempt, never $0.
 */
export function addProviderUsage(
  totals: LiveUsageTotals,
  provider: string,
  model: string | null | undefined,
  usage: Usage | null | undefined
): LiveUsageTotals {
  if (provider === PRICED_COST_PROVIDER && model === PRICED_COST_MODEL) {
    return addUsage(totals, usage);
  }
  const input = typeof usage?.input_tokens === 'number' ? usage.input_tokens : 0;
  const output = typeof usage?.output_tokens === 'number' ? usage.output_tokens : 0;
  const reported = typeof usage?.input_tokens === 'number' && typeof usage?.output_tokens === 'number';
  return {
    ...totals,
    unpricedStages: totals.unpricedStages + 1,
    unpricedInput: totals.unpricedInput + input,
    unpricedOutput: totals.unpricedOutput + output,
    unknownAttempts: totals.unknownAttempts + (reported ? 0 : 1),
  };
}

/**
 * Conservative peak-rate estimate for the known tokens. An unreported cache-hit split is counted as
 * misses (the expensive side), so this is an upper bound for the tokens that were actually reported.
 */
export function peakCostUsd(totals: LiveUsageTotals): number {
  const missInput = Math.max(0, totals.input - totals.cachedInput);
  return (
    (totals.cachedInput * DEEPSEEK_FLASH_PEAK_USD_PER_MILLION.cachedInput +
      missInput * DEEPSEEK_FLASH_PEAK_USD_PER_MILLION.missInput +
      totals.output * DEEPSEEK_FLASH_PEAK_USD_PER_MILLION.output) /
    1_000_000
  );
}

// ------------------------------------------------------------------------------------- guide lane stages

/** The four guide-lane calls, each accounted for separately. */
export type GuideStage = 'plan' | 'follow' | 'confirm' | 'talk';

export const GUIDE_STAGES: readonly GuideStage[] = ['plan', 'follow', 'confirm', 'talk'];

/** What the UI says for a call served by the local follower: compute on the user's own machine, no API bill. */
export const GUIDE_LOCAL_COST_TEXT = 'API 요금 없음';

/** The provider token the backend reports for the local (llama.cpp) follower. */
export const LOCAL_PROVIDER = 'local';
/** Followers that run on the app's own machine: counted as compute, never as API spend. */
export const LOCAL_PROVIDERS: readonly string[] = [LOCAL_PROVIDER, 'clef'];

export type GuideUsageTotals = Record<GuideStage, LiveUsageTotals>;

export const EMPTY_GUIDE_USAGE: GuideUsageTotals = {
  plan: EMPTY_USAGE_TOTALS,
  follow: EMPTY_USAGE_TOTALS,
  confirm: EMPTY_USAGE_TOTALS,
  talk: EMPTY_USAGE_TOTALS,
};

/**
 * One guide call that came back WITH an answer (accepted or not — a discarded answer was still served).
 *
 * Priced per provider: deepseek-flash at the DeepSeek peak rates, the local followers (llama.cpp, Clef) as compute with no
 * API charge (its tokens are kept, its missing usage is not "unknown spend" because nothing was billed),
 * anything else as reported-but-unpriced.
 */
export function addGuideCall(
  totals: GuideUsageTotals,
  stage: GuideStage,
  provider: string,
  model: string | null | undefined,
  usage: Usage | null | undefined
): GuideUsageTotals {
  const current = totals[stage];
  let next: LiveUsageTotals;
  if (LOCAL_PROVIDERS.includes(provider)) {
    const input = typeof usage?.input_tokens === 'number' ? usage.input_tokens : 0;
    const output = typeof usage?.output_tokens === 'number' ? usage.output_tokens : 0;
    next = {
      ...current,
      localStages: current.localStages + 1,
      localInput: current.localInput + input,
      localOutput: current.localOutput + output,
    };
  } else {
    next = addProviderUsage(current, provider, model, usage);
  }
  return { ...totals, [stage]: next };
}

/**
 * A guide call that was dispatched to a paid provider and produced no usage (transport failure, abort,
 * a refusal after the provider ran). Unknown spend, never $0. A local call costs no API money, so it is
 * not counted at all.
 */
export function addGuideUnknownAttempt(totals: GuideUsageTotals, stage: GuideStage, lane: 'local' | 'remote'): GuideUsageTotals {
  if (lane === 'local') return totals;
  return { ...totals, [stage]: addUsage(totals[stage], null) };
}

/** The money estimate of one stage: priced tokens only (DeepSeek peak rates). */
export function guideStageCostUsd(totals: GuideUsageTotals, stage: GuideStage): number {
  return peakCostUsd(totals[stage]);
}

/** Korean one-liner for one stage: local calls say there is no API charge, paid calls show the estimate. */
export function describeGuideStageCost(totals: GuideUsageTotals, stage: GuideStage): string {
  const t = totals[stage];
  const parts: string[] = [];
  if (t.input + t.output > 0) parts.push(`$${peakCostUsd(t).toFixed(6)}(피크 단가 추정)`);
  if (t.localStages > 0) parts.push(`로컬 ${t.localStages}회 · ${GUIDE_LOCAL_COST_TEXT}`);
  if (t.unpricedStages > 0) parts.push(`단가 미확인 ${t.unpricedStages}회`);
  if (t.unknownAttempts > 0) parts.push(`사용량 미수신 ${t.unknownAttempts}회(비용 불명)`);
  return parts.length > 0 ? parts.join(' · ') : '없음';
}
