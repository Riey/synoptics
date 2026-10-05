/**
 * Which core logic executes a guide run (design: selectable classic vs sequential), chosen before Start.
 *
 * The choice is independent of `plan_model`: any plan model may write a plan for either core. It is frozen
 * into the run that Start creates and sent as `core_mode` on that run's first plan request only; the server
 * stores it with the accepted plan and it does not change through revisions. A missing or different echo
 * refuses the plan rather than silently running the other core.
 *
 * - `classic`: the follower judges the checklist from the asked step on; a step it sees done advances at
 *   once and an asynchronous upper confirm can revert it.
 * - `sequential`: the follower judges only the CURRENT step, and its `yes` merely nominates that step. The
 *   screen moves exactly one step, and only when the upper model answers `step_done: yes` for that same
 *   step, plan revision, step activation and frame (`intent/sequentialCore.ts`).
 */

export const CORE_MODES = ['classic', 'sequential'] as const;
export type CoreMode = (typeof CORE_MODES)[number];
/** The server's default; the control starts here and never invents a third option. */
export const DEFAULT_CORE_MODE: CoreMode = 'classic';

export interface CoreModeOption {
  /** The option's own label. */
  label: string;
  /** What the core does on the next run, in one sentence (shown under the group). */
  help: string;
}

/** One entry per core: the closed set of choices and what each one does. */
export const CORE_MODE_OPTION: Record<CoreMode, CoreModeOption> = {
  classic: {
    label: '기존 코어',
    help: '남은 단계들을 함께 판정합니다. 먼저 진행한 뒤 상위 모델이 재확인하고, 어긋나면 되돌립니다.',
  },
  sequential: {
    label: '순차 코어',
    help: '현재 단계만 판정합니다. 상위 모델 확인이 끝나야 다음 한 단계로 진행하므로 확인을 기다리는 시간이 있습니다.',
  },
};

/** The note under the picker: what the chosen core will do, and that it is not the plan model. */
export function coreModeNote(mode: CoreMode): string {
  return `${CORE_MODE_OPTION[mode].help} 계획 모델과는 별개로 고릅니다.`;
}
