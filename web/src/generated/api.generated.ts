/* eslint-disable */
/**
 * Automatically generated from web/src/generated/api.schema.json. DO NOT EDIT MANUALLY.
 */

export type Kind = "action";
export type Anchor = string;
export type Action =
  | "press"
  | "grasp"
  | "move"
  | "rotate"
  | "open"
  | "close"
  | "insert"
  | "remove"
  | "attach"
  | "fold"
  | "place"
  | "fit"
  | "screw"
  | "hold"
  | "flip"
  | "pull"
  | "push"
  | "align"
  | "connect"
  | "disconnect"
  | "bend";
export type Direction =
  "up" | "down" | "left" | "right" | "clockwise" | "counterclockwise" | "tighten" | "loosen" | "none";
export type AnchorId = string;
export type TrackId = string;
export type Generation = number;
export type AnchorId1 = string;
export type Role = "target";
export type Label = string;
export type RunId = string;
export type TrackId1 = string;
export type Generation1 = number;
export type State = "acquiring" | "tracking" | "occluded" | "lost" | "unavailable";
export type X = number;
export type Y = number;
export type Width = number;
export type Height = number;
export type AnchorId2 = string;
export type Matches = "yes" | "no" | "unsure";
export type Version = string;
export type Commit = string;
export type BuiltAt = string;
export type Ended = boolean;
export type MaterialId = string;
export type Version1 = string | null;
export type Locator = string | null;
export type Quote = string | null;
export type Kind1 = "focus";
export type Anchor1 = string;
export type Pad = number;
export type SessionId = string;
export type ConsentAi = boolean;
export type FrameId = string;
export type ImageBase64 = string;
export type Label1 = string | null;
export type PlanId = string;
export type PlanRevision = number;
export type UserGoal = string;
export type Anchors = AnchorRef[];
export type Trigger = "goal_check" | "step_done" | "unsure_twice" | "replan" | "target_left";
export type IntentSeq = number;
export type TaskEpoch = string;
export type RunId1 = string;
export type CurrentStep = string | null;
export type FollowChecks = StepCheck[] | null;
export type StepId = string;
export type Visible = "yes" | "no" | "unsure";
export type ExitEdge = ("left" | "right" | "top" | "bottom" | "none") | null;
export type Status = "visually_satisfied" | "in_progress" | "uncertain" | "not_applicable";
export type Rationale = string;
export type EvidenceKind = "observed_scene" | "uncertain_view";
export type Id = string;
export type Say = string;
export type Details = string | null;
export type Kind2 = "label";
export type Anchor2 = string;
export type Text = string;
export type Commands = (FocusAnchorCommand | LabelAnchorCommand | ActionAnchorCommand)[];
export type DoneWhen = string;
export type ConditionKind = "state" | "event";
export type Check = "visual" | "user" | "measure";
export type Required = boolean;
export type GoalRequired = boolean;
export type Requires = string[];
export type Targets = string[];
export type Steps = GuideStep[];
export type NeedsClarification = boolean;
export type ClarificationPrompt = string | null;
export type StepCheck1 = ("yes" | "no" | "unsure") | null;
export type InferredDone = ("yes" | "no" | "unsure") | null;
export type StepChecks = StepCheck[] | null;
export type StepId1 = string | null;
export type IntentId = string;
export type FrameId1 = string;
export type IntentSeq1 = number;
export type Trigger1 = "goal_check" | "step_done" | "unsure_twice" | "replan" | "target_left";
export type TaskRevision = number;
export type PlanRevision1 = number;
export type AnchorsEcho = AnchorEcho[];
export type Provider = string;
export type Model = string | null;
export type InputTokens = number | null;
export type OutputTokens = number | null;
export type TotalTokens = number | null;
export type CachedInputTokens = number | null;
export type ThoughtTokens = number | null;
export type ToolUseTokens = number | null;
export type LatencyMs = number;
export type SessionId1 = string;
export type ConsentAi1 = boolean;
export type PlanId1 = string;
export type PlanRevision2 = number;
export type CurrentStep1 = string;
export type Anchors1 = AnchorRef[];
export type Trigger2 = "acquired" | "target_moved" | "target_changed" | "anchor_lost" | "manual" | "heartbeat";
export type IntentSeq2 = number;
export type StepChecks1 = StepCheck[];
export type AnchorVerdicts = AnchorVerdict[];
export type GoalSeen = "yes" | "no" | "unsure";
export type IntentId1 = string;
export type FrameId2 = string;
export type IntentSeq3 = number;
export type Trigger3 = "acquired" | "target_moved" | "target_changed" | "anchor_lost" | "manual" | "heartbeat";
export type TaskRevision1 = number;
export type PlanRevision3 = number;
export type AnchorsEcho1 = AnchorEcho[];
export type Provider1 = string;
export type Model1 = string | null;
export type LatencyMs1 = number;
export type NeedsReselect = boolean;
export type Ready = boolean;
export type Model2 = string;
export type Provider2 = string;
export type AccessCodeRequired = boolean;
export type AccessMode = ("code" | "local" | "open") | null;
export type TrackerReady = boolean;
export type GuideReady = boolean;
export type PlanReady = boolean;
export type FollowProvider = ("local" | "clef" | "deepseek") | null;
export type FollowReady = boolean;
export type SessionId2 = string;
export type PlanId2 = string;
export type PlanRevision4 = number;
export type Steps1 = GuideStep[];
export type GoalWhen = string;
export type UserGoal1 = string | null;
export type Context = string | null;
export type PlanId3 = string;
export type PlanRevision5 = number;
export type CoreMode = "classic" | "sequential" | "graph";
export type TaskRevision2 = number;
export type Steps2 = GuideStep[];
export type GoalWhen1 = string;
export type UserGoal2 = string;
export type SessionId3 = string;
export type ConsentAi2 = boolean;
export type UserGoal3 = string;
export type Context1 = string | null;
export type Materials = MaterialInput[] | null;
export type Title = string;
export type Version2 = string | null;
export type Text1 = string;
export type PlanModel = "deepseek:high" | "astra:high";
export type CoreMode1 = "classic" | "sequential" | "graph";
export type Assisted = boolean;
export type Research = boolean;
export type Question = string;
export type Answer = string;
export type Answers = PlanAnswer[];
export type ReferenceImages = SceneInput[];
/**
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "TargetSelectionStatus".
 */
export type TargetSelectionStatus = "selected" | "no_target" | "uncertain";
export type Target = string | null;
export type Rationale1 = string | null;
export type Steps3 = GuideStep[];
export type GoalWhen2 = string;
export type EvidenceKind1 = "observed_scene" | "uncertain_view";
export type NeedsClarification1 = boolean;
export type ClarificationPrompt1 = string | null;
export type Url = string;
export type Title1 = string;
export type Summary = string;
export type ResearchSources = ResearchSource[];
export type PlanId4 = string;
export type PlanRevision6 = number;
export type CoreMode2 = "classic" | "sequential" | "graph";
export type FrameId3 = string;
export type Provider3 = string;
export type Model3 = string | null;
export type SessionId4 = string;
export type PlanId5 = string;
export type PlanRevision7 = number;
export type SessionId5 = string;
export type ConsentAi3 = boolean;
export type PlanId6 = string;
export type PlanRevision8 = number;
export type CurrentStep2 = string;
export type Anchors2 = AnchorRef[];
export type Utterance = string;
export type PrevUtterance = string | null;
export type PrevReply = string | null;
export type FollowChecks1 = StepCheck[] | null;
export type ReplanAllowed = boolean;
export type IntentSeq4 = number;
export type UserSaysDone = boolean;
export type Reply = string;
export type Spoken = string;
export type StepSay = string | null;
export type Target1 = string | null;
export type StepMark = ("done" | "skipped") | null;
export type GoTo = string | null;
export type IntentId2 = string;
export type FrameId4 = string;
export type IntentSeq5 = number;
export type TaskRevision3 = number;
export type PlanRevision9 = number;
export type AnchorsEcho2 = AnchorEcho[];
export type Provider4 = string;
export type Model4 = string | null;
export type LatencyMs2 = number;
export type UserSaysDone1 = boolean;
export type Reply1 = string;
export type Spoken1 = string;
export type StepSay1 = string | null;
export type Target2 = string | null;
export type StepMark1 = ("done" | "skipped") | null;
export type GoTo1 = string | null;
export type Ready1 = boolean;
export type Model5 = string;
export type Provider5 = string;
export type AccessCodeRequired1 = boolean;
export type AccessMode1 = ("code" | "local" | "open") | null;
export type TrackerReady1 = boolean;
export type SessionId6 = string;
export type Filename = string;
export type ContentBase64 = string;
export type AccessCode = string | null;
export type SessionId7 = string;
export type SessionId8 = string;
export type RunId2 = string;
export type Action1 = "start" | "stop";
export type StartSeq = number | null;
export type Target3 = string | null;
export type RunId3 = string;
export type Active = boolean;
export type Target4 = string;
export type SessionId9 = string;
export type RunId4 = string;
export type FrameId5 = string;
export type FrameSeq = number;
export type SessionId10 = string;
export type RunId5 = string;
export type FrameId6 = string;
export type FrameSeq1 = number;
export type ImageBase641 = string;
export type RunId6 = string;
export type Target5 = string;
export type TrackId2 = string;
export type Generation2 = number;
export type State1 = "acquiring" | "tracking" | "occluded" | "lost" | "unavailable";
export type Transition = "none" | "acquired" | "tracking" | "occluded" | "lost" | "reacquired" | "ambiguous";
export type Source = ("grounder" | "user") | null;
export type FrameId7 = string;
export type FrameSeq2 = number;
export type Version3 = number;
export type Confidence = number | null;
export type IngestAgeMs = number;
export type SessionId11 = string;
export type ConsentAi4 = boolean;
export type UserGoal4 = string | null;
export type Context2 = string | null;
export type SelectionId = string;
export type FrameId8 = string;
export type Target6 = string | null;
export type Rationale2 = string | null;
export type Provider6 = string;
export type Model6 = string | null;

export interface GuidanceApiContracts {
  ActionAnchorCommand?: ActionAnchorCommand;
  AnchorEcho?: AnchorEcho;
  AnchorRef?: AnchorRef;
  AnchorVerdict?: AnchorVerdict;
  BuildInfo?: BuildInfo;
  EndSessionResponse?: EndSessionResponse;
  Evidence?: Evidence;
  FocusAnchorCommand?: FocusAnchorCommand;
  GuideConfirmRequest?: GuideConfirmRequest;
  GuideConfirmResponse?: GuideConfirmResponse;
  GuideFence?: GuideFence;
  GuideFollowRequest?: GuideFollowRequest;
  GuideFollowResponse?: GuideFollowResponse;
  GuideHealthResponse?: GuideHealthResponse;
  GuidePlanApproveRequest?: GuidePlanApproveRequest;
  GuidePlanCurrentResponse?: GuidePlanCurrentResponse;
  GuidePlanRequest?: GuidePlanRequest;
  GuidePlanResponse?: GuidePlanResponse;
  GuidePlanRevertRequest?: GuidePlanRevertRequest;
  GuideReplan?: GuideReplan;
  GuideStep?: GuideStep;
  GuideTalkRequest?: GuideTalkRequest;
  GuideTalkResponse?: GuideTalkResponse;
  GuideTalkToolOutput?: GuideTalkToolOutput;
  HealthResponse?: HealthResponse;
  LabelAnchorCommand?: LabelAnchorCommand;
  ManualImportRequest?: ManualImportRequest;
  MaterialInput?: MaterialInput;
  SceneInput?: SceneInput;
  SessionRequest?: SessionRequest;
  SessionResponse?: SessionResponse;
  TrackControlRequest?: TrackControlRequest;
  TrackControlResponse?: TrackControlResponse;
  TrackFrameQuery?: TrackFrameQuery;
  TrackFrameRequest?: TrackFrameRequest;
  TrackFrameResponse?: TrackFrameResponse;
  TrackSelectRequest?: TrackSelectRequest;
  TrackSelectResponse?: TrackSelectResponse;
  Usage?: Usage;
  VisualGoalStatus?: VisualGoalStatus;
}
/**
 * Action glyph attached to the anchor's live box (drawn by the client). Marks the whole target, not subpart.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "ActionAnchorCommand".
 */
export interface ActionAnchorCommand {
  kind: Kind;
  anchor: Anchor;
  action: Action;
  direction: Direction;
}
/**
 * Which anchor identity an answer was computed against (a generation change makes it text-only).
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "AnchorEcho".
 */
export interface AnchorEcho {
  anchor_id: AnchorId;
  track_id: TrackId;
  generation: Generation;
}
/**
 * The client's registry snapshot of one anchor, captured with the request frame. Not server authority.
 *
 * ``box`` is the tracker's box on the SAME capture as ``scene`` and exists iff ``state == "tracking"``
 * (the tracking contract's own rule). The server draws it on its copy of the frame together with
 * ``anchor_id`` (Set-of-Mark) so the model can refer to the object by id without ever writing a position.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "AnchorRef".
 */
export interface AnchorRef {
  anchor_id: AnchorId1;
  role: Role;
  label: Label;
  run_id: RunId;
  track_id: TrackId1;
  generation: Generation1;
  state: State;
  box?: Box | null;
}
/**
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "Box".
 */
export interface Box {
  x: X;
  y: Y;
  width: Width;
  height: Height;
}
/**
 * Does the box drawn with this id still contain the plan's target object?
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "AnchorVerdict".
 */
export interface AnchorVerdict {
  anchor_id: AnchorId2;
  matches: Matches;
}
/**
 * Which build answers (``backend/app/build.py``): project version, short commit, build time (ISO).
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "BuildInfo".
 */
export interface BuildInfo {
  version: Version;
  commit: Commit;
  built_at: BuiltAt;
}
/**
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "EndSessionResponse".
 */
export interface EndSessionResponse {
  ended: Ended;
}
/**
 * §15.2: which material (and where in it) a step's instruction came from.
 *
 * ``material_id`` names one of the request's ``materials`` (``m1`` .. in request order). The route knows that
 * list and DROPS an evidence it cannot ground — an id not among the materials, a ``version`` that does not
 * equal the material's own, a ``quote`` longer than ``EVIDENCE_QUOTE_MAX`` or not a whitespace-normalised
 * excerpt of the material's text, or a field outside its bound (the drop is logged; nothing is invented) — so
 * no fabricated or malformed citation ever reaches a client. A quote is never silently cut to fit: an
 * over-long quote is ungrounded and dropped, and a kept quote is stored whitespace-folded. A blank optional
 * string is absent.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "Evidence".
 */
export interface Evidence {
  material_id: MaterialId;
  version?: Version1;
  locator?: Locator;
  quote?: Quote;
}
/**
 * Emphasis ring around the anchor's live box; ``pad`` is a fraction of the box, not a position.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "FocusAnchorCommand".
 */
export interface FocusAnchorCommand {
  kind: Kind1;
  anchor: Anchor1;
  pad?: Pad;
}
/**
 * Ask the large model whether the goal is visibly done (or the plan must change) on a fresh frame.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "GuideConfirmRequest".
 */
export interface GuideConfirmRequest {
  session_id: SessionId;
  consent_ai: ConsentAi;
  scene: SceneInput;
  plan_id: PlanId;
  plan_revision: PlanRevision;
  user_goal: UserGoal;
  anchors: Anchors;
  trigger: Trigger;
  intent_seq: IntentSeq;
  fence: GuideFence;
  current_step?: CurrentStep;
  follow_checks?: FollowChecks;
  before_scene?: SceneInput | null;
  exit_edge?: ExitEdge;
}
/**
 * The single current frame the advice is drawn against.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "SceneInput".
 */
export interface SceneInput {
  frame_id: FrameId;
  image_base64: ImageBase64;
  label?: Label1;
}
/**
 * The client's own fence for one intent call, echoed back verbatim so a late answer is recognisable.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "GuideFence".
 */
export interface GuideFence {
  task_epoch: TaskEpoch;
  run_id: RunId1;
}
/**
 * Is this step's ``done_when`` (a visible result state) visible in the frame right now?
 *
 * A state observation only: it says nothing about order, about the action, or about which step is current.
 * ``unsure`` when the thing ``done_when`` talks about is out of view, covered or blurred: absence is
 * neither ``yes`` nor ``no``.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "StepCheck".
 */
export interface StepCheck {
  step_id: StepId;
  visible: Visible;
}
/**
 * The judge's answer plus the same acceptance envelope as follow.
 *
 * ``plan_revision`` is the revision AFTER this call: bumped when ``replan`` was applied to the stored plan.
 * ``step_id`` echoes the request's ``current_step`` on a ``step_done`` confirm, the one trigger that
 * carries ``step_check``; both are present together or not at all. Under ``graph`` that is the step the
 * upper was asked about — an earlier completed step or a future eligible one — and never a different
 * "suggested focus": the id is the request's, echoed verbatim.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "GuideConfirmResponse".
 */
export interface GuideConfirmResponse {
  goal_status: VisualGoalStatus;
  evidence_kind: EvidenceKind;
  replan?: GuideReplan | null;
  needs_clarification: NeedsClarification;
  clarification_prompt?: ClarificationPrompt;
  step_check?: StepCheck1;
  inferred_done?: InferredDone;
  step_checks?: StepChecks;
  step_id?: StepId1;
  intent_id: IntentId;
  frame_id: FrameId1;
  intent_seq: IntentSeq1;
  trigger: Trigger1;
  task_revision: TaskRevision;
  plan_revision: PlanRevision1;
  fence_echo: GuideFence;
  anchors_echo: AnchorsEcho;
  provider: Provider;
  model?: Model;
  usage?: Usage | null;
  latency_ms: LatencyMs;
}
/**
 * Whole-goal visual assessment for the application/user, distinct from step-level advice.
 *
 * - visually_satisfied: Current scene clearly and directly shows the user goal has been accomplished.
 * - in_progress: Goal is active and work visibly remains to be done.
 * - uncertain: Visual evidence in the current scene is insufficient, blurry, occluded, or missing targets.
 * - not_applicable: No explicit user goal was supplied.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "VisualGoalStatus".
 */
export interface VisualGoalStatus {
  status: Status;
  rationale: Rationale;
}
/**
 * Replacement steps for the current plan (same step shape, role-anchored commands only).
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "GuideReplan".
 */
export interface GuideReplan {
  steps: Steps;
}
/**
 * One plan step: what the screen says, what to draw on the target, and the visible end condition.
 *
 * Commands refer to the role ``"target"`` only. The step is plain text plus that role reference, which is
 * why the server can keep a plan in the session without keeping an image or a coordinate.
 *
 * A plan (or a confirm replan) is written before any anchor id exists, so ``"target"`` is the only thing a
 * step command can refer to: whatever ``anchor`` value the model wrote (DeepSeek wrote ``"a1"`` on 3/3
 * increase_temp plans, 2026-10-01) is rewritten to the role BEFORE validation instead of failing the whole
 * plan. Nothing else in the command is touched, and a command without an ``anchor`` is still refused. The
 * follow/confirm request anchors (``AnchorRef``) are a different model and are validated as before.
 *
 * The optional §15 review fields are defaulted, so a caller that does not use them is unchanged:
 *
 * * ``condition_kind`` is what ``done_when`` describes — ``state`` (default) while it holds, ``event`` once
 *   verified. Metadata for the client's graph state; it changes nothing about what the follower looks at.
 * * ``check`` is what decides the step is finished (``visual`` on the frame; ``user``/``measure`` only when
 *   the user says so — a check the camera cannot make).
 * * ``required`` marks a mandatory inspection point that must never be skipped.
 * * ``goal_required`` says whose condition this is: the USER'S GOAL (the default) or only safety,
 *   preparation, teardown or optional support around it. It is orthogonal to ``required``: a safety
 *   precondition such as unplugging the power before touching the wiring is ``goal_required=False`` WITH
 *   ``required=True`` (the user must still confirm it, but its state is not evidence the goal was reached),
 *   an optional supporting step is ``goal_required=False, required=False``, and only ``goal_required=True``
 *   steps are goal evidence — so the client never counts a procedure step toward goal completion and never
 *   revokes a completed goal when a precondition later stops holding.
 * * ``requires`` names steps of the SAME plan that must be finished before this one is reachable; the plan
 *   graph (unique ids, resolvable ``requires``, no self-reference, no cycle) is checked plan-wide.
 * * ``targets`` are the step's logical subjects (parts, boxes, tools) — labels, never coordinates.
 * * ``evidence`` cites the material the instruction came from, when the request supplied materials.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "GuideStep".
 */
export interface GuideStep {
  id: Id;
  say: Say;
  details?: Details;
  commands: Commands;
  done_when: DoneWhen;
  condition_kind?: ConditionKind;
  check?: Check;
  required?: Required;
  goal_required?: GoalRequired;
  requires?: Requires;
  targets?: Targets;
  evidence?: Evidence | null;
}
/**
 * A short callout attached above the anchor's live box (screen-space offset, drawn by the client).
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "LabelAnchorCommand".
 */
export interface LabelAnchorCommand {
  kind: Kind2;
  anchor: Anchor2;
  text: Text;
}
/**
 * Provider token accounting for one call. Belongs to the app/provider layer, not the renderer SDK.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "Usage".
 */
export interface Usage {
  input_tokens?: InputTokens;
  output_tokens?: OutputTokens;
  total_tokens?: TotalTokens;
  cached_input_tokens?: CachedInputTokens;
  thought_tokens?: ThoughtTokens;
  tool_use_tokens?: ToolUseTokens;
}
/**
 * One follow judgement about the frame captured right now, against the current plan revision.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "GuideFollowRequest".
 */
export interface GuideFollowRequest {
  session_id: SessionId1;
  consent_ai: ConsentAi1;
  scene: SceneInput;
  plan_id: PlanId1;
  plan_revision: PlanRevision2;
  current_step: CurrentStep1;
  anchors: Anchors1;
  trigger: Trigger2;
  intent_seq: IntentSeq2;
  fence: GuideFence;
}
/**
 * The follower's answer plus everything the client needs to accept it as bound, text-only or stale.
 *
 * ``needs_reselect`` is ``True`` exactly when some verdict is ``"no"``. This response has no
 * ``goal_status`` field at all: completion is never decided on this route.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "GuideFollowResponse".
 */
export interface GuideFollowResponse {
  step_checks: StepChecks1;
  anchor_verdicts: AnchorVerdicts;
  goal_seen: GoalSeen;
  intent_id: IntentId1;
  frame_id: FrameId2;
  intent_seq: IntentSeq3;
  trigger: Trigger3;
  task_revision: TaskRevision1;
  plan_revision: PlanRevision3;
  fence_echo: GuideFence;
  anchors_echo: AnchorsEcho1;
  provider: Provider1;
  model?: Model1;
  usage?: Usage | null;
  latency_ms: LatencyMs1;
  needs_reselect: NeedsReselect;
}
/**
 * ``/api/health`` with the guide lane's readiness, each reported independently.
 *
 * * ``guide_ready`` — the BASIC guide lane can run: the configured provider is DeepSeek and has a key. This
 *   is exactly the ``deepseek:high`` profile's capability. Global readiness is unchanged by the OpenAI lane:
 *   it never requires an OpenAI key.
 * * ``plan_models`` — per-choice configured capability, ``{'deepseek:high': bool, 'astra:high': bool}``. A
 *   ``True`` means that profile's credential is configured; it does NOT prove the key's model entitlement or
 *   remaining credit. The two entries are independent.
 * * ``plan_ready`` — aggregate any-ready: at least one ``plan_models`` entry is ``True``. A session may be
 *   opened on it alone.
 * * ``follow_provider`` — ``AISW_FOLLOW_PROVIDER`` (``None`` when it names no known follower).
 * * ``follow_ready`` — the follower can run: the local or Clef endpoint answers its health probe (cached
 *   2 s), or, for ``deepseek``, the same condition as ``guide_ready``.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "GuideHealthResponse".
 */
export interface GuideHealthResponse {
  ready: Ready;
  model: Model2;
  provider?: Provider2;
  access_code_required: AccessCodeRequired;
  access_mode?: AccessMode;
  tracker_ready?: TrackerReady;
  guide_ready?: GuideReady;
  plan_ready?: PlanReady;
  plan_models?: PlanModels;
  follow_provider?: FollowProvider;
  follow_ready?: FollowReady;
  build?: BuildInfo | null;
}
export interface PlanModels {
  [k: string]: boolean;
}
/**
 * ``POST /api/guide/plan/approve``: the client-reviewed plan becomes the session's authoritative plan.
 *
 * The client holds a plan it may have edited (changed a sentence or the order, added or removed a step);
 * approving it pins execution to exactly those steps, so a later fence names the adopted pair, not the plan
 * the model wrote. ``plan_id``/``plan_revision`` must name the session's CURRENT plan, else ``409 stale_plan``
 * with the usual body. No provider is called, no frame is read and no consent is asked: nothing here reaches a
 * model.
 *
 * ``steps`` (1-``MAX_STEPS``) are re-validated exactly like a model answer's: the §15 requires graph (unique
 * ids, resolvable and acyclic ``requires``) is fail-closed as ``422 invalid_request``. ``user_goal``/``context``
 * keep the adopted plan's own values when absent, so a client that only edits steps need not resend them.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "GuidePlanApproveRequest".
 */
export interface GuidePlanApproveRequest {
  session_id: SessionId2;
  plan_id: PlanId2;
  plan_revision: PlanRevision4;
  steps: Steps1;
  goal_when: GoalWhen;
  user_goal?: UserGoal1;
  context?: Context;
}
/**
 * ``GET /api/guide/plan/current``: the session's current plan as stored (text only), no provider call.
 *
 * The recovery read after a ``409 stale_plan`` (for example a streamed plan whose client disconnected, or a
 * replan applied by a confirm the client did not see): the client continues from exactly this
 * ``plan_id``/``plan_revision``. ``task_revision`` is the task the plan was made for (always the session's
 * current one: a task change clears the plan). ``steps`` may be empty only for a plan that selected no
 * target or asked for clarification, as on ``POST /api/guide/plan``. ``core_mode`` is the guide core the
 * plan was built for (fixed for its task), so a recovering client knows which follow semantics apply.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "GuidePlanCurrentResponse".
 */
export interface GuidePlanCurrentResponse {
  plan_id: PlanId3;
  plan_revision: PlanRevision5;
  core_mode?: CoreMode;
  task_revision: TaskRevision2;
  steps: Steps2;
  goal_when: GoalWhen1;
  user_goal: UserGoal2;
}
/**
 * Start a guide: the task text and the frame captured right now. A new plan replaces the old one.
 *
 * ``materials`` are reference documents: each is rendered into a prompt as a labelled block (``m1`` .. in
 * request order) that a step's ``evidence.material_id`` may cite. They join the session's task identity (a
 * changed list under the same goal/context is a new task) and are kept with the plan for its task, so a
 * confirm/talk replan stays grounded in the same source; they are never echoed by ``/plan/current`` and never
 * logged.
 *
 * ``plan_model`` (one of ``deepseek:high`` / ``astra:high``, default ``deepseek:high``) selects which upper
 * planning profile writes the plan: the basic DeepSeek provider at HIGH, or the OpenAI Responses adapter at
 * HIGH. The two profiles share one intent contract, one reasoning+output cap and one deadline; only the
 * issuer adapter differs, and a missing credential is a clean refusal — never a fallback to the other lane.
 * The choice joins the session's task identity too — changing it under the same goal/context is a new task,
 * so a plan made for one choice can never be inherited or revised by the other. There is no alias for it and
 * no legacy ``plan_mode``/``mode``/``manual`` field (any of those is an unknown field and refused). Once a
 * plan is accepted, confirm/talk route from that plan's own stored ``plan_model``, never from the client.
 *
 * ``core_mode`` (``classic`` by default, ``sequential`` or ``graph``) selects the guide core: how the follower
 * is asked and how the client may move the screen. ``classic`` is the whole-plan checklist (a follow answer
 * covers every step from the current one to the last). ``sequential`` scopes every follower answer to the
 * CURRENT step, so a later step's verdict can never move the screen. ``graph`` asks about EVERY step of the
 * plan, earlier and already-completed ones included, and treats the client's suggested focus as a hint only: a
 * step is committed by the upper's own ``step_done`` verdict, and an explicit ``condition_kind`` says whether
 * a step's claim can be revoked (``state``) or is a historical occurrence (``event``). All cores share one
 * follower model, mode and threshold; only the question set differs. The choice joins the session's task
 * identity too — changing it under the same goal/context is a new task, so a plan made for one core can never
 * be inherited, revised or reverted back into by the other. There is no alias for it. It is independent of
 * ``plan_model`` (which upper profile writes the plan) and of any review flag the client applies.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "GuidePlanRequest".
 */
export interface GuidePlanRequest {
  session_id: SessionId3;
  consent_ai: ConsentAi2;
  scene: SceneInput;
  user_goal: UserGoal3;
  context?: Context1;
  materials?: Materials;
  plan_model?: PlanModel;
  core_mode?: CoreMode1;
  assisted?: Assisted;
  research?: Research;
  answers?: Answers;
  reference_images?: ReferenceImages;
}
/**
 * §15.3: one reference document a plan may cite.
 *
 * ``text`` is the extracted plain text; it is rendered into a prompt as a labelled block (``m1``, ``m2``, ...
 * in request order) that a step's ``evidence.material_id`` can point at. A plan keeps the materials it was
 * made with for the life of its task (so a confirm/talk replan stays grounded in the same source) but they are
 * never echoed by ``/plan/current`` and never written to a log.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "MaterialInput".
 */
export interface MaterialInput {
  title: Title;
  version?: Version2;
  text: Text1;
}
/**
 * One answered clarification, retained as user evidence without replacing the goal.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "PlanAnswer".
 */
export interface PlanAnswer {
  question: Question;
  answer: Answer;
}
/**
 * The plan plus its server identity. ``plan_revision`` is per session and never goes backwards.
 *
 * ``core_mode`` echoes the guide core the plan was built for; it is fixed for the plan's whole task, so a
 * client never has to guess which follow semantics apply.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "GuidePlanResponse".
 */
export interface GuidePlanResponse {
  selection: TargetSelectionToolOutput;
  steps: Steps3;
  goal_when: GoalWhen2;
  evidence_kind: EvidenceKind1;
  needs_clarification: NeedsClarification1;
  clarification_prompt?: ClarificationPrompt1;
  research_sources?: ResearchSources;
  plan_id: PlanId4;
  plan_revision: PlanRevision6;
  core_mode?: CoreMode2;
  frame_id: FrameId3;
  provider: Provider3;
  model?: Model3;
  usage?: Usage | null;
}
/**
 * The provider's structured answer for one startup target-selection call.
 *
 * ``status`` is the honest closed set: ``selected`` names one object, ``no_target`` says the scene holds
 * no plausible object to track, ``uncertain`` says the frame cannot support the decision (blurred,
 * occluded, ambiguous). The latter two never carry a target, and they must say why.
 *
 * ``target`` is what the tracker's detector will be prompted with, so it is a short **English** noun
 * (``laptop``, ``screwdriver``) rather than a translated or descriptive phrase; ``rationale`` is the
 * human-facing Korean sentence. The two are never mixed, and no translation step sits between them.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "TargetSelectionToolOutput".
 */
export interface TargetSelectionToolOutput {
  status: TargetSelectionStatus;
  target?: Target;
  rationale?: Rationale1;
}
/**
 * A public source actually returned by a research tool, not an invented citation.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "ResearchSource".
 */
export interface ResearchSource {
  url: Url;
  title: Title1;
  summary: Summary;
}
/**
 * ``POST /api/guide/plan/revert``: undo the last plan change (an approve, a replan or a new plan).
 *
 * ``plan_id``/``plan_revision`` must name the session's CURRENT plan, else ``409 stale_plan``. On success the
 * session's most recent previous plan (bounded history; see ``guide.GuideSessionState``) becomes current again
 * under the NEXT ``plan_revision``, so every fence keeps moving forward. An empty history is ``404 no_plan``.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "GuidePlanRevertRequest".
 */
export interface GuidePlanRevertRequest {
  session_id: SessionId4;
  plan_id: PlanId5;
  plan_revision: PlanRevision7;
}
/**
 * What the user just said during a guide, with the frame captured right now.
 *
 * ``prev_utterance``/``prev_reply`` are the previous exchange (both or neither): the server keeps no
 * conversation. ``follow_checks`` is the client's last accepted follow checklist, prompt context only; every
 * id must be a step of the current plan, or ``422 unknown_step`` (checked by the route, like confirm).
 * ``replan_allowed`` is ``False`` once the client's run has spent its replans: the tool is then offered without
 * ``replan`` and a replan in the answer is a schema violation.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "GuideTalkRequest".
 */
export interface GuideTalkRequest {
  session_id: SessionId5;
  consent_ai: ConsentAi3;
  scene: SceneInput;
  plan_id: PlanId6;
  plan_revision: PlanRevision8;
  current_step: CurrentStep2;
  anchors: Anchors2;
  utterance: Utterance;
  prev_utterance?: PrevUtterance;
  prev_reply?: PrevReply;
  follow_checks?: FollowChecks1;
  replan_allowed?: ReplanAllowed;
  intent_seq: IntentSeq4;
  fence: GuideFence;
}
/**
 * The talk answer plus the same acceptance envelope as confirm (there is no trigger).
 *
 * ``plan_revision`` is the revision AFTER this call: bumped when ``step_say`` (``guide.restate``) or
 * ``replan`` (``guide.replan``) was applied to the stored plan, unchanged for every other action, because
 * the step position (``step_mark``/``go_to``) and the tracker (``target``) belong to the client.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "GuideTalkResponse".
 */
export interface GuideTalkResponse {
  user_says_done: UserSaysDone;
  reply: Reply;
  spoken: Spoken;
  step_say?: StepSay;
  target?: Target1;
  step_mark?: StepMark;
  go_to?: GoTo;
  replan?: GuideReplan | null;
  intent_id: IntentId2;
  frame_id: FrameId4;
  intent_seq: IntentSeq5;
  task_revision: TaskRevision3;
  plan_revision: PlanRevision9;
  fence_echo: GuideFence;
  anchors_echo: AnchorsEcho2;
  provider: Provider4;
  model?: Model4;
  usage?: Usage | null;
  latency_ms: LatencyMs2;
}
/**
 * The guide's answer to the user's words: a short Korean ``reply`` and at most ONE action.
 *
 * * ``step_say`` — the current step's sentence, rewritten (the plan ``say`` bound).
 * * ``target`` — a more specific detector noun; the client restarts tracking with it (no paid call).
 * * ``step_mark`` — the user says the current step is done (``done``) or wants it skipped (``skipped``).
 * * ``go_to`` — an EARLIER step to return to; the route refuses the current or a later one.
 * * ``replan`` — replacement steps, the same shape as a confirm replan.
 *
 * ``spoken`` is the one short sentence the voice reads (the screen shows ``reply``); a missing or blank one is the
 * reply's first sentence.
 *
 * ``user_says_done`` is a classification of the user's WORDS alone (not of the image): do they say the current
 * step is already done, or that they want to move on? When it is true the user's word is the answer (design
 * guide-talk §2, "사용자 확인"): the action becomes ``step_mark`` — ``done``, or ``skipped`` if the model said
 * so — and every other action is dropped, so a model that judged the image otherwise cannot refuse it
 * (glass e2e 2026-10-02: "이미 했어" was argued with 2/2 when ``step_mark`` was the model's own choice).
 *
 * Actions combine: at most one of ``TALK_STATE_ACTIONS`` (the first in precedence), plus ``target`` and
 * ``step_say`` (a ``replan`` drops ``step_say``). Whatever had to go is named in ``dropped_actions`` (the route
 * logs it); ``actions`` lists what was kept. Over-long ``reply``/``step_say`` are cut
 * (``_clip``) as plan text is. A blank reply or a target outside ``TalkTarget`` is a schema violation.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "GuideTalkToolOutput".
 */
export interface GuideTalkToolOutput {
  user_says_done: UserSaysDone1;
  reply: Reply1;
  spoken: Spoken1;
  step_say?: StepSay1;
  target?: Target2;
  step_mark?: StepMark1;
  go_to?: GoTo1;
  replan?: GuideReplan | null;
}
/**
 * Readiness probe.
 *
 * ``access_code_required`` reports the deployment's session authentication mode, never the
 * calling browser's identity: it is ``False`` only for an explicit local-only deployment
 * (``AISW_LOCAL_ONLY=1``, loopback peer + loopback authority) or an explicit open-access deployment
 * (``AISW_OPEN_ACCESS=1``), where the UI must not ask for a code at all. The client is not allowed
 * to infer this from its own hostname.
 *
 * ``access_mode`` names that mode (``code`` / ``local`` / ``open``). It is ``None`` only for the
 * refused configuration (both ``AISW_OPEN_ACCESS`` and ``AISW_LOCAL_ONLY`` set), where ``ready`` is
 * false, ``access_code_required`` is true and every session route answers ``503``.
 *
 * ``tracker_ready`` reports the standalone object tracker independently of the paid provider:
 * either readiness may be true without the other, and neither is derived from the other.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "HealthResponse".
 */
export interface HealthResponse {
  ready: Ready1;
  model: Model5;
  provider?: Provider5;
  access_code_required: AccessCodeRequired1;
  access_mode?: AccessMode1;
  tracker_ready?: TrackerReady1;
}
/**
 * One local import: the current app session and the raw file bytes, base64-encoded.
 *
 * ``content_base64`` may be a data URL (``data:...;base64,``), which the client's ``FileReader`` produces.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "ManualImportRequest".
 */
export interface ManualImportRequest {
  session_id: SessionId6;
  filename: Filename;
  content_base64: ContentBase64;
}
/**
 * Session request.
 *
 * ``access_code`` is optional on the wire. An explicit local-only or open-access deployment issues
 * sessions without a code (any supplied value is ignored), so the browser omits the field entirely;
 * every other deployment requires the configured ``DEMO_ACCESS_CODE`` and rejects a missing, empty,
 * or wrong value.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "SessionRequest".
 */
export interface SessionRequest {
  access_code?: AccessCode;
}
/**
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "SessionResponse".
 */
export interface SessionResponse {
  session_id: SessionId7;
}
/**
 * Start or stop a tracking run.
 *
 * ``action="start"`` requires ``start_seq`` (strictly increasing per application session) and a
 * non-blank ``target``. ``action="stop"`` names the ``run_id`` it intends to stop and carries neither.
 * Image/frame/seed fields are never accepted here.
 *
 * ``target`` stays optional on the wire only so a stop can omit it: the application refuses a start
 * with an absent or blank target (``422 target_required``) rather than substituting a default class
 * name. The client supplies the text — the object the startup analysis selected, or the user's own
 * words for a box they drew (see ``MANUAL_TARGET``).
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "TrackControlRequest".
 */
export interface TrackControlRequest {
  session_id: SessionId8;
  run_id: RunId2;
  action: Action1;
  start_seq?: StartSeq;
  target?: Target3;
}
/**
 * Control acknowledgement: no frame, no box, no fabricated result.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "TrackControlResponse".
 */
export interface TrackControlResponse {
  run_id: RunId3;
  active: Active;
  target: Target4;
}
/**
 * The same frame as ``TrackFrameRequest``, uploaded as raw JPEG bytes instead of JSON + base64.
 *
 * ``POST /api/track/frame`` with ``Content-Type: image/jpeg``: the body is the JPEG itself and these
 * fields are the query string. It carries no ``seed_box`` -- a run's seeded first frame is always sent as
 * JSON. A server that predates this form answers ``415 json_required``, which a client takes as "send JSON".
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "TrackFrameQuery".
 */
export interface TrackFrameQuery {
  session_id: SessionId9;
  run_id: RunId4;
  frame_id: FrameId5;
  frame_seq: FrameSeq;
}
/**
 * One captured frame of the active run.
 *
 * ``seed_box`` is the user's manual selection on that captured image (normalised 0..1 in the uploaded
 * image) and is accepted on the **first frame of the run only**; the frame still receives an ordinary
 * frame response.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "TrackFrameRequest".
 */
export interface TrackFrameRequest {
  session_id: SessionId10;
  run_id: RunId5;
  frame_id: FrameId6;
  frame_seq: FrameSeq1;
  image_base64: ImageBase641;
  seed_box?: Box | null;
}
/**
 * Track state for exactly the frame named by ``frame_id``/``frame_seq``.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "TrackFrameResponse".
 */
export interface TrackFrameResponse {
  run_id: RunId6;
  target: Target5;
  track_id: TrackId2;
  generation: Generation2;
  state: State1;
  transition: Transition;
  box?: Box | null;
  source?: Source;
  frame_id: FrameId7;
  frame_seq: FrameSeq2;
  version: Version3;
  confidence?: Confidence;
  ingest_age_ms: IngestAgeMs;
}
/**
 * One startup analysis: choose what the tracker should track, from the frame captured right now.
 *
 * ``scene.frame_id`` is the frame this analysis is about, and the client reuses that exact frame as the
 * tracking run's first frame. ``user_goal``/``context`` are the task text the user is working under, so
 * the choice is about *their* object rather than the most salient thing in the image.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "TrackSelectRequest".
 */
export interface TrackSelectRequest {
  session_id: SessionId11;
  consent_ai: ConsentAi4;
  scene: SceneInput;
  user_goal?: UserGoal4;
  context?: Context2;
}
/**
 * One startup analysis answer: a candidate target (or an honest non-selection) and what it cost.
 *
 * A non-selection is a successful answer, not an error: ``status`` says which one it is and ``rationale``
 * says why. ``target`` is a short **English** noun for the tracker's detector, never a location — this
 * analysis does not claim to have located anything, and the tracker's own acquisition accuracy is
 * unchanged by it. A client may show that noun as-is (it is the label the detector is working from);
 * ``rationale`` is the human-facing Korean sentence to show a person.
 * ``usage``/``provider``/``model`` are the one call's own accounting, reported for every outcome
 * including a non-selection, so a consumed call is never silently uncounted.
 *
 * This interface was referenced by `GuidanceApiContracts`'s JSON-Schema
 * via the `definition` "TrackSelectResponse".
 */
export interface TrackSelectResponse {
  selection_id: SelectionId;
  frame_id: FrameId8;
  status: TargetSelectionStatus;
  target?: Target6;
  rationale?: Rationale2;
  provider: Provider6;
  model?: Model6;
  usage?: Usage | null;
}
