/* eslint-disable */
/**
 * Automatically generated from packages/visual-tools/src/visual.schema.json. DO NOT EDIT MANUALLY.
 */

export type Id = string;
export type Kind = "arrow";
export type X = number;
export type Y = number;
export type Label = string;
export type X1 = number;
export type Y1 = number;
export type Width = number;
export type Height = number;
export type Id1 = string;
export type Kind1 = "focus";
export type Label1 = string;
export type GuidanceId = string | null;
export type FrameId = string;
export type Status = "rendered" | "rejected" | "stale";
export type CommandIds = string[];
export type Reason = string | null;
export type Id2 = string;
export type Kind2 = "gesture";
export type Points = Point[];
export type DurationMs = number;
export type Label2 = string;
export type Explanation = string;
export type Observation = string;
export type StepId = string;
export type Title = string;
export type Detail = string;
export type Steps = GuidelineStep[];
export type EvidenceKind = "reference" | "observed_scene" | "general_inference" | "uncertain_view";
export type References = string[];
export type Id3 = string;
export type Kind3 = "path";
export type Points1 = Point[];
export type Label3 = string;
export type Id4 = string;
export type Kind4 = "hint";
export type Text = string;
export type Reference = "none" | "line" | "circle" | "hatching";
export type Commands = (FocusCommand | ArrowCommand | PathCommand | GestureCommand | HintCommand)[];
export type NeedsClarification = boolean;
export type ClarificationPrompt = string | null;
export type ReadyToAdvance = boolean;
export type Warnings = string[];
export type GuidanceId1 = string;
export type FrameId1 = string;

export interface VisualContracts {
  ArrowCommand?: ArrowCommand;
  Box?: Box;
  FocusCommand?: FocusCommand;
  FramedRenderReport?: FramedRenderReport;
  GestureCommand?: GestureCommand;
  GuidanceAdvice?: GuidanceAdvice;
  GuidelineStep?: GuidelineStep;
  HintCommand?: HintCommand;
  PathCommand?: PathCommand;
  Point?: Point;
  VisualGuidance?: VisualGuidance;
}
/**
 * This interface was referenced by `VisualContracts`'s JSON-Schema
 * via the `definition` "ArrowCommand".
 */
export interface ArrowCommand {
  id: Id;
  kind: Kind;
  from: Point;
  to: Point;
  label: Label;
}
/**
 * This interface was referenced by `VisualContracts`'s JSON-Schema
 * via the `definition` "Point".
 */
export interface Point {
  x: X;
  y: Y;
}
/**
 * This interface was referenced by `VisualContracts`'s JSON-Schema
 * via the `definition` "Box".
 */
export interface Box {
  x: X1;
  y: Y1;
  width: Width;
  height: Height;
}
/**
 * This interface was referenced by `VisualContracts`'s JSON-Schema
 * via the `definition` "FocusCommand".
 */
export interface FocusCommand {
  id: Id1;
  kind: Kind1;
  box: Box;
  label: Label1;
}
/**
 * What the renderer did with a VisualGuidance payload.
 *
 * This is a *client-reported* observation: it records what the local renderer actually did on the
 * caller's machine. It is untrusted input to the server — never a completion proof, a security
 * assertion, or evidence that a human verified anything.
 *
 * ``frame_id`` is the frame the guidance **targeted** (the report's identity), so a report about an
 * older frame stays identifiable. In a live-camera flow the report a client forwards usually names the
 * *previous* capture, not the one being analysed now: it is then history about superseded guidance,
 * and it is never evidence about the current scene. Only when the payload is too malformed to identify
 * at all does the renderer fall back to the frame it was displaying. ``guidance_id`` is null in that
 * same case.
 *
 * This interface was referenced by `VisualContracts`'s JSON-Schema
 * via the `definition` "FramedRenderReport".
 */
export interface FramedRenderReport {
  guidance_id: GuidanceId;
  frame_id: FrameId;
  status: Status;
  command_ids: CommandIds;
  reason?: Reason;
}
/**
 * This interface was referenced by `VisualContracts`'s JSON-Schema
 * via the `definition` "GestureCommand".
 */
export interface GestureCommand {
  id: Id2;
  kind: Kind2;
  points: Points;
  duration_ms: DurationMs;
  label: Label2;
}
/**
 * The single advice envelope produced by an agent/provider and returned to the browser.
 *
 * This interface was referenced by `VisualContracts`'s JSON-Schema
 * via the `definition` "GuidanceAdvice".
 */
export interface GuidanceAdvice {
  explanation: Explanation;
  observation: Observation;
  steps: Steps;
  evidence_kind: EvidenceKind;
  references: References;
  commands: Commands;
  needs_clarification: NeedsClarification;
  clarification_prompt?: ClarificationPrompt;
  ready_to_advance: ReadyToAdvance;
  warnings: Warnings;
}
/**
 * An optional, agent-authored step. Steps are content, never a mandatory state machine.
 *
 * This interface was referenced by `VisualContracts`'s JSON-Schema
 * via the `definition` "GuidelineStep".
 */
export interface GuidelineStep {
  step_id: StepId;
  title: Title;
  detail: Detail;
}
/**
 * This interface was referenced by `VisualContracts`'s JSON-Schema
 * via the `definition` "PathCommand".
 */
export interface PathCommand {
  id: Id3;
  kind: Kind3;
  points: Points1;
  label: Label3;
}
/**
 * This interface was referenced by `VisualContracts`'s JSON-Schema
 * via the `definition` "HintCommand".
 */
export interface HintCommand {
  id: Id4;
  kind: Kind4;
  text: Text;
  reference: Reference;
}
/**
 * The framed envelope consumed by every renderer entrypoint: which guidance, which frame, what advice.
 *
 * This interface was referenced by `VisualContracts`'s JSON-Schema
 * via the `definition` "VisualGuidance".
 */
export interface VisualGuidance {
  guidance_id: GuidanceId1;
  frame_id: FrameId1;
  advice: GuidanceAdvice;
}
