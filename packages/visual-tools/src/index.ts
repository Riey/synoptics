import {
  renderCommands,
  clearCommands,
  renderGuidance,
  type RenderContext,
  type RenderReport,
  type FramedRenderContext,
  type RenderCommand,
  type FocusCommand,
  type ArrowCommand,
  type PathCommand,
  type GestureCommand,
  type HintCommand,
} from './renderer.js';
import {
  validateContract,
  validateCommands,
  validateAdviceRelational,
  validateCommandsSchema,
  validateGuidanceEnvelope,
  withinReportReasonLimit,
  type AdviceValidationContext,
  type ValidationResult,
  type VisualCommand,
} from './validator.js';

export {
  renderCommands,
  clearCommands,
  renderGuidance,
  validateContract,
  validateCommands,
  validateAdviceRelational,
  validateCommandsSchema,
  validateGuidanceEnvelope,
  withinReportReasonLimit,
};

export type {
  RenderContext,
  RenderReport,
  FramedRenderContext,
  AdviceValidationContext,
  ValidationResult,
  VisualCommand,
  RenderCommand,
  FocusCommand,
  ArrowCommand,
  PathCommand,
  GestureCommand,
  HintCommand,
};

// The domain-neutral visual contract only. The application HTTP/session contract is generated
// separately for the web app and is deliberately not part of this package's public surface.
export * from './visual.generated.js';
