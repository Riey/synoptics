import Ajv2020, { type ErrorObject, type ValidateFunction } from 'ajv/dist/2020.js';
import addFormats from 'ajv-formats';
import schema from './visual.schema.json' with { type: 'json' };
import type {
  ArrowCommand,
  FocusCommand,
  GestureCommand,
  GuidanceAdvice,
  GuidelineStep,
  HintCommand,
  PathCommand,
  Point,
  VisualContracts,
} from './visual.generated.js';

export type VisualCommand = FocusCommand | ArrowCommand | PathCommand | GestureCommand | HintCommand;
export type { GuidelineStep, GuidanceAdvice, Point };

const ajv = new Ajv2020({
  allErrors: true,
  strict: false,
  strictNumbers: true, // Strictly reject NaN and ±Infinity
});
// ajv-formats typing uses Ajv class rather than Ajv2020 instance
const plugin = addFormats as unknown as (instance: Ajv2020) => void;
plugin(ajv);

const defsMap = schema.$defs as Record<string, unknown>;
const compiledContracts: Record<string, ValidateFunction<unknown>> = {};

function schemaBound(definition: string, property: string, bound: 'maxLength' | 'maxItems'): number | undefined {
  const def = defsMap[definition];
  const properties = def !== null && typeof def === 'object' ? Reflect.get(def, 'properties') : undefined;
  const field = properties !== null && typeof properties === 'object' ? Reflect.get(properties, property) : undefined;
  const value = field !== null && typeof field === 'object' ? Reflect.get(field, bound) : undefined;
  return typeof value === 'number' ? value : undefined;
}

/** Keep a renderer reason inside the canonical wire bound for `FramedRenderReport.reason`. */
export function withinReportReasonLimit(reason: string): string {
  const limit = schemaBound('FramedRenderReport', 'reason', 'maxLength');
  if (limit === undefined || reason.length <= limit) return reason;
  return `${reason.slice(0, Math.max(0, limit - 1))}…`;
}

/** Per-advice command ceiling, read from the canonical schema so the two can never drift. */
const commandLimit = schemaBound('GuidanceAdvice', 'commands', 'maxItems');

function contractValidator(name: keyof VisualContracts): ValidateFunction<unknown> {
  let validator = compiledContracts[name];
  if (!validator) {
    validator = ajv.compile({ $ref: `#/$defs/${name}`, $defs: defsMap }) as ValidateFunction;
    compiledContracts[name] = validator;
  }
  return validator;
}

export function validateContract<K extends keyof VisualContracts>(
  name: K,
  value: unknown
): { valid: boolean; errors: string[] } {
  if (!(name in defsMap)) {
    return { valid: false, errors: [`unknown contract: ${String(name)}`] };
  }
  const validator = contractValidator(name);
  const valid = validator(value) as boolean;
  return { valid, errors: valid ? [] : formatErrors(validator.errors) };
}

const COMMAND_MODELS = {
  focus: 'FocusCommand',
  arrow: 'ArrowCommand',
  path: 'PathCommand',
  gesture: 'GestureCommand',
  hint: 'HintCommand',
} as const satisfies Record<string, keyof VisualContracts>;

type CommandKind = keyof typeof COMMAND_MODELS;

const kindValidators: Record<CommandKind, ValidateFunction> = Object.fromEntries(
  Object.entries(COMMAND_MODELS).map(([kind, model]) => [
    kind,
    ajv.compile({ $ref: `#/$defs/${model}`, $defs: defsMap }) as ValidateFunction,
  ])
) as Record<CommandKind, ValidateFunction>;

const commandUnion = {
  anyOf: Object.values(COMMAND_MODELS).map((model) => ({ $ref: `#/$defs/${model}` })),
};

export const validateCommandsSchema: ValidateFunction<VisualCommand[]> = ajv.compile({
  type: 'array',
  ...(commandLimit === undefined ? {} : { maxItems: commandLimit }),
  items: commandUnion,
  $defs: defsMap,
}) as ValidateFunction<VisualCommand[]>;

export interface AdviceValidationContext {
  guidanceMode?: 'text' | 'visual';
  /** The frame the caller is displaying; its id is the implicit grounding of `observed_scene`. */
  sceneId?: string;
  /** Ids of the reference assets the caller actually supplied for this guidance. */
  referenceIds?: string[];
}

export interface ValidationResult {
  valid: boolean;
  errors: string[];
}

function formatErrors(errors: ErrorObject[] | null | undefined, prefix = ''): string[] {
  if (!errors || errors.length === 0) return [`${prefix}value does not match the contract`.trim()];
  return errors.map((error) => {
    const where = `${prefix}${error.instancePath || ''}`.replace(/^\/+/, '');
    return where ? `${where}: ${error.message ?? 'invalid'}` : (error.message ?? 'invalid');
  });
}

function duplicateIdErrors(commands: unknown): string[] {
  if (!Array.isArray(commands)) return [];
  const seen = new Set<string>();
  const errors: string[] = [];
  for (const command of commands) {
    const id = command !== null && typeof command === 'object' ? Reflect.get(command, 'id') : undefined;
    if (typeof id !== 'string') continue;
    if (seen.has(id)) errors.push(`duplicate command id: ${id}`);
    seen.add(id);
  }
  return errors;
}

function kindValidatorFor(kind: unknown): ValidateFunction | undefined {
  // Own-property lookup only: `kind` is untrusted input and must never resolve through the prototype.
  if (typeof kind !== 'string' || !Object.hasOwn(kindValidators, kind)) return undefined;
  return kindValidators[kind as CommandKind];
}

/**
 * Relational rules shared by the low-level and the framed render paths, so the two can never
 * disagree about the same command bundle. JSON Schema cannot express any of these.
 */
function commandRelationalErrors(commands: unknown): string[] {
  const errors = duplicateIdErrors(commands);
  if (!Array.isArray(commands)) return errors;
  let gestures = 0;
  for (const command of commands) {
    if (command === null || typeof command !== 'object') continue;
    const kind = Reflect.get(command, 'kind');
    if (kind === 'gesture') gestures += 1;
    if (kind !== 'focus') continue;
    const box = Reflect.get(command, 'box');
    if (box === null || typeof box !== 'object') continue;
    const x = Reflect.get(box, 'x');
    const y = Reflect.get(box, 'y');
    const width = Reflect.get(box, 'width');
    const height = Reflect.get(box, 'height');
    if (typeof x === 'number' && typeof width === 'number' && x + width > 1) {
      errors.push(`focus box extends beyond the frame: ${String(Reflect.get(command, 'id'))}`);
    } else if (typeof y === 'number' && typeof height === 'number' && y + height > 1) {
      errors.push(`focus box extends beyond the frame: ${String(Reflect.get(command, 'id'))}`);
    }
  }
  if (gestures > 1) errors.push('at most one gesture command allowed');
  return errors;
}

/**
 * Attribute a rejection to the command kind that actually failed.
 *
 * A plain `anyOf` validation surfaces the first branch's errors, so every failure would be reported
 * against the Focus branch. Here each command is validated against its own kind, and the array-level
 * count rule is reported explicitly.
 */
function describeCommandFailures(commands: unknown): string[] {
  if (!Array.isArray(commands)) {
    return ['commands must be an array'];
  }
  const errors: string[] = [];
  if (commandLimit !== undefined && commands.length > commandLimit) {
    errors.push(`at most ${commandLimit} commands allowed (received ${commands.length})`);
  }
  commands.forEach((raw, index) => {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
      errors.push(`command[${index}] must be an object`);
      return;
    }
    const kind = Reflect.get(raw, 'kind');
    const validator = kindValidatorFor(kind);
    if (!validator) {
      errors.push(`command[${index}] has unknown kind ${JSON.stringify(kind ?? null)}`);
      return;
    }
    if (!validator(raw)) {
      const [reason] = formatErrors(validator.errors, `command[${index}] `);
      errors.push(`${kind}: ${reason}`);
    }
  });
  return errors.length > 0 ? errors : formatErrors(validateCommandsSchema.errors);
}

export function validateCommands(commands: unknown): ValidationResult {
  const errors = validateCommandsSchema(commands) ? [] : describeCommandFailures(commands);
  errors.push(...commandRelationalErrors(commands));
  return { valid: errors.length === 0, errors };
}

/** Kind-attributed diagnostics for an advice payload, used to keep framed rejections specific. */
function adviceCommandErrors(advice: unknown): string[] {
  if (advice === null || typeof advice !== 'object') return [];
  const commands = Reflect.get(advice, 'commands');
  return commands === undefined ? [] : validateCommands(commands).errors;
}

function adviceFromEnvelope(envelope: unknown): unknown {
  return envelope !== null && typeof envelope === 'object' ? Reflect.get(envelope, 'advice') : undefined;
}

function collectAdviceErrors(advice: GuidanceAdvice, context: AdviceValidationContext): string[] {
  const errors: string[] = [];
  if (advice.evidence_kind === 'uncertain_view' && !advice.needs_clarification) {
    errors.push('uncertain_view evidence requires needs_clarification');
  }
  const prompt = advice.clarification_prompt;
  const hasPrompt = typeof prompt === 'string' && prompt.trim().length > 0;
  if (advice.needs_clarification) {
    if (advice.ready_to_advance) errors.push('clarification and ready_to_advance cannot both be true');
    if (!hasPrompt) errors.push('clarification_prompt is required when clarification is needed');
    if (advice.commands.some((command) => command.kind !== 'hint')) {
      errors.push('only hint commands are allowed when clarification is needed');
    }
  } else if (prompt !== undefined && prompt !== null) {
    errors.push('clarification_prompt must be absent when clarification is not needed');
  }
  const referenceIds = context.referenceIds ?? [];
  if (context.sceneId === undefined) {
    if (advice.evidence_kind === 'reference' && advice.references.length === 0) {
      errors.push('reference evidence requires at least one cited reference');
    }
  } else {
    const supplied = new Set([context.sceneId, ...referenceIds]);
    for (const cited of advice.references) {
      if (!supplied.has(cited)) errors.push(`cited reference was not supplied: ${cited}`);
    }
    if (advice.evidence_kind === 'reference' && !advice.references.some((cited) => referenceIds.includes(cited))) {
      errors.push('reference evidence must cite at least one supplied reference image');
    }
  }
  if (new Set(advice.references).size !== advice.references.length) {
    errors.push('duplicate references');
  }
  errors.push(...validateCommands(advice.commands).errors);
  return errors;
}

/** Structural validation of the framed envelope, with kind-attributed diagnostics when a command is at fault. */
export function validateGuidanceEnvelope(envelope: unknown): ValidationResult {
  const structural = validateContract('VisualGuidance', envelope);
  if (structural.valid) return { valid: true, errors: [] };
  const commandErrors = adviceCommandErrors(adviceFromEnvelope(envelope));
  return { valid: false, errors: commandErrors.length > 0 ? commandErrors : structural.errors };
}

/** Relational rules that need the request context, checked client-side before rendering. */
export function validateAdviceRelational(
  advice: unknown,
  context: AdviceValidationContext = {}
): ValidationResult {
  const structural = validateContract('GuidanceAdvice', advice);
  if (!structural.valid) {
    const commandErrors = adviceCommandErrors(advice);
    return { valid: false, errors: commandErrors.length > 0 ? commandErrors : structural.errors };
  }
  const errors = collectAdviceErrors(advice as GuidanceAdvice, context);
  if (context.guidanceMode === 'text' && (advice as GuidanceAdvice).commands.length > 0) {
    errors.push('text guidance cannot carry visual commands');
  }
  return { valid: errors.length === 0, errors };
}

