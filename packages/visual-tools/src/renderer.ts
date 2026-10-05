import type {
  ArrowCommand as WireArrow,
  FocusCommand as WireFocus,
  FramedRenderReport,
  GestureCommand as WireGesture,
  HintCommand as WireHint,
  PathCommand as WirePath,
  Point,
  VisualGuidance,
} from './visual.generated.js';
import {
  validateAdviceRelational,
  validateCommands,
  validateGuidanceEnvelope,
  withinReportReasonLimit,
} from './validator.js';

export interface RenderContext {
  svg: SVGSVGElement;
  hintRoot: HTMLElement;
  project: (point: Point) => Point;
  reducedMotion: boolean;
}

export type RenderCommand = WireFocus | WireArrow | WirePath | WireGesture | WireHint;
export type FocusCommand = WireFocus;
export type ArrowCommand = WireArrow;
export type PathCommand = WirePath;
export type GestureCommand = WireGesture;
export type HintCommand = WireHint;
export interface RenderReport {
  status: 'rendered' | 'rejected' | 'stale';
  command_ids: string[];
  reason: string | null;
}

// Track per-context generation & active animation frames / timeouts
const contextGenerations = new WeakMap<SVGSVGElement, number>();
const activeGestureCleanups = new WeakMap<SVGSVGElement, () => void>();

export function clearCommands(context: RenderContext): void {
  const currentGen = (contextGenerations.get(context.svg) ?? 0) + 1;
  contextGenerations.set(context.svg, currentGen);

  // Cancel any active gesture animation
  const cleanup = activeGestureCleanups.get(context.svg);
  if (cleanup) {
    cleanup();
    activeGestureCleanups.delete(context.svg);
  }

  while (context.svg.firstChild) {
    context.svg.removeChild(context.svg.firstChild);
  }
  while (context.hintRoot.firstChild) {
    context.hintRoot.removeChild(context.hintRoot.firstChild);
  }
}

export function renderCommands(
  commands: unknown,
  context: RenderContext
): Promise<RenderReport> {
  // 1. REJECT DISCONNECTED DOM ELEMENTS
  if (!context || !context.svg || !context.hintRoot || !context.svg.isConnected || !context.hintRoot.isConnected) {
    return Promise.resolve({
      status: 'rejected',
      command_ids: [],
      reason: 'SVG or hintRoot is disconnected from DOM',
    });
  }

  // 2. ATOMIC VALIDATION BEFORE ANY ID EXTRACTION OR DOM MUTATION
  const validation = validateCommands(commands);
  if (!validation.valid) {
    // Only extract structurally safe string IDs (max 64 chars) if commands is an array
    const safeIds: string[] = [];
    if (Array.isArray(commands)) {
      for (const c of commands) {
        if (c && typeof c === 'object' && typeof c.id === 'string' && c.id.length >= 1 && c.id.length <= 64) {
          safeIds.push(c.id);
        }
      }
    }
    return Promise.resolve({
      status: 'rejected',
      command_ids: safeIds,
      reason: validation.errors[0] || 'Commands schema or relational validation failed',
    });
  }

  const validCommands = commands as readonly RenderCommand[];
  const commandIds = validCommands.map((c) => c.id);

  // 3. PROJECT ALL POINTS SAFELY BEFORE ANY DOM MUTATION
  // Any throw or non-finite coordinate rejects entire bundle atomically
  const { project, reducedMotion } = context;

  type ProjectedCommand =
    | { kind: 'focus'; cmd: FocusCommand; corners: [Point, Point, Point, Point] }
    | { kind: 'arrow'; cmd: ArrowCommand; from: Point; to: Point }
    | { kind: 'path'; cmd: PathCommand; points: Point[] }
    | { kind: 'gesture'; cmd: GestureCommand; points: Point[] }
    | { kind: 'hint'; cmd: HintCommand };

  const projectedList: ProjectedCommand[] = [];

  const safeProject = (pt: Point): Point | null => {
    try {
      const res = project(pt);
      if (!res || typeof res.x !== 'number' || typeof res.y !== 'number' || !Number.isFinite(res.x) || !Number.isFinite(res.y)) {
        return null;
      }
      return res;
    } catch {
      return null;
    }
  };

  for (let i = 0; i < validCommands.length; i++) {
    const cmd = validCommands[i];
    if (cmd.kind === 'focus') {
      const p1 = safeProject({ x: cmd.box.x, y: cmd.box.y });
      const p2 = safeProject({ x: cmd.box.x + cmd.box.width, y: cmd.box.y });
      const p3 = safeProject({ x: cmd.box.x + cmd.box.width, y: cmd.box.y + cmd.box.height });
      const p4 = safeProject({ x: cmd.box.x, y: cmd.box.y + cmd.box.height });
      if (!p1 || !p2 || !p3 || !p4) {
        return Promise.resolve({
          status: 'rejected',
          command_ids: commandIds,
          reason: 'Failed to project focus box coordinates',
        });
      }
      projectedList.push({ kind: 'focus', cmd, corners: [p1, p2, p3, p4] });
    } else if (cmd.kind === 'arrow') {
      const from = safeProject(cmd.from);
      const to = safeProject(cmd.to);
      if (!from || !to) {
        return Promise.resolve({
          status: 'rejected',
          command_ids: commandIds,
          reason: 'Failed to project arrow coordinates',
        });
      }
      projectedList.push({ kind: 'arrow', cmd, from, to });
    } else if (cmd.kind === 'path') {
      const pts: Point[] = [];
      for (const p of cmd.points) {
        const proj = safeProject(p);
        if (!proj) {
          return Promise.resolve({
            status: 'rejected',
            command_ids: commandIds,
            reason: 'Failed to project path point',
          });
        }
        pts.push(proj);
      }
      projectedList.push({ kind: 'path', cmd, points: pts });
    } else if (cmd.kind === 'gesture') {
      const pts: Point[] = [];
      for (const p of cmd.points) {
        const proj = safeProject(p);
        if (!proj) {
          return Promise.resolve({
            status: 'rejected',
            command_ids: commandIds,
            reason: 'Failed to project gesture point',
          });
        }
        pts.push(proj);
      }
      projectedList.push({ kind: 'gesture', cmd, points: pts });
    } else if (cmd.kind === 'hint') {
      projectedList.push({ kind: 'hint', cmd });
    }
  }

  // Increment per-context generation & cancel old gesture
  const currentGen = (contextGenerations.get(context.svg) ?? 0) + 1;
  contextGenerations.set(context.svg, currentGen);

  const prevCleanup = activeGestureCleanups.get(context.svg);
  if (prevCleanup) {
    prevCleanup();
    activeGestureCleanups.delete(context.svg);
  }

  // Clear previous DOM elements immediately
  while (context.svg.firstChild) {
    context.svg.removeChild(context.svg.firstChild);
  }
  while (context.hintRoot.firstChild) {
    context.hintRoot.removeChild(context.hintRoot.firstChild);
  }

  const { promise, resolve } = Promise.withResolvers<RenderReport>();

  // First rAF: DOM attachment
  requestAnimationFrame(() => {
    if (contextGenerations.get(context.svg) !== currentGen || !context.svg.isConnected || !context.hintRoot.isConnected) {
      resolve({
        status: 'stale',
        command_ids: commandIds,
        reason: 'superseded by a newer render or clear before any DOM mutation',
      });
      return;
    }

    const { svg, hintRoot } = context;

    // SVG Defs & Marker
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
    const markerId = `arrowhead-${currentGen}`;
    marker.setAttribute('id', markerId);
    marker.setAttribute('markerWidth', '10');
    marker.setAttribute('markerHeight', '8');
    marker.setAttribute('refX', '8');
    marker.setAttribute('refY', '4');
    marker.setAttribute('orient', 'auto');
    const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    polygon.setAttribute('points', '0 0, 10 4, 0 8');
    polygon.setAttribute('fill', '#ffffff');
    polygon.setAttribute('stroke', '#000000');
    polygon.setAttribute('stroke-width', '1');
    marker.appendChild(polygon);
    defs.appendChild(marker);
    svg.appendChild(defs);

    for (let i = 0; i < projectedList.length; i++) {
      const item = projectedList[i];
      const num = i + 1;

      if (item.kind === 'focus') {
        const [p1, p2, p3, p4] = item.corners;
        const ptsStr = `${p1.x},${p1.y} ${p2.x},${p2.y} ${p3.x},${p3.y} ${p4.x},${p4.y}`;

        // Black outer stroke
        const outer = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        outer.setAttribute('points', ptsStr);
        outer.setAttribute('fill', 'none');
        outer.setAttribute('stroke', '#000000');
        outer.setAttribute('stroke-width', '5');
        svg.appendChild(outer);

        // White inner stroke
        const inner = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        inner.setAttribute('points', ptsStr);
        inner.setAttribute('fill', 'rgba(255, 255, 255, 0.15)');
        inner.setAttribute('stroke', '#ffffff');
        inner.setAttribute('stroke-width', '2.5');
        svg.appendChild(inner);

        // Number badge on video (NUMBER ONLY)
        const badgeBg = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        badgeBg.setAttribute('cx', `${p1.x}`);
        badgeBg.setAttribute('cy', `${p1.y}`);
        badgeBg.setAttribute('r', '10');
        badgeBg.setAttribute('fill', '#000000');
        svg.appendChild(badgeBg);

        const badge = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        badge.setAttribute('x', `${p1.x}`);
        badge.setAttribute('y', `${p1.y + 4}`);
        badge.setAttribute('text-anchor', 'middle');
        badge.setAttribute('fill', '#ffffff');
        badge.setAttribute('font-size', '12px');
        badge.setAttribute('font-weight', 'bold');
        badge.textContent = `${num}`;
        svg.appendChild(badge);

        // Description card in hintRoot
        appendHintDescriptionCard(hintRoot, num, '초점 영역', item.cmd.label);
      } else if (item.kind === 'arrow') {
        const { from, to } = item;

        // Black outer line
        const outer = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        outer.setAttribute('x1', `${from.x}`);
        outer.setAttribute('y1', `${from.y}`);
        outer.setAttribute('x2', `${to.x}`);
        outer.setAttribute('y2', `${to.y}`);
        outer.setAttribute('stroke', '#000000');
        outer.setAttribute('stroke-width', '6');
        svg.appendChild(outer);

        // White inner line
        const inner = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        inner.setAttribute('x1', `${from.x}`);
        inner.setAttribute('y1', `${from.y}`);
        inner.setAttribute('x2', `${to.x}`);
        inner.setAttribute('y2', `${to.y}`);
        inner.setAttribute('stroke', '#ffffff');
        inner.setAttribute('stroke-width', '3');
        inner.setAttribute('marker-end', `url(#${markerId})`);
        svg.appendChild(inner);

        // Number badge on video (NUMBER ONLY)
        const midX = (from.x + to.x) / 2;
        const midY = (from.y + to.y) / 2;
        const badgeBg = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        badgeBg.setAttribute('cx', `${midX}`);
        badgeBg.setAttribute('cy', `${midY}`);
        badgeBg.setAttribute('r', '10');
        badgeBg.setAttribute('fill', '#000000');
        svg.appendChild(badgeBg);

        const badge = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        badge.setAttribute('x', `${midX}`);
        badge.setAttribute('y', `${midY + 4}`);
        badge.setAttribute('text-anchor', 'middle');
        badge.setAttribute('fill', '#ffffff');
        badge.setAttribute('font-size', '12px');
        badge.setAttribute('font-weight', 'bold');
        badge.textContent = `${num}`;
        svg.appendChild(badge);

        // Description card in hintRoot
        appendHintDescriptionCard(hintRoot, num, '방향 화살표', item.cmd.label);
      } else if (item.kind === 'path') {
        const d = item.points.map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');

        // Black outer path
        const outer = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        outer.setAttribute('d', d);
        outer.setAttribute('fill', 'none');
        outer.setAttribute('stroke', '#000000');
        outer.setAttribute('stroke-width', '5');
        svg.appendChild(outer);

        // White inner path
        const inner = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        inner.setAttribute('d', d);
        inner.setAttribute('fill', 'none');
        inner.setAttribute('stroke', '#ffffff');
        inner.setAttribute('stroke-width', '2.5');
        svg.appendChild(inner);

        // Number badge at start point
        const start = item.points[0];
        const badgeBg = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        badgeBg.setAttribute('cx', `${start.x}`);
        badgeBg.setAttribute('cy', `${start.y}`);
        badgeBg.setAttribute('r', '10');
        badgeBg.setAttribute('fill', '#000000');
        svg.appendChild(badgeBg);

        const badge = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        badge.setAttribute('x', `${start.x}`);
        badge.setAttribute('y', `${start.y + 4}`);
        badge.setAttribute('text-anchor', 'middle');
        badge.setAttribute('fill', '#ffffff');
        badge.setAttribute('font-size', '12px');
        badge.setAttribute('font-weight', 'bold');
        badge.textContent = `${num}`;
        svg.appendChild(badge);

        // Description card in hintRoot
        appendHintDescriptionCard(hintRoot, num, '경로 안내', item.cmd.label);
      } else if (item.kind === 'gesture') {
        const d = item.points.map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');

        // Background guide path (black outer + white inner)
        const outer = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        outer.setAttribute('d', d);
        outer.setAttribute('fill', 'none');
        outer.setAttribute('stroke', '#000000');
        outer.setAttribute('stroke-width', '4');
        outer.setAttribute('stroke-dasharray', '6 4');
        svg.appendChild(outer);

        const inner = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        inner.setAttribute('d', d);
        inner.setAttribute('fill', 'none');
        inner.setAttribute('stroke', '#ffffff');
        inner.setAttribute('stroke-width', '2');
        inner.setAttribute('stroke-dasharray', '6 4');
        svg.appendChild(inner);

        if (reducedMotion) {
          // Static numbered waypoints
          item.points.forEach((p, ptIdx) => {
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', `${p.x}`);
            circle.setAttribute('cy', `${p.y}`);
            circle.setAttribute('r', '6');
            circle.setAttribute('fill', '#ffffff');
            circle.setAttribute('stroke', '#000000');
            circle.setAttribute('stroke-width', '2');
            svg.appendChild(circle);

            const ptText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            ptText.setAttribute('x', `${p.x}`);
            ptText.setAttribute('y', `${p.y + 3}`);
            ptText.setAttribute('text-anchor', 'middle');
            ptText.setAttribute('fill', '#000000');
            ptText.setAttribute('font-size', '9px');
            ptText.setAttribute('font-weight', 'bold');
            ptText.textContent = `${ptIdx + 1}`;
            svg.appendChild(ptText);
          });
        } else {
          // Moving dot animation once over duration_ms (1000..5000ms)
          const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
          dot.setAttribute('r', '7');
          dot.setAttribute('fill', '#ffffff');
          dot.setAttribute('stroke', '#000000');
          dot.setAttribute('stroke-width', '2.5');
          svg.appendChild(dot);

          const duration = item.cmd.duration_ms;
          const startTime = performance.now();
          let animId: number | null = null;

          const animateGesture = (now: number) => {
            if (contextGenerations.get(svg) !== currentGen || !svg.isConnected) {
              if (dot.parentNode) dot.parentNode.removeChild(dot);
              return;
            }

            const elapsed = now - startTime;
            const progress = Math.min(1, Math.max(0, elapsed / duration));

            // Interpolate position along points
            const totalSegments = item.points.length - 1;
            const segProgress = progress * totalSegments;
            const segIndex = Math.min(totalSegments - 1, Math.floor(segProgress));
            const subT = segProgress - segIndex;

            const pA = item.points[segIndex];
            const pB = item.points[segIndex + 1];
            const curX = pA.x + (pB.x - pA.x) * subT;
            const curY = pA.y + (pB.y - pA.y) * subT;

            dot.setAttribute('cx', `${curX}`);
            dot.setAttribute('cy', `${curY}`);

            if (progress < 1) {
              animId = requestAnimationFrame(animateGesture);
            }
          };

          animId = requestAnimationFrame(animateGesture);

          activeGestureCleanups.set(svg, () => {
            if (animId !== null) cancelAnimationFrame(animId);
            if (dot.parentNode) dot.parentNode.removeChild(dot);
          });
        }

        // Description card in hintRoot
        appendHintDescriptionCard(hintRoot, num, `손 제스처 시연 (${item.cmd.duration_ms}ms)`, item.cmd.label);
      } else if (item.kind === 'hint') {
        // Hint reference SVG preview outside video in hint card
        appendHintCardWithReference(hintRoot, num, item.cmd.text, item.cmd.reference);
      }
    }

    // Second rAF: report after DOM is attached and rendered
    requestAnimationFrame(() => {
      if (contextGenerations.get(context.svg) !== currentGen || !context.svg.isConnected || !context.hintRoot.isConnected) {
        resolve({
          status: 'stale',
          command_ids: commandIds,
          reason: 'superseded by a newer render or clear before completion',
        });
        return;
      }

      resolve({
        status: 'rendered',
        command_ids: commandIds,
        reason: null,
      });
    });
  });

  return promise;
}

export interface FramedRenderContext extends RenderContext {
  /** The frame the host is currently displaying. Guidance for any other frame is stale, not rendered. */
  currentFrameId: string;
  /** Ids of the reference assets supplied for this guidance (enables client-side provenance checks). */
  referenceIds?: string[];
  guidanceMode?: 'text' | 'visual';
}

/**
 * The framed entrypoint every host should use: validate the agent-authored envelope, refuse to touch
 * the DOM when it targets a different frame, render through the low-level primitives, and report back
 * exactly what the renderer did.
 */
export function renderGuidance(
  guidance: unknown,
  context: FramedRenderContext
): Promise<FramedRenderReport> {
  const structural = validateGuidanceEnvelope(guidance);
  if (!structural.valid) {
    return Promise.resolve({
      guidance_id: null,
      frame_id: context.currentFrameId,
      status: 'rejected',
      command_ids: [],
      reason: withinReportReasonLimit(structural.errors[0]),
    });
  }

  const framed = guidance as VisualGuidance;
  if (framed.frame_id !== context.currentFrameId) {
    // Checked before any DOM work so a superseded frame can never paint over the current one.
    return Promise.resolve({
      guidance_id: framed.guidance_id,
      frame_id: framed.frame_id,
      status: 'stale',
      command_ids: [],
      reason: withinReportReasonLimit('guidance targets a frame that is no longer displayed'),
    });
  }

  const relational = validateAdviceRelational(framed.advice, {
    guidanceMode: context.guidanceMode ?? 'visual',
    sceneId: framed.frame_id,
    referenceIds: context.referenceIds,
  });
  if (!relational.valid) {
    return Promise.resolve({
      guidance_id: framed.guidance_id,
      frame_id: framed.frame_id,
      status: 'rejected',
      command_ids: [],
      reason: withinReportReasonLimit(relational.errors[0]),
    });
  }

  return renderCommands(framed.advice.commands, context).then((report) => ({
    guidance_id: framed.guidance_id,
    frame_id: framed.frame_id,
    status: report.status,
    command_ids: report.command_ids,
    reason: report.reason === null ? null : withinReportReasonLimit(report.reason),
  }));
}

function appendHintDescriptionCard(
  hintRoot: HTMLElement,
  num: number,
  title: string,
  label: string
) {
  const card = document.createElement('div');
  card.className = 'coach-hint-card';
  card.style.padding = '8px 12px';
  card.style.borderRadius = '8px';
  card.style.background = '#1e293b';
  card.style.border = '1px solid #334155';
  card.style.color = '#f8fafc';
  card.style.fontSize = '13px';
  card.style.display = 'flex';
  card.style.flexDirection = 'column';
  card.style.gap = '2px';

  const header = document.createElement('div');
  header.style.fontWeight = 'bold';
  header.style.color = '#60a5fa';
  header.textContent = `[${num}] ${title}`;
  card.appendChild(header);

  const body = document.createElement('div');
  // Safe plain text
  body.textContent = label;
  card.appendChild(body);

  hintRoot.appendChild(card);
}

function appendHintCardWithReference(
  hintRoot: HTMLElement,
  num: number,
  text: string,
  reference: 'none' | 'line' | 'circle' | 'hatching'
) {
  const card = document.createElement('div');
  card.className = 'coach-hint-card';
  card.style.padding = '8px 12px';
  card.style.borderRadius = '8px';
  card.style.background = '#1e293b';
  card.style.border = '1px solid #3b82f6';
  card.style.color = '#f8fafc';
  card.style.fontSize = '13px';
  card.style.display = 'flex';
  card.style.flexDirection = 'column';
  card.style.gap = '6px';

  const header = document.createElement('div');
  header.style.fontWeight = 'bold';
  header.style.color = '#93c5fd';
  header.textContent = `[${num}] 작업 힌트`;
  card.appendChild(header);

  const textSpan = document.createElement('div');
  // Strict plain text
  textSpan.textContent = text;
  card.appendChild(textSpan);

  // Reference SVG icon (local geometry, NOT hallucinated AI image)
  if (reference !== 'none') {
    const previewSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    previewSvg.setAttribute('viewBox', '0 0 60 24');
    previewSvg.setAttribute('width', '60');
    previewSvg.setAttribute('height', '24');
    previewSvg.style.background = '#0f172a';
    previewSvg.style.borderRadius = '4px';
    previewSvg.style.border = '1px solid #475569';

    if (reference === 'line') {
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', '6');
      line.setAttribute('y1', '12');
      line.setAttribute('x2', '54');
      line.setAttribute('y2', '12');
      line.setAttribute('stroke', '#38bdf8');
      line.setAttribute('stroke-width', '2');
      previewSvg.appendChild(line);
    } else if (reference === 'circle') {
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', '30');
      circle.setAttribute('cy', '12');
      circle.setAttribute('r', '8');
      circle.setAttribute('fill', 'none');
      circle.setAttribute('stroke', '#38bdf8');
      circle.setAttribute('stroke-width', '2');
      previewSvg.appendChild(circle);
    } else if (reference === 'hatching') {
      for (let x = 6; x < 54; x += 8) {
        const hline = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        hline.setAttribute('x1', `${x}`);
        hline.setAttribute('y1', '20');
        hline.setAttribute('x2', `${x + 8}`);
        hline.setAttribute('y2', '4');
        hline.setAttribute('stroke', '#38bdf8');
        hline.setAttribute('stroke-width', '1.5');
        previewSvg.appendChild(hline);
      }
    }

    const refContainer = document.createElement('div');
    refContainer.style.display = 'flex';
    refContainer.style.alignItems = 'center';
    refContainer.style.gap = '8px';

    const refLabel = document.createElement('span');
    refLabel.style.fontSize = '11px';
    refLabel.style.color = '#94a3b8';
    refLabel.textContent = `참고 형상 (${reference}):`;

    refContainer.appendChild(refLabel);
    refContainer.appendChild(previewSvg);
    card.appendChild(refContainer);
  }

  hintRoot.appendChild(card);
}
