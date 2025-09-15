import React, { useCallback, useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Play, Pause, StepForward, RotateCcw, Zap, Settings2, ArrowRight, GitBranch, Activity } from "lucide-react";
// UI (shadcn)
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

// =============================
// ReCoN – Table-1 Strict + ARC Demo (fixed waiting + puzzle selector)
// =============================

// States from the paper
export type NodeState =
  | "inactive"
  | "requested"
  | "active"
  | "suppressed"
  | "waiting"
  | "true"
  | "confirmed"
  | "failed";

export type NodeType = "script" | "terminal";

// ARC grid types
type Grid = number[][];

type Op = (g: Grid) => Grid;

export interface ReCoNNode {
  id: string;
  name: string;
  type: NodeType;
  state: NodeState;
  parent: string | null; // sur target (upwards)
  children: string[]; // sub targets (downwards)
  prev: string | null; // ret
  next: string | null; // por
  // terminal behavior
  op?: Op; // operation applied at terminal
  measureOutcome?: boolean; // fallback (for non-ARC scenarios)
  measureFn?: (ctx: ExecCtx, self: ReCoNNode) => { ok: boolean; produced?: Grid };
  produced?: Grid; // last produced grid (for ARC terminals)
  // layout
  x?: number;
  y?: number;
}

interface Graph {
  nodes: Record<string, ReCoNNode>;
  rootId: string;
}

interface ExecCtx {
  arc?: { input: Grid; target: Grid; name: string };
}

const STATE_COLORS: Record<NodeState, string> = {
  inactive: "bg-slate-400 text-white",
  requested: "bg-sky-500 text-white",
  active: "bg-blue-500 text-white",
  suppressed: "bg-zinc-400 text-white",
  waiting: "bg-amber-500 text-white",
  true: "bg-emerald-500 text-white",
  confirmed: "bg-green-600 text-white",
  failed: "bg-rose-600 text-white",
};

const STATE_LABEL: Record<NodeState, string> = {
  inactive: "inactive",
  requested: "requested",
  active: "active",
  suppressed: "suppressed",
  waiting: "waiting",
  true: "true (success, not yet propagating)",
  confirmed: "confirmed (success propagated)",
  failed: "failed",
};

// =============================
// ARC helpers
// =============================

const ARC_PALETTE: string[] = [
  "#000000", // 0 black
  "#0074D9", // 1 blue
  "#FF4136", // 2 red
  "#2ECC40", // 3 green
  "#FFDC00", // 4 yellow
  "#AAAAAA", // 5 gray
  "#F012BE", // 6 magenta
  "#FF851B", // 7 orange
  "#7FDBFF", // 8 light blue
  "#870C25", // 9 maroon
];

function gridEq(a: Grid, b: Grid): boolean {
  if (!a || !b || a.length !== b.length) return false;
  for (let r = 0; r < a.length; r++) {
    if (a[r].length !== b[r].length) return false;
    for (let c = 0; c < a[r].length; c++) if (a[r][c] !== b[r][c]) return false;
  }
  return true;
}

function mirrorH(g: Grid): Grid { // horizontal mirror (flip columns)
  return g.map((row) => [...row].reverse());
}

function mirrorV(g: Grid): Grid { // vertical mirror (flip rows)
  return [...g].reverse();
}

function rot90(g: Grid): Grid {
  const rows = g.length, cols = g[0].length;
  const out: Grid = Array.from({ length: cols }, () => Array(rows).fill(0));
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) out[c][rows - 1 - r] = g[r][c];
  return out;
}

function rot270(g: Grid): Grid { // 270 = rotate right three times
  const rows = g.length, cols = g[0].length;
  const out: Grid = Array.from({ length: cols }, () => Array(rows).fill(0));
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) out[cols - 1 - c][r] = g[r][c];
  return out;
}

const MIRROR_INPUT: Grid = [
  [1, 1, 1],
  [1, 0, 0],
  [1, 0, 0],
];
const MIRROR_TARGET: Grid = mirrorH(MIRROR_INPUT);

const ROT_INPUT: Grid = [
  [0, 1, 0],
  [0, 1, 0],
  [0, 1, 0],
];
const ROT_TARGET: Grid = rot90(ROT_INPUT);

const PUZZLES = {
  mirror: {
    key: "mirror",
    name: "Mirror H vs V",
    input: MIRROR_INPUT,
    target: MIRROR_TARGET,
    ops: [
      { name: "Mirror H", op: mirrorH },
      { name: "Mirror V", op: mirrorV },
    ] as { name: string; op: Op }[],
  },
  rotate: {
    key: "rotate",
    name: "Rotate 90 vs 270",
    input: ROT_INPUT,
    target: ROT_TARGET,
    ops: [
      { name: "Rotate 90", op: rot90 },
      { name: "Rotate 270", op: rot270 },
    ] as { name: string; op: Op }[],
  },
};

// =============================
// Scenario builders
// =============================

function makeNode(id: string, name: string, type: NodeType, overrides: Partial<ReCoNNode> = {}): ReCoNNode {
  return { id, name, type, state: "inactive", parent: null, children: [], prev: null, next: null, ...overrides };
}

function connectSub(g: Graph, parentId: string, childId: string) {
  g.nodes[parentId].children.push(childId);
  g.nodes[childId].parent = parentId;
}

function connectSeq(g: Graph, firstId: string, secondId: string) {
  g.nodes[firstId].next = secondId; // por
  g.nodes[secondId].prev = firstId; // ret
}

function buildScenarioARC(puzzle: typeof PUZZLES[keyof typeof PUZZLES]): Graph {
  // Root -> Hyp1(op1) | Hyp2(op2)
  const nodes: Record<string, ReCoNNode> = {};
  nodes["root"] = makeNode("root", "Root", "script");
  nodes["hyp1"] = makeNode("hyp1", `Hyp: ${puzzle.ops[0].name}`, "script");
  nodes["hyp2"] = makeNode("hyp2", `Hyp: ${puzzle.ops[1].name}`, "script");
  nodes["term1"] = makeNode("term1", puzzle.ops[0].name, "terminal", {
    op: puzzle.ops[0].op,
    measureFn: (ctx, self) => {
      const out = self.op ? self.op(ctx.arc!.input) : ctx.arc!.input;
      return { ok: gridEq(out, ctx.arc!.target), produced: out };
    },
  });
  nodes["term2"] = makeNode("term2", puzzle.ops[1].name, "terminal", {
    op: puzzle.ops[1].op,
    measureFn: (ctx, self) => {
      const out = self.op ? self.op(ctx.arc!.input) : ctx.arc!.input;
      return { ok: gridEq(out, ctx.arc!.target), produced: out };
    },
  });

  const g: Graph = { nodes, rootId: "root" };
  connectSub(g, "root", "hyp1");
  connectSub(g, "root", "hyp2");
  connectSub(g, "hyp1", "term1");
  connectSub(g, "hyp2", "term2");
  return g;
}

// Toy scenarios (unchanged)
function buildScenarioAlternatives(): Graph {
  const nodes: Record<string, ReCoNNode> = {};
  nodes["root"] = makeNode("root", "Root", "script");
  nodes["A"] = makeNode("A", "Hyp A", "script");
  nodes["B"] = makeNode("B", "Hyp B", "script");
  nodes["C"] = makeNode("C", "Hyp C", "script");
  nodes["tA"] = makeNode("tA", "Check A", "terminal", { measureOutcome: true });
  nodes["tB"] = makeNode("tB", "Check B", "terminal", { measureOutcome: false });
  nodes["tC"] = makeNode("tC", "Check C", "terminal", { measureOutcome: true });
  const g: Graph = { nodes, rootId: "root" };
  connectSub(g, "root", "A");
  connectSub(g, "root", "B");
  connectSub(g, "root", "C");
  connectSub(g, "A", "tA");
  connectSub(g, "B", "tB");
  connectSub(g, "C", "tC");
  return g;
}

function buildScenarioSequence(): Graph {
  const nodes: Record<string, ReCoNNode> = {};
  nodes["root"] = makeNode("root", "Root", "script");
  nodes["S"] = makeNode("S", "Sequence", "script");
  nodes["s1"] = makeNode("s1", "Step 1", "terminal", { measureOutcome: true });
  nodes["s2"] = makeNode("s2", "Step 2", "terminal", { measureOutcome: true });
  nodes["s3"] = makeNode("s3", "Step 3", "terminal", { measureOutcome: true });
  const g: Graph = { nodes, rootId: "root" };
  connectSub(g, "root", "S");
  connectSub(g, "S", "s1");
  connectSub(g, "S", "s2");
  connectSub(g, "S", "s3");
  connectSeq(g, "s1", "s2");
  connectSeq(g, "s2", "s3");
  return g;
}

// =============================
// Layout
// =============================

function layoutGraph(g: Graph) {
  const LEVEL_Y = 140;
  const NODE_X = 160;
  const { nodes } = g;
  const depth: Record<string, number> = {};
  function dfs(id: string, d: number) {
    depth[id] = Math.max(depth[id] ?? 0, d);
    g.nodes[id].children.forEach((cid) => dfs(cid, d + 1));
  }
  dfs(g.rootId, 0);
  const byDepth: Record<number, string[]> = {};
  Object.keys(nodes).forEach((id) => {
    const d = depth[id] ?? 0;
    if (!byDepth[d]) byDepth[d] = [];
    byDepth[d].push(id);
  });
  Object.values(byDepth).forEach((ids) => ids.sort());
  Object.entries(byDepth).forEach(([dStr, ids]) => {
    const d = Number(dStr);
    const rowWidth = (ids.length - 1) * NODE_X;
    ids.forEach((id, i) => {
      nodes[id].x = -rowWidth / 2 + i * NODE_X;
      nodes[id].y = d * LEVEL_Y;
    });
  });
}

// =============================
// Strict Table-1 Engine (with ARC exec ctx)
// =============================

function isDone(s: NodeState) { return s === "true" || s === "confirmed" || s === "failed"; }

function cloneGraph(g: Graph): Graph {
  return { rootId: g.rootId, nodes: Object.fromEntries(Object.entries(g.nodes).map(([k, v]) => [k, { ...v }])) };
}

interface TickMessage {
  type: "request" | "confirm" | "inhibit" | "wait"; // Table-1 doesn’t explicitly send 'fail'
  from: string;
  to: string;
  link: "sub" | "sur" | "por" | "ret";
}

function stepReCoNStrict(g: Graph, ctx: ExecCtx): { next: Graph; changed: boolean; messages: TickMessage[] } {
  const curr = g;
  const next = cloneGraph(g);
  const messages: TickMessage[] = [];
  const get = (id: string) => curr.nodes[id];

  // Phase 1: emit messages exactly per Table-1
  for (const id of Object.keys(curr.nodes)) {
    const n = get(id);
    const emit = (type: TickMessage["type"], to: string, link: TickMessage["link"]) => messages.push({ type, from: id, to, link });

    const sendInhibitReqOnPor = n.state === "requested" || n.state === "active" || n.state === "suppressed" || n.state === "waiting" || n.state === "failed";
    const sendInhibitConfOnRet = n.state !== "inactive";

    if (n.next && sendInhibitReqOnPor) emit("inhibit", n.next, "por");
    if (n.prev && sendInhibitConfOnRet) emit("inhibit", n.prev, "ret");

    if ((n.state === "active" || n.state === "waiting") && n.children.length > 0) {
      for (const cid of n.children) emit("request", cid, "sub");
    }
    if ((n.state === "requested" || n.state === "active" || n.state === "waiting") && n.parent) emit("wait", n.parent, "sur");
    if (n.state === "confirmed" && n.parent) emit("confirm", n.parent, "sur");
  }

  // Phase 2: collect inbox
  type Inbox = { reqFromParent: boolean; inhibitReqFromPrev: boolean; inhibitConfFromNext: boolean; waitsFromChildren: number; confirmFromChild: boolean };
  const inbox: Record<string, Inbox> = Object.fromEntries(Object.keys(curr.nodes).map((id) => [id, { reqFromParent: false, inhibitReqFromPrev: false, inhibitConfFromNext: false, waitsFromChildren: 0, confirmFromChild: false }]));
  for (const m of messages) {
    const ib = inbox[m.to];
    if (!ib) continue;
    if (m.link === "sub" && m.type === "request") ib.reqFromParent = true;
    if (m.link === "por" && m.type === "inhibit") ib.inhibitReqFromPrev = true;
    if (m.link === "ret" && m.type === "inhibit") ib.inhibitConfFromNext = true;
    if (m.link === "sur" && m.type === "wait") ib.waitsFromChildren += 1;
    if (m.link === "sur" && m.type === "confirm") ib.confirmFromChild = true;
  }

  // Phase 3: transition (waiting doesn't fail while any child is 'true')
  for (const id of Object.keys(curr.nodes)) {
    const n = curr.nodes[id];
    const ib = inbox[id];
    const set = (s: NodeState) => { next.nodes[id].state = s; };

    switch (n.state) {
      case "inactive": { if (ib.reqFromParent) set("requested"); break; }
      case "requested": { if (ib.inhibitReqFromPrev) set("suppressed"); else set("active"); break; }
      case "suppressed": { if (!ib.inhibitReqFromPrev) set("requested"); break; }
      case "active": {
        if (n.type === "terminal") {
          let ok = !!n.measureOutcome; let produced: Grid | undefined;
          if (n.measureFn && ctx) { const r = n.measureFn(ctx, n); ok = r.ok; produced = r.produced; }
          if (produced) next.nodes[id].produced = produced; // write output on activation
          if (ok) { if (ib.inhibitConfFromNext) set("true"); else set("confirmed"); } else { set("failed"); }
        } else { set("waiting"); }
        break;
      }
      case "waiting": {
        const childStates = n.children.map((cid) => curr.nodes[cid]?.state as NodeState);
        const allChildrenDone = n.children.length > 0 && childStates.every((s) => isDone(s));
        const anyChildTrue = childStates.some((s) => s === "true");
        if (ib.confirmFromChild) set("true");
        else if (allChildrenDone && !anyChildTrue) set("failed");
        break;
      }
      case "true": { if (!ib.inhibitConfFromNext) set("confirmed"); break; }
      case "confirmed":
      case "failed": break;
    }
  }

  let changed = false;
  for (const id of Object.keys(curr.nodes)) if (curr.nodes[id].state !== next.nodes[id].state) changed = true;
  return { next, changed, messages };
}

// =============================
// SVG Graph Pieces
// =============================

function ArrowMarkerDefs() {
  return (
    <defs>
      <marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" />
      </marker>
    </defs>
  );
}

function Edge({ x1, y1, x2, y2, kind, active }: { x1: number; y1: number; x2: number; y2: number; kind: "sub" | "sur" | "por" | "ret"; active?: boolean }) {
  const color = kind === "sub" ? "stroke-slate-500" : kind === "sur" ? "stroke-slate-400" : kind === "por" ? "stroke-blue-500" : "stroke-rose-500";
  const dash = kind === "sur" ? "2 4" : kind === "ret" ? "2 4" : undefined;
  const thickness = active ? 3 : 2;
  const marker = kind === "sub" || kind === "por" ? "url(#arrow)" : undefined;
  return (
    <motion.line x1={x1} y1={y1} x2={x2} y2={y2} className={`${color}`} strokeWidth={thickness} strokeDasharray={dash} markerEnd={marker} initial={{ opacity: 0.7 }} animate={{ opacity: active ? 1 : 0.7 }} transition={{ duration: 0.3 }} />
  );
}

function NodeBubble({ node, highlight }: { node: ReCoNNode; highlight?: boolean }) {
  const color = STATE_COLORS[node.state];
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <motion.g initial={{ scale: 0.95 }} animate={{ scale: highlight ? 1.08 : 1 }} transition={{ type: "spring", stiffness: 220, damping: 18 }}>
            <g transform={`translate(${node.x}, ${node.y})`}>
              <circle r={28} className={`${color} drop-shadow`} />
              <text textAnchor="middle" dy={5} className="fill-white text-xs font-semibold">{node.name}</text>
            </g>
          </motion.g>
        </TooltipTrigger>
        <TooltipContent>
          <div className="text-xs">
            <div className="font-semibold mb-1">{node.name}</div>
            <div>type: {node.type}</div>
            <div>state: {STATE_LABEL[node.state]}</div>
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

// =============================
// Grid UI
// =============================

function Cell({ v, size = 20 }: { v: number; size?: number }) {
  const bg = ARC_PALETTE[v] ?? "#222";
  return <div style={{ width: size, height: size, background: bg, border: "1px solid #e2e8f0" }} />;
}

function GridView({ grid, title }: { grid?: Grid; title: string }) {
  return (
    <div>
      <div className="text-xs mb-1 text-slate-600">{title}</div>
      {grid ? (
        <div className="inline-flex flex-col bg-white p-1 rounded border">
          {grid.map((row, ri) => (
            <div key={ri} className="flex">
              {row.map((v, ci) => (
                <Cell key={ci} v={v} />
              ))}
            </div>
          ))}
        </div>
      ) : (
        <div className="w-[80px] h-[80px] flex items-center justify-center text-[10px] text-slate-400 border border-dashed border-slate-300 rounded">pending</div>
      )}
    </div>
  );
}

// =============================
// Main Component
// =============================

export default function ReCoNVisualizer() {
  const [scenario, setScenario] = useState<string>("arc");
  const [arcKey, setArcKey] = useState<keyof typeof PUZZLES>("mirror");
  const [graph, setGraph] = useState<Graph>(() => buildScenarioARC(PUZZLES["mirror"]));
  const [ctx, setCtx] = useState<ExecCtx>({ arc: { input: PUZZLES["mirror"].input, target: PUZZLES["mirror"].target, name: PUZZLES["mirror"].name } });
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(0.9); // sec / tick
  const [stepCount, setStepCount] = useState(0);
  const [showMsgs, setShowMsgs] = useState(true);
  const [lastMsgs, setLastMsgs] = useState<TickMessage[]>([]);

  // layout & reset when scenario or puzzle changes
  useEffect(() => {
    if (scenario === "arc") {
      const p = PUZZLES[arcKey];
      const g = buildScenarioARC(p);
      layoutGraph(g);
      setGraph(g);
      setCtx({ arc: { input: p.input, target: p.target, name: p.name } });
    } else if (scenario === "alts") {
      const g = buildScenarioAlternatives(); layoutGraph(g); setGraph(g); setCtx({});
    } else {
      const g = buildScenarioSequence(); layoutGraph(g); setGraph(g); setCtx({});
    }
    setRunning(false);
    setStepCount(0);
    setLastMsgs([]);
  }, [scenario, arcKey]);

  const resetScenario = useCallback(() => {
    if (scenario === "arc") {
      const p = PUZZLES[arcKey];
      const g = buildScenarioARC(p); layoutGraph(g); setGraph(g); setCtx({ arc: { input: p.input, target: p.target, name: p.name } });
    } else if (scenario === "alts") { const g = buildScenarioAlternatives(); layoutGraph(g); setGraph(g); setCtx({}); }
    else { const g = buildScenarioSequence(); layoutGraph(g); setGraph(g); setCtx({}); }
    setRunning(false); setStepCount(0); setLastMsgs([]);
  }, [scenario, arcKey]);

  const tick = useCallback(() => {
    const { next, changed, messages } = stepReCoNStrict(graph, ctx);
    setGraph(next);
    setStepCount((c) => c + 1);
    setLastMsgs(messages);
    const root = next.nodes[next.rootId];
    if (root.state === "confirmed" || root.state === "failed" || !changed) setRunning(false);
  }, [graph, ctx]);

  useEffect(() => { if (!running) return; const t = setInterval(() => tick(), Math.max(0.2, speed) * 1000); return () => clearInterval(t); }, [running, speed, tick]);

  const onRequestRoot = () => {
    const g = cloneGraph(graph);
    const r = g.nodes[g.rootId];
    if (r.state === "inactive") {
      r.state = "requested";
      layoutGraph(g);
      setGraph(g);
      setStepCount((c) => c + 1);
      setLastMsgs([{ type: "request", from: "user", to: r.id, link: "sub" }]);
    }
  };

  const edges = useMemo(() => {
    const arr: { x1: number; y1: number; x2: number; y2: number; kind: "sub" | "sur" | "por" | "ret"; key: string }[] = [];
    for (const n of Object.values(graph.nodes)) {
      for (const cid of n.children) {
        const c = graph.nodes[cid];
        arr.push({ x1: n.x!, y1: n.y! + 30, x2: c.x!, y2: c.y! - 30, kind: "sub", key: `${n.id}->${cid}` });
        arr.push({ x1: c.x!, y1: c.y! - 30, x2: n.x!, y2: n.y! + 30, kind: "sur", key: `${cid}->${n.id}` });
      }
      if (n.next) {
        const s = graph.nodes[n.next];
        arr.push({ x1: n.x! + 30, y1: n.y!, x2: s.x! - 30, y2: s.y!, kind: "por", key: `${n.id}>>${s.id}` });
        arr.push({ x1: s.x! - 30, y1: s.y!, x2: n.x! + 30, y2: n.y!, kind: "ret", key: `${s.id}<<${n.id}` });
      }
    }
    return arr;
  }, [graph]);

  const isActiveEdge = useCallback((edge: { key: string; kind: "sub" | "sur" | "por" | "ret" }) => {
    if (!showMsgs) return false;
    return lastMsgs.some((m) => {
      const key = m.link === "sub" ? `${m.from}->${m.to}` : m.link === "sur" ? `${m.from}->${m.to}` : m.link === "por" ? `${m.from}>>${m.to}` : `${m.from}<<${m.to}`;
      return key === edge.key;
    });
  }, [lastMsgs, showMsgs]);

  const root = graph.nodes[graph.rootId];

  // Winner + produced grid (generic)
  const winner = useMemo(() => {
    if (scenario !== "arc") return null;
    const kids = graph.nodes[graph.rootId].children.map((id) => graph.nodes[id]);
    const winHyp = kids.find((n) => n.state === "confirmed" || n.state === "true");
    if (!winHyp) return null;
    const termId = graph.nodes[winHyp.id].children[0];
    const produced = termId ? (graph.nodes[termId].produced as Grid | undefined) : undefined;
    return { name: winHyp.name.replace(/^Hyp: /, ""), produced };
  }, [graph, scenario]);

  const puzzle = PUZZLES[arcKey];

  return (
    <div className="w-full min-h-[100vh] bg-gradient-to-b from-slate-50 to-white p-6">
      <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls */}
        <Card className="lg:col-span-1 shadow-md">
          <CardHeader className="space-y-2">
            <CardTitle className="flex items-center gap-2 text-xl"><Activity className="w-5 h-5" /> ReCoN – Table‑1 + ARC Demo</CardTitle>
            <p className="text-sm text-slate-500">Exact Table‑1 messaging with selectable ARC puzzles and live grid manipulation.</p>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <div className="text-xs uppercase tracking-wide text-slate-500">Scenario</div>
              <Select value={scenario} onValueChange={(v) => setScenario(v)}>
                <SelectTrigger className="w-full"><SelectValue placeholder="Pick a scenario" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="arc">ARC (Hypothesis OR)</SelectItem>
                  <SelectItem value="alts">Toy: Alternatives (OR)</SelectItem>
                  <SelectItem value="seq">Toy: Sequence (AND)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {scenario === "arc" && (
              <div className="space-y-2">
                <div className="text-xs uppercase tracking-wide text-slate-500">ARC Puzzle</div>
                <Select value={arcKey} onValueChange={(v) => setArcKey(v as keyof typeof PUZZLES)}>
                  <SelectTrigger className="w-full"><SelectValue placeholder="Pick a puzzle" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="mirror">Mirror H vs V</SelectItem>
                    <SelectItem value="rotate">Rotate 90 vs 270</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            )}

            <div className="flex items-center gap-2">
              <Button variant="secondary" onClick={resetScenario} className="gap-2"><RotateCcw className="w-4 h-4" /> Reset</Button>
              <Button onClick={onRequestRoot} className="gap-2"><Zap className="w-4 h-4" /> Request Root</Button>
            </div>

            <div className="flex items-center gap-2">
              <Button variant={running ? "secondary" : "default"} onClick={() => setRunning((r) => !r)} className="gap-2">{running ? <><Pause className="w-4 h-4" /> Pause</> : <><Play className="w-4 h-4" /> Play</>}</Button>
              <Button variant="outline" onClick={() => tick()} className="gap-2"><StepForward className="w-4 h-4" /> Step</Button>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between"><div className="text-xs uppercase tracking-wide text-slate-500">Speed</div><div className="text-xs text-slate-600">{speed.toFixed(2)}s / tick</div></div>
              <Slider value={[speed]} min={0.2} max={2} step={0.1} onValueChange={(v) => setSpeed(v[0])} />
            </div>

            <Separator />

            <div className="flex items-center justify-between">
              <div className="text-sm text-slate-700">Show edge messages</div>
              <Switch checked={showMsgs} onCheckedChange={setShowMsgs} />
            </div>

            <div className="text-xs text-slate-500">Last tick messages:</div>
            <div className="flex flex-wrap gap-2 min-h-[28px]">
              {lastMsgs.length === 0 && <span className="text-xs text-slate-400">(none)</span>}
              {lastMsgs.map((m, i) => (
                <Badge key={i} variant={m.type === "confirm" ? "default" : "secondary"}>
                  {m.type} · {m.link} · {m.from}→{m.to}
                </Badge>
              ))}
            </div>

            <Separator />
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="p-2 rounded border bg-white">
                <div className="text-slate-500">Step</div>
                <div className="font-semibold">{stepCount}</div>
              </div>
              <div className="p-2 rounded border bg-white">
                <div className="text-slate-500">Root State</div>
                <div className="font-semibold capitalize">{root?.state}</div>
              </div>
            </div>

            <Separator />
            <div>
              <div className="text-xs uppercase tracking-wide text-slate-500 mb-2">Legend</div>
              <div className="flex flex-wrap gap-2 text-xs">
                {Object.entries(STATE_COLORS).map(([s, cls]) => (
                  <div key={s} className="flex items-center gap-2 px-2 py-1 rounded border border-slate-200">
                    <span className={`inline-block w-3 h-3 rounded ${cls}`} />
                    <span className="text-slate-700">{s}</span>
                  </div>
                ))}
              </div>
              <div className="mt-3 text-xs text-slate-500 flex items-center gap-3">
                <div className="flex items-center gap-1 text-blue-600"><ArrowRight className="w-4 h-4" /> por</div>
                <div className="flex items-center gap-1 text-rose-600"><ArrowRight className="w-4 h-4 rotate-180" /> ret</div>
                <div className="flex items-center gap-1 text-slate-600"><GitBranch className="w-4 h-4" /> sub/sur</div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Canvas + ARC panel */}
        <div className="lg:col-span-2 grid gap-6">
          <Card className="shadow-md">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg"><Settings2 className="w-5 h-5" /> Network – Table‑1 <span className="text-xs font-normal text-slate-500">(Request Root, then Play/Step)</span></CardTitle>
            </CardHeader>
            <CardContent>
              <div className="relative w-full h-[560px] rounded-lg border bg-white overflow-hidden">
                <svg viewBox="-500 -40 1000 600" className="w-full h-full">
                  <ArrowMarkerDefs />
                  {edges.map((e) => (
                    <Edge key={e.key} x1={e.x1!} y1={e.y1!} x2={e.x2!} y2={e.y2!} kind={e.kind} active={isActiveEdge(e)} />
                  ))}
                  {Object.values(graph.nodes).map((n) => (
                    <NodeBubble key={n.id} node={n} highlight={lastMsgs.some((m) => m.from === n.id || m.to === n.id)} />
                  ))}
                </svg>
              </div>
            </CardContent>
          </Card>

          {scenario === "arc" && (
            <Card className="shadow-md">
              <CardHeader>
                <CardTitle className="text-lg">ARC Demo: {puzzle.name}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-6 items-start">
                  <GridView grid={puzzle.input} title="Input" />
                  <GridView grid={puzzle.target} title="Target" />
                  <div>
                    <div className="text-xs mb-1 text-slate-600">Candidate: {puzzle.ops[0].name}</div>
                    <GridView grid={graph.nodes["term1"].produced} title={`Produced (${puzzle.ops[0].name.split(" ")[0]})`} />
                  </div>
                  <div>
                    <div className="text-xs mb-1 text-slate-600">Candidate: {puzzle.ops[1].name}</div>
                    <GridView grid={graph.nodes["term2"].produced} title={`Produced (${puzzle.ops[1].name.split(" ")[0]})`} />
                  </div>
                </div>
                <Separator className="my-4" />
                <div className="text-sm">
                  {root.state === "confirmed" && winner ? (
                    <div>
                      <span className="font-semibold">Solved via:</span> {winner.name}
                    </div>
                  ) : root.state === "failed" ? (
                    <div className="text-rose-600">No hypothesis confirmed.</div>
                  ) : (
                    <div className="text-slate-500">Run the network — terminals write their produced grids when they enter <span className="font-mono">active</span>.</div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {/* Footer note */}
      <div className="max-w-6xl mx-auto mt-6 text-[13px] text-slate-500">
        <p>
          This simulator follows <span className="font-medium">Table‑1</span> message-passing exactly. Parents in <em>waiting</em> no longer fail while a child is <span className="font-mono">true</span> (success awaiting confirmation). The ARC panel now supports multiple puzzles and shows only the grids that terminals actually produced during execution.
        </p>
      </div>
    </div>
  );
}
