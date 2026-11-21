// Sokoban solver: room-decomposed push-optimal A*.
// Structure: hallway row (player's row) + rooms (connected components off it).
// Each room is solved push-optimally by A* (assignment heuristic + deadlock
// pruning); rooms are stitched with BFS walks and the full move string is
// replay-verified before it is written. Deterministic: no clock, no random,
// fixed neighbour order, expansion-count budgets only.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

type Dir = { dr: number; dc: number; ch: string };
const DIRS: Dir[] = [
  { dr: -1, dc: 0, ch: 'u' },
  { dr: 1, dc: 0, ch: 'd' },
  { dr: 0, dc: -1, ch: 'l' },
  { dr: 0, dc: 1, ch: 'r' },
];

// ---------- grid ----------
type Grid = {
  w: number;
  h: number;
  wall: boolean[];
  goal: boolean[];
  boxes: number[];
  player: number;
};

function parseLevel(text: string): Grid {
  const lines = text.split('\n').map((s) => s.replace(/\r$/, ''));
  while (lines.length > 0 && lines[lines.length - 1] === '') lines.pop();
  const h = lines.length;
  let w = 0;
  for (const l of lines) w = Math.max(w, l.length);
  const wall = new Array<boolean>(w * h).fill(true);
  const goal = new Array<boolean>(w * h).fill(false);
  const boxes: number[] = [];
  let player = 0;
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const ch = c < lines[r].length ? lines[r][c] : '#';
      const i = r * w + c;
      if (ch === '#') { wall[i] = true; continue; }
      wall[i] = false;
      if (ch === '.' || ch === '*' || ch === '+') goal[i] = true;
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  }
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goal, boxes, player };
}

function rc(i: number, w: number): [number, number] {
  return [Math.floor(i / w), i % w];
}

// BFS walk from `from` to `to` over floor cells (wall[] false) avoiding
// `blocked` cells. Returns the move string, or null if unreachable.
// Deterministic: fixed DIRS order, first-found shortest path.
function bfsWalk(
  from: number, to: number, w: number, h: number,
  wall: boolean[], blocked: Set<number>,
): string | null {
  if (from === to) return '';
  const prev = new Int32Array(w * h).fill(-2);
  const prevCh = new Array<string>(w * h).fill('');
  prev[from] = -1;
  const q: number[] = [from];
  let qi = 0;
  while (qi < q.length) {
    const cur = q[qi++];
    const [r, c] = rc(cur, w);
    for (const d of DIRS) {
      const nr = r + d.dr;
      const nc = c + d.dc;
      if (nr < 0 || nr >= h || nc < 0 || nc >= w) continue;
      const ni = nr * w + nc;
      if (wall[ni] || blocked.has(ni) || prev[ni] !== -2) continue;
      prev[ni] = cur;
      prevCh[ni] = d.ch;
      if (ni === to) {
        let s = '';
        let x = ni;
        while (x !== from) { s = prevCh[x] + s; x = prev[x]; }
        return s;
      }
      q.push(ni);
    }
  }
  return null;
}

// BFS distances from `src` over cells where floorOk[i] is true.
function bfsDist(src: number, w: number, h: number, floorOk: boolean[]): Int32Array {
  const dist = new Int32Array(w * h).fill(-1);
  dist[src] = 0;
  const q: number[] = [src];
  let qi = 0;
  while (qi < q.length) {
    const cur = q[qi++];
    const [r, c] = rc(cur, w);
    for (const d of DIRS) {
      const nr = r + d.dr;
      const nc = c + d.dc;
      if (nr < 0 || nr >= h || nc < 0 || nc >= w) continue;
      const ni = nr * w + nc;
      if (!floorOk[ni] || dist[ni] !== -1) continue;
      dist[ni] = dist[cur] + 1;
      q.push(ni);
    }
  }
  return dist;
}

// ---------- push A* ----------
type Push = { box: number; dr: number; dc: number; ch: string };
type ANode = {
  boxes: number[];
  player: number;
  parent: number;
  push: Push | null;
  g: number;
  h: number;
};

type SolveOpts = {
  floorOk: boolean[]; // cells the player may stand on / boxes may occupy
  goals: number[]; // goal cells, same count as boxes
  budget: number; // max expansions
};

// Min-heap of node indices ordered by (f, h, seq).
class Heap {
  a: number[] = [];
  f: number[] = [];
  h: number[] = [];
  s: number[] = [];
  get size(): number { return this.a.length; }
  push(idx: number, f: number, h: number, seq: number): void {
    this.a.push(idx); this.f.push(f); this.h.push(h); this.s.push(seq);
    let i = this.a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.less(i, p)) { this.swap(i, p); i = p; } else break;
    }
  }
  pop(): number {
    const top = this.a[0];
    const last = this.a.length - 1;
    this.a[0] = this.a[last]; this.f[0] = this.f[last];
    this.h[0] = this.h[last]; this.s[0] = this.s[last];
    this.a.pop(); this.f.pop(); this.h.pop(); this.s.pop();
    let i = 0;
    for (;;) {
      const l = 2 * i + 1;
      const r = l + 1;
      let m = i;
      if (l < this.a.length && this.less(l, m)) m = l;
      if (r < this.a.length && this.less(r, m)) m = r;
      if (m === i) break;
      this.swap(i, m); i = m;
    }
    return top;
  }
  less(i: number, j: number): boolean {
    if (this.f[i] !== this.f[j]) return this.f[i] < this.f[j];
    if (this.h[i] !== this.h[j]) return this.h[i] < this.h[j];
    return this.s[i] < this.s[j];
  }
  swap(i: number, j: number): void {
    const t = this.a[i]; this.a[i] = this.a[j]; this.a[j] = t;
    const tf = this.f[i]; this.f[i] = this.f[j]; this.f[j] = tf;
    const th = this.h[i]; this.h[i] = this.h[j]; this.h[j] = th;
    const ts = this.s[i]; this.s[i] = this.s[j]; this.s[j] = ts;
  }
}

function permutations(n: number): number[][] {
  const out: number[][] = [];
  const a: number[] = [];
  for (let i = 0; i < n; i++) a.push(i);
  const rec = (k: number): void => {
    if (k === n) { out.push(a.slice()); return; }
    for (let i = k; i < n; i++) {
      const t = a[k]; a[k] = a[i]; a[i] = t;
      rec(k + 1);
      const t2 = a[k]; a[k] = a[i]; a[i] = t2;
    }
  };
  rec(0);
  return out;
}

// Push-optimal A*. Returns the push list, or null if unsolved / over budget.
function pushAStar(
  g: Grid, boxes0: number[], player0: number, opts: SolveOpts,
): Push[] | null {
  const { w, h } = g;
  const n = boxes0.length;
  const goals = opts.goals;
  const goalSet = new Set<number>(goals);
  const perms = n <= 5 ? permutations(n) : null;
  // distFromGoal[k][cell] = BFS distance from goals[k] over floorOk.
  const distFromGoal: Int32Array[] = goals.map((gl) => bfsDist(gl, w, h, opts.floorOk));
  // deadSquare[i]: cell from which no goal is reachable (static floor).
  const deadSq = new Array<boolean>(w * h).fill(false);
  for (let i = 0; i < w * h; i++) {
    if (!opts.floorOk[i] || goalSet.has(i)) continue;
    let ok = false;
    for (const d of distFromGoal) if (d[i] !== -1) { ok = true; break; }
    if (!ok) deadSq[i] = true;
  }
  const isWallLike = (i: number): boolean => {
    if (i < 0 || i >= w * h) return true;
    return !opts.floorOk[i];
  };
  const cornerDead = (i: number): boolean => {
    if (goalSet.has(i)) return false;
    const [r, c] = rc(i, w);
    const up = isWallLike(i - w);
    const dn = isWallLike(i + w);
    const lf = c === 0 || isWallLike(i - 1);
    const rt = c === w - 1 || isWallLike(i + 1);
    return (up && lf) || (up && rt) || (dn && lf) || (dn && rt);
  };
  const heuristic = (boxes: number[]): number => {
    // Small rooms: exact minimum-cost assignment (admissible, tight).
    // Large fallback states (whole level, many boxes): sum of each box's
    // nearest-goal distance (admissible, cheap; permutations would explode).
    if (perms === null) {
      let s = 0;
      for (const b of boxes) {
        let m = Infinity;
        for (const d of distFromGoal) if (d[b] !== -1 && d[b] < m) m = d[b];
        if (m === Infinity) return Infinity;
        s += m;
      }
      return s;
    }
    let best = Infinity;
    for (const p of perms) {
      let s = 0;
      let bad = false;
      for (let bi = 0; bi < n; bi++) {
        const d = distFromGoal[p[bi]][boxes[bi]];
        if (d === -1) { bad = true; break; }
        s += d;
      }
      if (!bad && s < best) best = s;
    }
    return best;
  };
  const deadlocked = (boxes: number[]): boolean => {
    const set = new Set<number>(boxes);
    const occ = (i: number): boolean => {
      if (i < 0 || i >= w * h) return true;
      if (!opts.floorOk[i]) return true;
      return set.has(i);
    };
    for (const b of boxes) {
      if (goalSet.has(b)) continue;
      if (deadSq[b] || cornerDead(b)) return true;
    }
    // 2x2 freeze: any 2x2 block of boxes/walls with no goal inside.
    for (const b of boxes) {
      if (goalSet.has(b)) continue;
      const [r, c] = rc(b, w);
      for (let dr = 0; dr <= 1; dr++) {
        for (let dc = 0; dc <= 1; dc++) {
          const r0 = r - dr;
          const c0 = c - dc;
          if (r0 < 0 || c0 < 0 || r0 + 1 >= h || c0 + 1 >= w) continue;
          const cells = [
            r0 * w + c0, r0 * w + c0 + 1,
            (r0 + 1) * w + c0, (r0 + 1) * w + c0 + 1,
          ];
          let allOcc = true;
          let hasGoal = false;
          for (const cell of cells) {
            if (goalSet.has(cell)) hasGoal = true;
            if (!occ(cell)) { allOcc = false; break; }
          }
          if (allOcc && !hasGoal) return true;
        }
      }
    }
    return false;
  };

  const keyOf = (boxes: number[], player: number): string =>
    boxes.join(',') + '|' + player;
  const solved = (boxes: number[]): boolean => {
    for (const b of boxes) if (!goalSet.has(b)) return false;
    return true;
  };

  const h0 = heuristic(boxes0);
  if (h0 === Infinity || deadlocked(boxes0)) return null;
  const nodes: ANode[] = [];
  const best = new Map<string, number>();
  const heap = new Heap();
  let seq = 0;
  const startBoxes = boxes0.slice().sort((a, b) => a - b);
  nodes.push({ boxes: startBoxes, player: player0, parent: -1, push: null, g: 0, h: h0 });
  best.set(keyOf(startBoxes, player0), 0);
  heap.push(0, h0, h0, seq++);
  let expansions = 0;
  while (heap.size > 0) {
    const ni = heap.pop();
    const node = nodes[ni];
    if (node.g > (best.get(keyOf(node.boxes, node.player)) ?? Infinity)) continue;
    if (solved(node.boxes)) {
      const pushes: Push[] = [];
      let cur = ni;
      while (nodes[cur].push !== null) {
        pushes.push(nodes[cur].push as Push);
        cur = nodes[cur].parent;
      }
      pushes.reverse();
      return pushes;
    }
    if (++expansions > opts.budget) return null;
    // Player reachable region (boxes as obstacles).
    const boxSet = new Set<number>(node.boxes);
    const reach = new Set<number>();
    const q: number[] = [node.player];
    reach.add(node.player);
    let qi = 0;
    while (qi < q.length) {
      const cur = q[qi++];
      const [r, c] = rc(cur, w);
      for (const d of DIRS) {
        const nr = r + d.dr;
        const nc = c + d.dc;
        if (nr < 0 || nr >= h || nc < 0 || nc >= w) continue;
        const nxi = nr * w + nc;
        if (!opts.floorOk[nxi] || boxSet.has(nxi) || reach.has(nxi)) continue;
        reach.add(nxi);
        q.push(nxi);
      }
    }
    for (let bi = 0; bi < n; bi++) {
      const b = node.boxes[bi];
      const [br, bc] = rc(b, w);
      for (const d of DIRS) {
        const fromR = br - d.dr;
        const fromC = bc - d.dc;
        const toR = br + d.dr;
        const toC = bc + d.dc;
        if (fromR < 0 || fromR >= h || fromC < 0 || fromC >= w) continue;
        if (toR < 0 || toR >= h || toC < 0 || toC >= w) continue;
        const from = fromR * w + fromC;
        const to = toR * w + toC;
        if (!reach.has(from)) continue;
        if (!opts.floorOk[to] || boxSet.has(to)) continue;
        const nb = node.boxes.slice();
        nb[bi] = to;
        nb.sort((a, b2) => a - b2);
        if (deadlocked(nb)) continue;
        const hh = heuristic(nb);
        if (hh === Infinity) continue;
        const np = b; // player ends on the box's old cell
        const ng = node.g + 1;
        const k = keyOf(nb, np);
        if (ng >= (best.get(k) ?? Infinity)) continue;
        best.set(k, ng);
        nodes.push({ boxes: nb, player: np, parent: ni, push: { box: b, dr: d.dr, dc: d.dc, ch: d.ch }, g: ng, h: hh });
        heap.push(nodes.length - 1, ng + hh, hh, seq++);
      }
    }
  }
  return null;
}

// ---------- rooms ----------
type Room = {
  cells: Set<number>;
  door: number; // room cell adjacent to a hallway floor cell
  entry: number; // hallway floor cell adjacent to the door
  boxes: number[];
  goals: number[];
};

function decompose(g: Grid): Room[] | null {
  const { w, h } = g;
  const [pr] = rc(g.player, w);
  const hallwayRow = pr;
  const seen = new Array<boolean>(w * h).fill(false);
  const rooms: Room[] = [];
  for (let i = 0; i < w * h; i++) {
    if (g.wall[i]) continue;
    const [r] = rc(i, w);
    if (r === hallwayRow) continue;
    if (seen[i]) continue;
    // BFS one component of non-hallway floor.
    const comp = new Set<number>();
    const q: number[] = [i];
    seen[i] = true;
    comp.add(i);
    let qi = 0;
    while (qi < q.length) {
      const cur = q[qi++];
      const [cr, cc] = rc(cur, w);
      for (const d of DIRS) {
        const nr = cr + d.dr;
        const nc = cc + d.dc;
        if (nr < 0 || nr >= h || nc < 0 || nc >= w) continue;
        if (nr === hallwayRow) continue;
        const ni = nr * w + nc;
        if (g.wall[ni] || seen[ni]) continue;
        seen[ni] = true;
        comp.add(ni);
        q.push(ni);
      }
    }
    const boxes = g.boxes.filter((b) => comp.has(b));
    if (boxes.length === 0) continue;
    // Door: component cell adjacent to a hallway floor cell.
    let door = -1;
    let entry = -1;
    for (const cell of comp) {
      const [cr, cc] = rc(cell, w);
      for (const d of DIRS) {
        const nr = cr + d.dr;
        const nc = cc + d.dc;
        if (nr !== hallwayRow || nc < 0 || nc >= w) continue;
        const ni = nr * w + nc;
        if (!g.wall[ni]) { door = cell; entry = ni; break; }
      }
      if (door !== -1) break;
    }
    if (door === -1) return null;
    // Goals of this room are exactly the goal cells inside the component.
    const goals = [...comp].filter((cell) => g.goal[cell]);
    rooms.push({ cells: comp, door, entry, boxes, goals });
  }
  // Every box must belong to exactly one room.
  const covered = new Set<number>();
  for (const room of rooms) for (const b of room.boxes) covered.add(b);
  if (covered.size !== g.boxes.length) return null;
  for (const room of rooms) {
    if (room.goals.length !== room.boxes.length) return null;
  }
  rooms.sort((a, b) => {
    const [, ac] = rc(a.door, w);
    const [, bc] = rc(b.door, w);
    if (ac !== bc) return ac - bc;
    const [ar] = rc(a.door, w);
    const [br] = rc(b.door, w);
    return ar - br;
  });
  return rooms;
}

// Replay a push list for one room into moves, updating global boxes/player.
// Floor for walks: all non-wall cells; blocked = current global box cells.
function replayPushes(
  g: Grid, pushes: Push[], boxes: number[], player: number,
): { moves: string; player: number } | null {
  let moves = '';
  let cur = player;
  const boxArr = boxes;
  for (const p of pushes) {
    const [br, bc] = rc(p.box, g.w);
    const from = (br - p.dr) * g.w + (bc - p.dc);
    const blocked = new Set<number>(boxArr);
    const walk = bfsWalk(cur, from, g.w, g.h, g.wall, blocked);
    if (walk === null) return null;
    moves += walk;
    moves += p.ch;
    const idx = boxArr.indexOf(p.box);
    if (idx === -1) return null;
    boxArr[idx] = (br + p.dr) * g.w + (bc + p.dc);
    cur = p.box;
  }
  return { moves, player: cur };
}

function solveLevel(text: string): string {
  const g = parseLevel(text);
  let moves = '';
  try {
    moves = solveDecomposed(g) ?? solveWhole(g) ?? '';
  } catch {
    moves = '';
  }
  if (!moves) return '';
  // Self-verify by replaying on the original level.
  if (!verify(g, moves)) return '';
  return moves;
}

function solveDecomposed(g: Grid): string | null {
  const rooms = decompose(g);
  if (rooms === null) return null;
  const boxes = g.boxes.slice();
  let player = g.player;
  let moves = '';
  for (const room of rooms) {
    // Walk to the entry cell (hallway beside the door).
    const walkIn = bfsWalk(player, room.entry, g.w, g.h, g.wall, new Set<number>(boxes));
    if (walkIn === null) return null;
    moves += walkIn;
    player = room.entry;
    // Room floor = room cells + entry cell.
    const floorOk = new Array<boolean>(g.w * g.h).fill(false);
    for (const cell of room.cells) floorOk[cell] = true;
    floorOk[room.entry] = true;
    const roomBoxes = boxes.filter((b) => room.cells.has(b));
    const pushes = pushAStar(g, roomBoxes, room.entry, {
      floorOk, goals: room.goals, budget: 2000000,
    });
    if (pushes === null) return null;
    const rep = replayPushes(g, pushes, boxes, player);
    if (rep === null) return null;
    moves += rep.moves;
    player = rep.player;
    // Walk back out to the entry cell.
    const walkOut = bfsWalk(player, room.entry, g.w, g.h, g.wall, new Set<number>(boxes));
    if (walkOut === null) return null;
    moves += walkOut;
    player = room.entry;
  }
  return moves;
}

function solveWhole(g: Grid): string | null {
  const floorOk = g.wall.map((x) => !x);
  const goals: number[] = [];
  for (let i = 0; i < g.w * g.h; i++) if (g.goal[i]) goals.push(i);
  const pushes = pushAStar(g, g.boxes.slice(), g.player, {
    floorOk, goals, budget: 2000000,
  });
  if (pushes === null) return null;
  const boxes = g.boxes.slice();
  const rep = replayPushes(g, pushes, boxes, g.player);
  return rep === null ? null : rep.moves;
}

function verify(g: Grid, moves: string): boolean {
  const boxes = new Set<number>(g.boxes);
  let player = g.player;
  for (const ch of moves) {
    const lc = ch.toLowerCase();
    const d = DIRS.find((x) => x.ch === lc);
    if (!d) continue;
    const [r, c] = rc(player, g.w);
    const nr = r + d.dr;
    const nc = c + d.dc;
    if (nr < 0 || nr >= g.h || nc < 0 || nc >= g.w) return false;
    const ni = nr * g.w + nc;
    if (g.wall[ni]) return false;
    if (boxes.has(ni)) {
      const br = nr + d.dr;
      const bc = nc + d.dc;
      if (br < 0 || br >= g.h || bc < 0 || bc >= g.w) return false;
      const bi = br * g.w + bc;
      if (g.wall[bi] || boxes.has(bi)) return false;
      boxes.delete(ni);
      boxes.add(bi);
    }
    player = ni;
  }
  for (const b of boxes) if (!g.goal[b]) return false;
  return true;
}

for (const id of readdirSync(I).sort()) {
  let moves = '';
  try {
    moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  } catch {
    moves = '';
  }
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
