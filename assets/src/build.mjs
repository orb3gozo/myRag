// Builds the myRag logo and banner: an isometric filing cabinet whose top drawer is
// pulled open, with one index card lifted out of the file. The cabinet is the vector
// store, the cards are the indexed chunks and the lifted card is the retrieved one.
//
// Usage: node assets/src/build.mjs
// Writes assets/{logo,banner}-{dark,light}.svg.

import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as k from "./vendor/iso-kit.mjs";

const OUT_DIR = join(dirname(fileURLToPath(import.meta.url)), "..");

// World units are about 1 cm of a real three-drawer cabinet (62 deep, 47 wide).
const CASE = { x: -31, y: -23.5, w: 62, d: 47, r: 1.2 };
const PLINTH = { x: -29, y: -21.5, w: 58, d: 43, r: 0.8 };
const PLINTH_H = 5;
const SLOT_H = 32;
const DRAWERS = 3;
const CASE_TOP = PLINTH_H + SLOT_H * DRAWERS;
const CAP = { x: -31.6, y: -24.1, w: 63.2, d: 48.2, r: 1.6 };
const CAP_H = 3;
const FRONT_X = CASE.x + CASE.w;
const FRONT_W = 1.4;
const FRONT_Y = 22.2;

const OPEN = DRAWERS - 1;
const PULLED = 30;
const WALL_H = 14;
const CARD_H = 24;
const CARD_STEP = 1.6;
const CARD_Y = 19.5;
const LIFT = 17;
const DIVIDERS = new Map([[3, -12], [10, 4]]);
const PICKED = 7;

const slotZ = (index) => PLINTH_H + SLOT_H * index;

function cabinet(P) {
  const at = (point) => k.iso(point, P);
  const S = (paths, style) => k.solidSvg(paths, style);
  const L = (d, style) => k.lineSvg(d, style);
  const D = (points, style) => k.dotsSvg(points, style);
  const seg = (a, b) => k.segment(a, b, P);
  const out = [];

  // Base: halo, recessed plinth, case body and top cap.
  out.push(`<path class="iso-halo" d="${k.haloOf(k.insetPlan(CASE, -4), 0, 0.1, P)}" style="filter:blur(6px)"/>`);
  out.push(S(k.slabOf(PLINTH, 0, PLINTH_H, P, 4, 0.4), { tone: "lo" }));
  out.push(S(k.slabOf(CASE, PLINTH_H, CASE_TOP - PLINTH_H, P, 4, 0.6), { tone: "mid" }));

  // Side panel: rear seam, a row of rivets along the front edge, and a label plate.
  const sideY = CASE.y + CASE.d;
  out.push(L(seg([CASE.x + 3, sideY, PLINTH_H], [CASE.x + 3, sideY, CASE_TOP]), { tone: "faint" }));
  const rivets = [];
  for (let z = PLINTH_H + 6; z < CASE_TOP - 2; z += 12) rivets.push(at([FRONT_X - 2.2, sideY, z]));
  out.push(D(rivets, { size: 0.45, tone: "lo" }));
  const plate = [
    [-14, sideY, CASE_TOP - 12],
    [8, sideY, CASE_TOP - 12],
    [8, sideY, CASE_TOP - 22],
    [-14, sideY, CASE_TOP - 22],
  ];
  out.push(`<path class="iso-line" data-tone="lo" d="${k.pathOf(plate.map(at), true)}"/>`);
  for (const z of [CASE_TOP - 15, CASE_TOP - 17, CASE_TOP - 19]) {
    out.push(L(seg([-11, sideY, z], [z === CASE_TOP - 19 ? -2 : 5, sideY, z]), { tone: "faint", free: true }));
  }
  out.push(D([at([-12.6, sideY, CASE_TOP - 13.4]), at([6.6, sideY, CASE_TOP - 13.4]), at([-12.6, sideY, CASE_TOP - 20.6]), at([6.6, sideY, CASE_TOP - 20.6])], { size: 0.4, tone: "mid" }));

  // The open slot shows the dark cavity behind the drawer.
  const z0 = slotZ(OPEN);
  const recess = [
    [FRONT_X, -FRONT_Y, z0 + 0.8],
    [FRONT_X, FRONT_Y, z0 + 0.8],
    [FRONT_X, FRONT_Y, z0 + SLOT_H - 0.4],
    [FRONT_X, -FRONT_Y, z0 + SLOT_H - 0.4],
  ];
  out.push(`<path class="iso-halo" d="${k.pathOf(recess.map(at), true)}"/>`);
  out.push(`<path class="iso-line" data-tone="lo" d="${k.pathOf(recess.map(at), true)}"/>`);

  // Closed drawers, bottom first so each pull paints over the front below it.
  for (let index = 0; index < OPEN; index++) out.push(drawerFront(P, FRONT_X, slotZ(index)));

  out.push(S(k.slabOf(CAP, CASE_TOP, CAP_H, P, 4, 0.8), { tone: "mid" }));
  out.push(L(k.planOutline(k.insetPlan(CAP, 3), CASE_TOP + CAP_H, P, 4), { tone: "faint" }));
  out.push(D(k.corners(CAP, 5).map(([x, y]) => at([x, y, CASE_TOP + CAP_H])), { size: 0.45, tone: "lo" }));

  out.push(openDrawer(P, z0));
  return out.join("");
}

function drawerFront(P, x, z0) {
  const at = (point) => k.iso(point, P);
  const S = (paths, style) => k.solidSvg(paths, style);
  const out = [];
  const top = z0 + SLOT_H - 0.4;
  const bottom = z0 + 0.8;
  out.push(S(k.slabOf({ x, y: -FRONT_Y, w: FRONT_W, d: FRONT_Y * 2, r: 0.5 }, bottom, top - bottom, P, 3, 0.4), { tone: "mid" }));
  const face = x + FRONT_W;

  // Label holder: a frame with a paper card behind its lip and three printed lines.
  const holderZ = top - 10;
  out.push(S(k.slabOf({ x: face, y: -7.5, w: 0.6, d: 15, r: 0.3 }, holderZ, 6.4, P, 2), { tone: "mid", crease: "none" }));
  out.push(S(k.slabOf({ x: face + 0.6, y: -6.5, w: 0.3, d: 13, r: 0 }, holderZ + 0.9, 4.6, P, 1), { tone: "hi", crease: "none" }));
  const card = face + 0.9;
  for (const [z, to] of [[holderZ + 4.3, 4], [holderZ + 3.2, 2], [holderZ + 2.1, 3]]) {
    out.push(k.lineSvg(k.segment([card, -5, z], [card, to, z], P), { tone: "lo", free: true }));
  }
  out.push(k.dotsSvg([at([face + 0.6, -6.9, holderZ + 3.2]), at([face + 0.6, 6.9, holderZ + 3.2])], { size: 0.4, tone: "mid" }));

  // Pull: two posts and a bar, far post first.
  const pullZ = holderZ - 6;
  for (const y of [-9, 7]) {
    out.push(S(k.slabOf({ x: face, y, w: 2.6, d: 2, r: 0.4 }, pullZ - 1, 2, P, 2), { tone: "mid", crease: "none" }));
  }
  out.push(S(k.slabOf({ x: face + 2.2, y: -10, w: 1.6, d: 20, r: 0.8 }, pullZ - 1.1, 2.2, P, 4, 0.3), { tone: "hi", crease: "lo" }));
  return out.join("");
}

function openDrawer(P, z0) {
  const S = (paths, style) => k.solidSvg(paths, style);
  const L = (d, style) => k.lineSvg(d, style);
  const D = (points, style) => k.dotsSvg(points, style);
  const at = (point) => k.iso(point, P);
  const out = [];
  const x0 = FRONT_X;
  const floorZ = z0 + 2;
  const wallZ = floorZ + 1;

  out.push(S(k.slabOf({ x: x0, y: -21.5, w: PULLED, d: 43, r: 0 }, floorZ, 1, P, 1), { tone: "lo" }));
  out.push(S(k.slabOf({ x: x0, y: -21.5, w: PULLED, d: 0.8, r: 0 }, wallZ, WALL_H, P, 1), { tone: "mid" }));

  // Index cards stand across the drawer, back to front. Two guide cards carry tabs,
  // and the picked card rides higher than the rest: the retrieved chunk.
  const count = Math.floor((PULLED - 4) / CARD_STEP);
  for (let index = 0; index < count; index++) {
    const cx = x0 + 2 + index * CARD_STEP;
    const picked = index === PICKED;
    const z = wallZ + (picked ? LIFT : 0);
    const plan = { x: cx, y: -CARD_Y, w: 0.4, d: CARD_Y * 2, r: 0 };
    out.push(S(k.slabOf(plan, z, CARD_H, P, 1), { tone: picked ? "hi" : "mid", crease: "none", lit: picked }));
    const tab = DIVIDERS.get(index);
    if (tab !== undefined) {
      out.push(S(k.slabOf({ x: cx, y: tab, w: 0.4, d: 9, r: 0 }, z + CARD_H, 3, P, 1), { tone: "mid", crease: "none" }));
    }
    if (picked) {
      const face = cx + 0.4;
      const top = z + CARD_H;
      out.push(L(k.segment([face, -CARD_Y, top - 3], [face, CARD_Y, top - 3], P), { tone: "lo" }));
      for (const [dz, to] of [[6, 12], [8.2, 15], [10.4, 9], [12.6, 13], [14.8, 6]]) {
        out.push(L(k.segment([face, -15, top - dz], [face, to, top - dz], P), { tone: "lo", free: true }));
      }
      out.push(D([at([face, -CARD_Y + 2, top - 1.5])], { size: 0.55, tone: "hi" }));
    }
  }

  // Near wall, then the telescopic slide on its outside with ball-cage rivets.
  out.push(S(k.slabOf({ x: x0, y: 20.7, w: PULLED, d: 0.8, r: 0 }, wallZ, WALL_H, P, 1), { tone: "mid" }));
  const slideZ = wallZ + 6;
  out.push(S(k.slabOf({ x: x0, y: 21.5, w: PULLED - 2, d: 0.7, r: 0 }, slideZ, 3.4, P, 1), { tone: "lo" }));
  out.push(S(k.slabOf({ x: x0, y: 22.2, w: PULLED - 9, d: 0.5, r: 0 }, slideZ + 0.6, 2.2, P, 1), { tone: "mid", crease: "none" }));
  const balls = [];
  for (let x = x0 + 3; x < x0 + PULLED - 10; x += 3.5) balls.push(at([x, 22.7, slideZ + 1.7]));
  out.push(D(balls, { size: 0.35, tone: "lo" }));

  out.push(drawerFront(P, x0 + PULLED, z0));
  return out.join("");
}

function projectionFor(width, height, pad) {
  const reach = FRONT_X + PULLED + FRONT_W + 4;
  const points = [
    ...k.boxCorners(CAP, 0, CASE_TOP + CAP_H),
    ...k.boxCorners({ x: FRONT_X, y: -FRONT_Y, w: reach - FRONT_X, d: FRONT_Y * 2 + 2 }, slotZ(OPEN), slotZ(OPEN) + SLOT_H + LIFT),
  ];
  return k.fitProjection(points, width, height, { pad, azimuth: 38 });
}

const LABEL =
  "An isometric filing cabinet with three drawers. The top drawer is pulled out on its slides, full of index cards, and one card stands higher than the rest.";

const BG = (width, height, radius) =>
  `<rect x="0.5" y="0.5" width="${width - 1}" height="${height - 1}" rx="${radius}" style="fill:var(--anatomy-card);stroke:var(--anatomy-faint);stroke-width:1"/>`;

function logo(theme) {
  const size = 512;
  const P = projectionFor(size, size, 56);
  return k.figureSvg({ width: size, height: size, label: LABEL, standalone: theme, body: [BG(size, size, 96), cabinet(P)] });
}

function banner(theme) {
  const width = 1280;
  const height = 400;
  const art = 440;
  const P = projectionFor(art, height, 40);
  const text = [
    `<text x="520" y="190" style="font:600 112px system-ui,-apple-system,'Segoe UI',sans-serif;letter-spacing:-3px;fill:var(--anatomy-strong)">myRag</text>`,
    `<text x="524" y="248" style="font:400 30px system-ui,-apple-system,'Segoe UI',sans-serif;fill:var(--anatomy-text)">Retrieval-augmented generation on a laptop CPU</text>`,
    `<text x="525" y="300" style="font:400 20px ui-monospace,'SF Mono',Menlo,Consolas,monospace;fill:var(--anatomy-muted)">LlamaIndex · LanceDB · fastembed · Qwen3</text>`,
  ];
  return k.figureSvg({
    width,
    height,
    label: `myRag. ${LABEL}`,
    standalone: theme,
    body: [BG(width, height, 24), `<g transform="translate(50 0)">${cabinet(P)}</g>`, ...text],
  });
}

// Replaces every CSS custom property with its value for the theme, so the files render
// the same in viewers that don't support var(), such as librsvg.
function paletteOf(theme) {
  const declarations = (selector) => {
    const start = k.ISO_CSS.indexOf(`${selector}{`);
    const body = k.ISO_CSS.slice(start + selector.length + 1, k.ISO_CSS.indexOf("}", start));
    return [...body.matchAll(/(--anatomy-[a-z0-9-]+):([^;]+);?/g)].map(([, name, value]) => [name, value.trim()]);
  };
  const palette = new Map(declarations(".iso"));
  if (theme === "light") for (const [name, value] of declarations('.iso[data-theme="light"]')) palette.set(name, value);
  return palette;
}

function resolveVars(svg, theme) {
  const palette = paletteOf(theme);
  return svg.replace(/var\((--anatomy-[a-z0-9-]+)\)/g, (match, name) => palette.get(name) ?? match);
}

for (const theme of ["dark", "light"]) {
  writeFileSync(join(OUT_DIR, `logo-${theme}.svg`), resolveVars(logo(theme), theme));
  writeFileSync(join(OUT_DIR, `banner-${theme}.svg`), resolveVars(banner(theme), theme));
}
console.log(`wrote logo and banner (dark, light) to ${OUT_DIR}`);
