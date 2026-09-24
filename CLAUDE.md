# slime-sim

WebGPU slime-mould (physarum) simulation. Single-page, no build step, no bundler,
no dependencies besides a CDN import of `dat.gui` in `index.js`.

- Entry point: `index.html` loads `index.js` as an ES module.
- All logic — shaders, pipeline setup, frame loop — lives in `index.js`.

## Running it

Serve the two static files with any static file server and open in a
WebGPU-capable browser (e.g. Chrome):

```
python3 -m http.server
```

**README.md is stale — ignore it.** It references `python3 make-skinning.py`
and a `dist` folder, neither of which exist in this repo. There is no build
step.

## Architecture

`index.js` defines five WGSL compute shaders and one vertex/fragment pair:

- `AgentShader` — one thread per agent (workgroup size 256). Senses trail +
  food at front/left/right sensor points, turns toward the strongest signal
  (or randomly if at a local peak), moves, wraps position, deposits a trail
  value into the storage texture.
- `DiffuseShader` — one thread per pixel (16×16). 3×3 box blur with
  wraparound + decay + diffuse-rate mix on the trail texture.
- `foodDecayShader` — one thread per pixel (16×16). Fades food texture
  toward brown/transparent over time.
- `destroyShader` / `destroyAgentsShader` — conditional, run only on a plain
  click. Clear trail pixels and respawn agents within a radius of the click.
- `VertShader` / `FragShader` — full-screen triangle; samples trail + food
  textures, applies GUI color, presents to canvas.

Per-frame order (`frame()` in `index.js`): agent pass → diffuse pass →
food decay pass → destroy pass (if triggered) → render pass.

### Ping-pong buffers

- `trailTextures[2]`, indexed by `currentTexture` / `nextTexture = 1 -
  currentTexture`. Agent pass reads `[currentTexture]`, writes
  `[nextTexture]`. Diffuse pass then reads `[nextTexture]` back and writes
  into `[currentTexture]` — i.e. it writes back into the *original* slot,
  not `nextTexture`. `currentTexture` still toggles at the end of every
  frame regardless. This is non-standard ping-pong; understand it before
  reordering the agent/diffuse passes.
- `foodTextures[2]`, indexed by `currentFood` / `1 - currentFood`. Simple
  standard ping-pong: decay pass always reads `[currentFood]`, writes
  `[1-currentFood]`, and `currentFood` toggles once per frame.

### State

- Agent state (position `vec2f`, angle `f32`) lives in one GPU storage
  buffer (`agentBuffer`), fully reinitialized on the GUI's `RESET`.
- Bind groups and pipelines all use `layout: "auto"`; bind groups are
  rebuilt every frame rather than cached.
- `NUM_AGENTS` (250,000) and canvas size are fixed at
  `window.innerWidth`/`innerHeight` at load time — no resize handling.

### Interaction

- Plain click: destroy trail + respawn agents in a 150px radius
  (`window.destroy` flag set, consumed next frame).
- Ctrl+click: paint a food blob at the cursor.
- `f` key: drop a food blob at a random position.
- dat.gui panel: live sensor angle/distance, move/turn speed, decay,
  diffuse rate, trail color, spawn shape (random/circle), and reset.

For exact shader code, read `index.js` directly rather than relying on this
summary.
