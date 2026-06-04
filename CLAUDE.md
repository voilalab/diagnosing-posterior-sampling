# Claude instructions

## General instructions

- Always run Python entrypoints with `uv run`.
- Never call `python`, `pip`, or `pytest` directly unless I explicitly ask.
- Do not add additional dependencies to the project.
- Keep commands scoped to this repo.
- Before potentially long commands, state the exact command and why it is necessary.

## Tools

- You have access to the following tools, which can be run from within `uv`:
  - `ruff check` for format checking
  - `ruff check --fix` for safe fixes
  - `ty check` for type checking
  - `pydoclint --style="google"` for doc linting
  - `pydocstyle --convention="google"` for doc style
- To automatically call the tools on touched files only, run `bash tools.sh`

## Git management

- When asked to implement a plan, switch to a fresh git branch with a descriptive title
  - force user to clean `git` working tree before branching to avoid conflicts
- If the user states `Issue resolved` as a full message,
  - run `bash tools.sh` to check for error and warnings on touched files,
  - resolve errors and warnings,
  - commit the changes,
  - merge to main, and
  - delete the branch.

## Documentation

- Write docstrings in Google style/convention
- Use RST formatting for code and for math
  - Do not use unicode characters for math symbols in docs (e.g. use \nabla not ∇)

## Plotting

Default to matplotlib, but avoid cartoon look from default styling.
Fix the defaults rather than reaching for another tool.

### Style

- Set rcParams once at module load. Do not repeat styling inside plotting functions.
- Prefer `plt.style.use(['science'])` (SciencePlots) as a baseline. Override sparingly.
- Thin lines (`linewidth` ~0.8), hairline spines, small markers. Remove top/right spines unless the data needs them.
- Use a serif font for papers, a clean sans (Helvetica, Inter) for slides. Pick one and commit.
- Colormaps: `viridis` for sequential, `RdBu_r` for divergent, `cmocean`/`cmcrameri` for domain work. Never `jet`. Never `rainbow`.
- Label axes with units. Math goes in `$...$`. No title on figures destined for papers — the caption is the title.

### Resolution and output

- Vector formats (PDF, SVG) for static figures going into LaTeX. Raster (PNG at 300+ dpi) only when the figure has >10k elements or rasterized layers (`set_rasterized(True)` on the heavy artist, vector everything else).
- Set `figsize` in inches to match the target column width. Do not rely on the renderer to scale — text sizes go wrong when you do.
- `bbox_inches='tight'` on save. `pad_inches=0.02` for tight layouts.

### Animations

- Render frames in Python, encode with ffmpeg. Do not let mpl pick defaults for either step.
- Reuse artists across frames. `quiver.set_UVC(...)`, `line.set_data(...)`, `im.set_array(...)`. Never `ax.clear()` per frame — it is 10–100× slower and breaks blitting.
- `FuncAnimation(..., blit=True)` for the integrated path. Return the changed artists from the update function.
- Supersample: render at 2× the final resolution, let ffmpeg downsample with `lanczos`. This is the cheapest anti-aliasing you will get for quiver arrows and streamlines.
- Encode with `libx264`, `-crf 18`, `-pix_fmt yuv420p`. The pixel format is required for QuickTime and most web players. Both output dimensions must be even.
- Decouple frame rendering from encoding (PNGs to disk, then `ffmpeg -framerate ... -i
  frame_%05d.png ...`) when frames are expensive, when you want to parallelize, or when a
  crash mid-render would hurt.

### Vector fields specifically

- `streamplot` reads better than `quiver` for dense fields. Use `quiver` when individual vectors carry meaning.
- Overlay vectors on a `pcolormesh` or `contourf` of magnitude (or a related scalar) for depth. Set the scalar layer's alpha low enough that the vectors stay legible.
- Subsample the grid for `quiver` (`x[::k], y[::k], u[::k, ::k], v[::k, ::k]`). Visual density beyond ~40 arrows per axis becomes noise.
- Fix `vmin`/`vmax` across all frames of an animation. A colorbar that rescales per frame is a bug, not a feature.

### When to reach past matplotlib

- **VisPy** or **Datoviz** when the field has >100k vectors and mpl drops below interactive framerates.
- **ParaView** or **PyVista** for genuine 3D scientific visualization. Do not fight mpl's
  `mplot3d` for this — it was not built for it.

### Anti-patterns

- Styling inside the plotting function instead of via rcParams.
- Recreating the figure inside an animation update.
- Saving at 72 dpi and scaling up in the document.
- Mixing rasterized and vector elements without `set_rasterized` — the whole figure rasterizes.
- Default colormap on divergent data centered at zero.
- Legend with a frame and a shadow on a publication figure.

## Behavior

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.