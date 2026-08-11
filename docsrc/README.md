# Project page source

This directory is the renderer-independent source contract for the publication
page. It is intentionally not a Sphinx project and has no local build step.

## Contract

- `project.yml` contains versioned presentation-neutral metadata, ordered navigation
  sections, structured publication links, and an asset manifest.
- `index.md` contains the page narrative in standard Markdown with dollar-delimited
  LaTeX because `math: true` is declared in the metadata.
- Every `sections[].id` in `project.yml` is the lowercase Markdown heading slug
  of the corresponding level-two heading in `index.md`.
- Asset paths are relative to `docsrc/`. External publication and repository
  URLs appear only as structured links or citations.
- `assets/*.svg` are accessible, web-native explanatory graphics. Their origin
  and limitations are recorded in `assets/PROVENANCE.md`.

The VOILA website importer treats `project.yml` as data and `index.md` as
content. The contract does not require Sphinx, MyST directives,
reStructuredText, Python imports, or repository code execution.
