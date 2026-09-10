/**
 * Frontend type surface.
 *
 * `generated.ts` is produced by `make types` from the backend pydantic
 * contracts and is git-ignored. Importing through this barrel keeps components
 * decoupled from the generated filename, so regenerating or renaming the
 * artefact does not touch any component.
 *
 * Types here mirror the backend exactly. If a component needs a shape that is
 * not in this list, add it to the pydantic model and re-run `make types` rather
 * than declaring it locally — a hand-written frontend type is how the two sides
 * drift apart.
 */

// The first offline MVP keeps its small UI contract next to the API client.
export {}
