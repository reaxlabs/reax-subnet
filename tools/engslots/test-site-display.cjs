const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync(process.argv[2], 'utf8').replace(/load\(\);\s*$/, 'globalThis.done = load();');
(async () => {
  for (const slot of ['2026-10-27T14:30:00Z', 'invalid', null]) {
    let note;
    const host = { querySelector: () => note, appendChild: n => { note = n; } };
    const context = {
      document: {
        querySelector: s => s === '[data-competition="engine-speed"]' ? host : null,
        createElement: () => ({ dataset: {} }),
      },
      fetch: async () => ({ ok: true, json: async () => ({schema: 2, tracks: [], competitions: [],
        engine: { id: 'engine-speed', leaderboard: [], next_slot_utc: slot }}) }),
      CSS: { escape: s => s }, Intl, Date,
    };
    vm.createContext(context); vm.runInContext(source, context); await context.done;
    if (slot === '2026-10-27T14:30:00Z') assert.equal(note.textContent, 'Next engine crowning slot: Tue 27 Oct 14:30 UTC');
    else assert.equal(note, undefined);
  }
  console.log('site display: valid slot exact UTC text; missing/invalid slot omitted (3 cases pass)');
})().catch(err => { console.error(err); process.exitCode = 1; });
