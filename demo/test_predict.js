const fs = require('fs');
const { gpPredict, decide } = require('./predict.js');
const M = JSON.parse(fs.readFileSync(__dirname + '/model.json'));
const out = {};
for (const [name, m] of Object.entries(M.models)) out[name] = m.examples.map(e => { const p = gpPredict(m, e.volts); const d = decide(p, 80, 5, 20); return [e.true_soh, p.soh, p.sd, d.p, d.action]; });
fs.writeFileSync(__dirname + '/js_preds.json', JSON.stringify(out));
for (const [k, v] of Object.entries(out)) console.log(k, v.map(r => `${r[0]}→${r[1].toFixed(2)}±${r[2].toFixed(2)} p=${r[3].toFixed(3)} ${r[4]}`).join(' | '));
