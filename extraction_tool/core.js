// Streaming extractor for Zhu et al. 2022 cycling CSVs.
// Columns: time/s, control/V/mA, Ecell/V, <I>/mA, Q discharge/mA.h, Q charge/mA.h, control/V, control/mA, cycle number
// For each cycle it keeps only: the rest that follows the charge (raw time and voltage),
// the charge capacity before it, and the discharge capacity and end voltage after it.
function makeExtractor(dataset, cell, emit) {
  let buf = '';
  let header = true;
  let cyc = null;
  let segs = [];
  let seg = null;
  let k0 = NaN, k1 = NaN, k2 = NaN;
  let rowsSeen = 0;

  const kindOf = (cV, cI) => (cV > 0 ? 'cv' : cI > 0 ? 'chg' : cI < 0 ? 'dis' : 'rest');

  function finishCycle() {
    if (cyc === null) return;
    if (seg) segs.push(seg);
    seg = null;
    let r = -1;
    for (let i = 1; i < segs.length; i++) {
      if (segs[i].kind === 'rest' && (segs[i - 1].kind === 'cv' || segs[i - 1].kind === 'chg')) { r = i; break; }
    }
    if (r > 0) {
      let d = -1;
      for (let i = r + 1; i < segs.length; i++) { if (segs[i].kind === 'dis') { d = i; break; } }
      if (d > 0) {
        const rest = segs[r], dis = segs[d], chg = segs[r - 1];
        emit([
          dataset, cell, cyc,
          dis.qd.toFixed(2), chg.qc.toFixed(2), dis.vlast.toFixed(4), rest.ts.length,
          rest.ts.map(x => x.toFixed(1)).join(';'),
          rest.vs.map(v => v.toFixed(5)).join(';'),
        ].join(','));
      }
    }
    segs = [];
  }

  function row(line) {
    if (line.endsWith('\r')) line = line.slice(0, -1);
    if (!line) return;
    if (header) { header = false; return; }
    const p = line.split(',');
    if (p.length < 9 || p[8] === '') return;
    const t = +p[0], V = +p[2], Qd = +p[4], Qc = +p[5], ctrl = +p[1], cV = +p[6], cI = +p[7], c = +p[8];
    rowsSeen++;
    if (c !== cyc) { finishCycle(); cyc = c; k0 = NaN; }
    if (ctrl !== k0 || cV !== k1 || cI !== k2) {
      if (seg) segs.push(seg);
      seg = { kind: kindOf(cV, cI), t0: t, ts: [], vs: [], qd: 0, qc: 0, vlast: V };
      k0 = ctrl; k1 = cV; k2 = cI;
    }
    if (seg.kind === 'rest') { seg.ts.push(t - seg.t0); seg.vs.push(V); }
    if (Qd > seg.qd) seg.qd = Qd;
    if (Qc > seg.qc) seg.qc = Qc;
    seg.vlast = V;
  }

  return {
    push(text) {
      buf += text;
      let i, start = 0;
      while ((i = buf.indexOf('\n', start)) !== -1) { row(buf.slice(start, i)); start = i + 1; }
      buf = buf.slice(start);
    },
    end() { if (buf) row(buf); buf = ''; finishCycle(); return rowsSeen; },
  };
}

const HEADER = 'dataset,cell,cycle,cap_mAh,qc_mAh,v_dis_end,n_rest,rest_t,rest_v';

if (typeof module !== 'undefined') module.exports = { makeExtractor, HEADER };
