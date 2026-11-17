// Gaussian-process prediction and the three-way decision, as exported by export_demo.py.
function erf(x) {
  // Abramowitz & Stegun 7.1.26 (max error 1.5e-7)
  const s = Math.sign(x); x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}
function normSf(z) { return 0.5 * (1 - erf(z / Math.SQRT2)); }

function gpPredict(m, volts) {
  const x = [volts[0]].concat(volts.slice(1).map(v => v - volts[0]));
  const z = x.map((v, i) => (v - m.scaler_mean[i]) / m.scaler_scale[i]);
  const n = m.X.length, k = new Float64Array(n);
  let mu = 0;
  for (let i = 0; i < n; i++) {
    let d2 = 0; const Xi = m.X[i];
    for (let j = 0; j < z.length; j++) { const q = (z[j] - Xi[j]) / m.length_scales[j]; d2 += q * q; }
    k[i] = m.const * Math.exp(-0.5 * d2); mu += k[i] * m.alpha[i];
  }
  // forward substitution: L v = k
  const v = new Float64Array(n); let vv = 0;
  for (let i = 0; i < n; i++) {
    const Li = m.L[i]; let s = k[i];
    for (let j = 0; j < i; j++) s -= Li[j] * v[j];
    v[i] = s / Li[i]; vv += v[i] * v[i];
  }
  const sd = m.y_std * Math.sqrt(Math.max(m.const + m.noise - vv, 1e-12));
  const zmax = Math.max(...z.map(Math.abs));
  return { soh: m.y_mean + m.y_std * mu, sd, sdCal: sd * m.calib_scale, zmax };
}

function decide(pred, threshold, cFN, cFP) {
  const p = normSf((threshold - pred.soh) / pred.sdCal);
  const costs = { 'second life': (1 - p) * cFP, recycle: p * cFN, retest: 1 };
  const order = ['second life', 'recycle', 'retest'];
  let best = order[0];
  for (const a of order) if (costs[a] < costs[best] - 1e-12) best = a;
  return { p, costs, action: best, lo: pred.soh - 1.6449 * pred.sdCal, hi: pred.soh + 1.6449 * pred.sdCal };
}

if (typeof module !== 'undefined') module.exports = { gpPredict, decide, normSf };
