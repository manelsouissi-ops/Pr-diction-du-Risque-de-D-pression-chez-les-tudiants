document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('predict-form');
  if (!form) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const btn = form.querySelector('button[type=submit]');
    btn.textContent = 'Analyse en cours…';
    btn.disabled = true;

    try {
      const resp = await fetch('/predict', {
        method: 'POST',
        body: new FormData(form),
      });
      const data = await resp.json();

      if (data.error) {
        alert('Erreur : ' + data.error);
        return;
      }

      showResult(data);
    } catch (err) {
      alert('Erreur réseau : ' + err.message);
    } finally {
      btn.textContent = 'Analyser mon profil';
      btn.disabled = false;
    }
  });
});

function showResult(data) {
  const panel = document.getElementById('result-panel');
  const box   = document.getElementById('result-box');

  box.className = 'result-box risk-' + data.risk_color;

  document.getElementById('result-icon').textContent =
    data.risk_color === 'success' ? '✅' : '⚠️';

  document.getElementById('result-risk').textContent =
    'Risque ' + data.risk_label;

  document.getElementById('result-confidence').textContent =
    'Confiance : ' + data.confidence + '%';

  const safe = data.probability_not_depressed;
  const risk = data.probability_depressed;
  document.getElementById('prob-safe').style.width = safe + '%';
  document.getElementById('prob-risk').style.width = risk + '%';
  document.getElementById('prob-safe-pct').textContent = safe + '%';
  document.getElementById('prob-risk-pct').textContent = risk + '%';

  panel.style.display = 'block';
  panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function hideResult() {
  const panel = document.getElementById('result-panel');
  if (panel) panel.style.display = 'none';
}
