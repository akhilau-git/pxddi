async function checkRisk() {
    const a = document.getElementById('smilesA').value;
    const b = document.getElementById('smilesB').value;
    const resultEl = document.getElementById('result');
    resultEl.innerText = "Checking...";

    try {
        const res = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({smiles_a: a, smiles_b: b})
        });

        if (!res.ok) {
            const err = await res.json();
            resultEl.innerText = `Error: ${err.detail || 'Invalid input'}`;
            return;
        }

        const data = await res.json();
        resultEl.innerHTML = `
            <strong>${data.disclaimer}</strong><br><br>
            Model-estimated interaction likelihood: ${(data.interaction_risk_estimate * 100).toFixed(1)}%<br>
            Drug A toxicity: ${data.drug_a_toxicity.score.toFixed(2)}
                ${data.drug_a_toxicity.known ? '(based on real FAERS data)' : '(UNKNOWN — no data available, not a "safe" result)'}<br>
            Drug B toxicity: ${data.drug_b_toxicity.score.toFixed(2)}
                ${data.drug_b_toxicity.known ? '(based on real FAERS data)' : '(UNKNOWN — no data available, not a "safe" result)'}<br>
            Patient context applied: ${data.patient_context_applied}
        `;
    } catch (e) {
        resultEl.innerText = "Error: could not reach backend. Is it running?";
    }
}
