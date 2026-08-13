async function checkRisk() {
    const a = document.getElementById('smilesA').value;
    const b = document.getElementById('smilesB').value;
    const res = await fetch('http://localhost:8000/predict', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({smiles_a: a, smiles_b: b})
    });
    const data = await res.json();
    document.getElementById('result').innerText = `Risk: ${(data.interaction_risk*100).toFixed(1)}%`;
}
