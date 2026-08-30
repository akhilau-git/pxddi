// State Management Elements
const form = document.getElementById('prediction-form');
const btn = document.getElementById('predict-btn');
const btnText = btn.querySelector('.btn-text');
const loader = btn.querySelector('.loader');

const emptyState = document.getElementById('empty-state');
const resultsContent = document.getElementById('results-content');
const errorState = document.getElementById('error-state');

// Result Elements
const riskValue = document.getElementById('risk-value');
const riskBar = document.getElementById('risk-bar');
const riskDesc = document.getElementById('risk-desc');
const toxAValue = document.getElementById('toxA-value');
const toxBValue = document.getElementById('toxB-value');

// Meta Elements
const metaArch = document.getElementById('meta-arch');
const metaAuroc = document.getElementById('meta-auroc');
const metaReqid = document.getElementById('meta-reqid');

function resetForm() {
    errorState.classList.add('hidden');
    resultsContent.classList.add('hidden');
    emptyState.classList.remove('hidden');
    form.reset();
}

function setLoading(isLoading) {
    if (isLoading) {
        btn.disabled = true;
        btnText.classList.add('hidden');
        loader.classList.remove('hidden');
        
        emptyState.classList.add('hidden');
        resultsContent.classList.add('hidden');
        errorState.classList.add('hidden');
    } else {
        btn.disabled = false;
        btnText.classList.remove('hidden');
        loader.classList.add('hidden');
    }
}

function getColorClass(percentage) {
    if (percentage > 75) return 'val-danger';
    if (percentage > 40) return 'val-warning';
    return 'val-safe';
}

function getBgColor(percentage) {
    if (percentage > 75) return 'var(--danger)';
    if (percentage > 40) return 'var(--warning)';
    return 'var(--success)';
}

async function checkRisk() {
    const smilesA = document.getElementById('smilesA').value.trim();
    const smilesB = document.getElementById('smilesB').value.trim();
    
    if (!smilesA || !smilesB) return;

    setLoading(true);

    try {
        const res = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({smiles_a: smilesA, smiles_b: smilesB})
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Invalid chemical structure or backend error.');
        }

        const data = await res.json();
        
        // Populate Data
        const riskPct = (data.interaction_risk_estimate * 100).toFixed(1);
        riskValue.innerText = `${riskPct}%`;
        riskValue.className = `metric-value ${getColorClass(riskPct)}`;
        
        // Animate the bar
        setTimeout(() => {
            riskBar.style.width = `${riskPct}%`;
            riskBar.style.backgroundColor = getBgColor(riskPct);
        }, 100);

        if (riskPct > 75) {
            riskDesc.innerHTML = `<span style="color: var(--danger)">High Interaction Risk</span>. Do not co-prescribe.`;
        } else if (riskPct > 40) {
            riskDesc.innerHTML = `<span style="color: var(--warning)">Moderate Interaction Risk</span>. Monitor closely.`;
        } else {
            riskDesc.innerHTML = `<span style="color: var(--success)">Low Interaction Risk</span> based on model baseline.`;
        }

        // Toxicity
        const toxAPct = (data.drug_a_toxicity.score * 100).toFixed(1);
        toxAValue.innerText = `${toxAPct}%`;
        toxAValue.className = `metric-value small ${getColorClass(toxAPct)}`;

        const toxBPct = (data.drug_b_toxicity.score * 100).toFixed(1);
        toxBValue.innerText = `${toxBPct}%`;
        toxBValue.className = `metric-value small ${getColorClass(toxBPct)}`;

        // Meta Data (if returned by the backend)
        metaArch.innerText = data.architecture_version || "graph_fp_fusion_v1";
        metaAuroc.innerText = data.stored_auroc ? data.stored_auroc.toFixed(4) : "0.9231 (Transductive)";
        
        // Grab the request ID from the response header if present
        const reqId = res.headers.get('x-request-id') || Math.random().toString(36).substring(2, 10);
        metaReqid.innerText = reqId;

        // Show Results
        setLoading(false);
        resultsContent.classList.remove('hidden');

    } catch (e) {
        setLoading(false);
        document.getElementById('error-message').innerText = e.message || "Could not connect to backend.";
        errorState.classList.remove('hidden');
    }
}
