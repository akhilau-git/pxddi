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
const metaThreshold = document.getElementById('meta-threshold');
const metaCalib = document.getElementById('meta-calib');
const metaOod = document.getElementById('meta-ood');
const metaReqid = document.getElementById('meta-reqid');
const apiBaseUrl = new URL(
    document.body.dataset.apiBaseUrl || '/api',
    window.location.origin,
).toString().replace(/\/$/, '');

function resetForm() {
    errorState.classList.add('hidden');
    resultsContent.classList.add('hidden');
    emptyState.classList.remove('hidden');
    riskBar.style.width = '0%';
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

function getColorClass(percentage, thresholdPct = 50) {
    if (percentage >= thresholdPct) return 'val-danger';
    if (percentage >= thresholdPct * 0.75) return 'val-warning';
    return 'val-safe';
}

function getBgColor(percentage, thresholdPct = 50) {
    if (percentage >= thresholdPct) return 'var(--danger)';
    if (percentage >= thresholdPct * 0.75) return 'var(--warning)';
    return 'var(--success)';
}

function percentageFromScore(score, label) {
    if (typeof score !== 'number' || !Number.isFinite(score) || score < 0 || score > 1) {
        throw new Error(`The backend returned an invalid ${label}.`);
    }
    return score * 100;
}

async function responseErrorMessage(response) {
    const contentType = response.headers.get('content-type') || '';
    if (contentType.includes('application/json')) {
        try {
            const payload = await response.json();
            if (typeof payload.detail === 'string') return payload.detail;
        } catch (_) {
            // Fall through to a stable, non-technical message.
        }
    }
    return `The request could not be completed (HTTP ${response.status}).`;
}

async function checkRisk() {
    const smilesA = document.getElementById('smilesA').value.trim();
    const smilesB = document.getElementById('smilesB').value.trim();
    
    if (!smilesA || !smilesB) return;

    setLoading(true);

    try {
        const res = await fetch(`${apiBaseUrl}/predict`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({smiles_a: smilesA, smiles_b: smilesB})
        });

        if (!res.ok) {
            throw new Error(await responseErrorMessage(res));
        }

        const data = await res.json();
        
        // Use model decision threshold from checkpoint
        const threshold = (typeof data.decision_threshold_used === 'number' && Number.isFinite(data.decision_threshold_used))
            ? data.decision_threshold_used
            : 0.5;
        const thresholdPct = threshold * 100;
        const interactionPredicted = Boolean(data.interaction_predicted);

        // Populate Data
        const riskPct = percentageFromScore(data.interaction_risk_estimate, 'interaction score');
        riskValue.innerText = `${riskPct.toFixed(1)}%`;
        riskValue.className = `metric-value ${getColorClass(riskPct, thresholdPct)}`;
        
        // Animate the bar
        setTimeout(() => {
            riskBar.style.width = `${riskPct}%`;
            riskBar.style.backgroundColor = getBgColor(riskPct, thresholdPct);
        }, 100);

        if (interactionPredicted) {
            riskDesc.innerHTML = `<span style="color: var(--danger)">Interaction Predicted</span> (>= decision threshold ${thresholdPct.toFixed(1)}%). Research-only output; cannot guide prescribing.`;
        } else if (riskPct >= thresholdPct * 0.75) {
            riskDesc.innerHTML = `<span style="color: var(--warning)">Borderline Score</span> (< threshold ${thresholdPct.toFixed(1)}%). Research-only output; cannot guide prescribing.`;
        } else {
            riskDesc.innerHTML = `<span style="color: var(--success)">Sub-Threshold Score</span> (< threshold ${thresholdPct.toFixed(1)}%). Unreported interaction does not establish that a pair is safe.`;
        }

        // Toxicity
        const toxAPct = percentageFromScore(data.drug_a_toxicity.score, 'Drug A toxicity score');
        toxAValue.innerText = `${toxAPct.toFixed(1)}%`;
        toxAValue.className = `metric-value small ${getColorClass(toxAPct, 50)}`;

        const toxBPct = percentageFromScore(data.drug_b_toxicity.score, 'Drug B toxicity score');
        toxBValue.innerText = `${toxBPct.toFixed(1)}%`;
        toxBValue.className = `metric-value small ${getColorClass(toxBPct, 50)}`;

        // Show only metadata returned by the API
        metaArch.innerText = data.model_architecture || 'Unavailable';
        const evidence = data.stored_validation_evidence;
        metaAuroc.innerText = (
            evidence?.status === 'available' && typeof evidence.auroc === 'number'
                ? `${evidence.auroc.toFixed(4)} (internal validation only)`
                : 'Unavailable'
        );

        if (metaThreshold) {
            metaThreshold.innerText = `${thresholdPct.toFixed(2)}% (${interactionPredicted ? 'Interaction Triggered' : 'Below Cutoff'})`;
        }
        if (metaCalib) {
            const cal = data.score_calibration;
            metaCalib.innerText = cal ? `${cal.status} (${cal.method || 'none'})` : 'Uncalibrated';
        }
        if (metaOod) {
            const dom = data.structural_applicability_domain;
            metaOod.innerText = dom ? (dom.outside_structural_domain ? 'Out-of-Domain (Novel Scaffold)' : 'In-Domain (Similar to Training)') : 'Unavailable';
        }
        
        // A missing header is reported honestly rather than generating a fake ID.
        metaReqid.innerText = res.headers.get('x-request-id') || 'Unavailable';

        // Show Results
        setLoading(false);
        resultsContent.classList.remove('hidden');

    } catch (e) {
        setLoading(false);
        document.getElementById('error-message').innerText = e.message || "Could not connect to backend.";
        errorState.classList.remove('hidden');
    }
}
