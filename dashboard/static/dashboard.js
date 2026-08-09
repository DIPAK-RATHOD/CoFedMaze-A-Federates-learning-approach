/**
 * CoFedMaze Research Dashboard JavaScript
 * Handles API data fetching, real-time polling, moving average smoothing,
 * historical log handling, and Chart.js visualizations.
 */

let currentNode = "N1";
let maWindow = 10;
let isPolling = true;
let pollTimer = null;

// Chart instances
let chartLoss = null;
let chartReward = null;
let chartSuccessRate = null;
let chartEvalSuccess = null;
let chartEvalSteps = null;
let chartEvalReward = null;
let chartEvalBreakdown = null;
let chartCompareRewards = null;
let chartCompareLoss = null;

document.addEventListener("DOMContentLoaded", () => {
    initChartDefaults();
    initEventListeners();
    fetchDashboardData();
    startPolling();
});

function initChartDefaults() {
    Chart.defaults.color = "#94a3b8";
    Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.plugins.tooltip.padding = 10;
    Chart.defaults.plugins.tooltip.cornerRadius = 6;
}

function initEventListeners() {
    // Node selection buttons
    document.querySelectorAll(".node-btn").forEach(btn => {
        btn.addEventListener("click", (e) => {
            document.querySelectorAll(".node-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            currentNode = btn.getAttribute("data-node");

            if (currentNode === "ALL NODES") {
                document.getElementById("all-nodes-section").style.display = "block";
            } else {
                document.getElementById("all-nodes-section").style.display = "none";
            }
            fetchDashboardData();
        });
    });

    // Moving average window buttons
    document.querySelectorAll(".ma-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".ma-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            maWindow = parseInt(btn.getAttribute("data-ma"));
            fetchDashboardData();
        });
    });

    // Live polling toggle
    const toggle = document.getElementById("live-polling-toggle");
    toggle.addEventListener("change", (e) => {
        isPolling = e.target.checked;
        if (isPolling) startPolling();
        else stopPolling();
    });
}

function startPolling() {
    stopPolling();
    pollTimer = setInterval(() => {
        if (isPolling) fetchDashboardData();
    }, 3000);
}

function stopPolling() {
    if (pollTimer) clearInterval(pollTimer);
}

function computeMovingAverage(data, windowSize) {
    if (!data || data.length === 0) return [];
    let result = [];
    for (let i = 0; i < data.length; i++) {
        if (i < windowSize - 1) {
            result.push(null);
        } else {
            let sum = 0;
            let count = 0;
            for (let j = i - windowSize + 1; j <= i; j++) {
                if (data[j] !== null && data[j] !== undefined && !isNaN(data[j])) {
                    sum += data[j];
                    count++;
                }
            }
            result.push(count > 0 ? sum / count : null);
        }
    }
    return result;
}

async function fetchDashboardData() {
    try {
        const response = await fetch(`/api/metrics?node=${encodeURIComponent(currentNode)}`);
        const json = await response.json();

        if (json.mode === "all") {
            renderAllNodesDashboard(json.nodes);
        } else {
            renderSingleNodeDashboard(json.data);
        }
        document.getElementById("telemetry-time").innerText = new Date().toLocaleTimeString();
    } catch (err) {
        console.error("Dashboard fetch error:", err);
    }
}

function renderSingleNodeDashboard(nodeData) {
    const history = nodeData.training_history || [];
    const evalHistory = nodeData.evaluation_history || [];
    const hasGoal = nodeData.has_goal_logged;
    const latest = history.length > 0 ? history[history.length - 1] : {};

    // Telemetry header
    document.getElementById("telemetry-node").innerText = nodeData.node_id;
    document.getElementById("telemetry-run-id").innerText = nodeData.current_run_id || "run_1";
    document.getElementById("telemetry-episode").innerText = latest.episode || 0;
    document.getElementById("telemetry-steps").innerText = latest.total_env_steps || 0;
    document.getElementById("telemetry-epsilon").innerText = latest.epsilon !== undefined ? latest.epsilon.toFixed(3) : "1.000";

    // Training KPIs
    document.getElementById("kpi-train-episode").innerText = latest.episode || 0;
    document.getElementById("kpi-train-steps").innerText = latest.total_env_steps || 0;
    document.getElementById("kpi-train-epsilon").innerText = latest.epsilon !== undefined ? latest.epsilon.toFixed(3) : "1.000";
    document.getElementById("kpi-train-loss").innerText = latest.loss !== undefined && latest.loss !== null ? latest.loss.toFixed(4) : "N/A";

    const rewards = history.map(h => h.reward);
    const last10 = rewards.slice(-10);
    const avgReward10 = last10.length > 0 ? (last10.reduce((a, b) => a + b, 0) / last10.length).toFixed(3) : "0.000";
    document.getElementById("kpi-train-reward").innerText = avgReward10;

    // Evaluation KPIs & Historical Log Handling
    if (!hasGoal) {
        document.getElementById("kpi-eval-success-rate").innerText = "N/A";
        document.getElementById("kpi-eval-success-sub").innerText = "Goal completion not logged for this run";
        document.getElementById("kpi-eval-avg-steps").innerText = "N/A";
        document.getElementById("kpi-eval-reward").innerText = "N/A";
        document.getElementById("kpi-eval-timeout-rate").innerText = "N/A";
        document.getElementById("success-rate-na-overlay").style.display = "flex";
    } else {
        document.getElementById("success-rate-na-overlay").style.display = "none";
        const successful = history.filter(h => h.goal_reached === true);
        const succRate = history.length > 0 ? ((successful.length / history.length) * 100).toFixed(1) + "%" : "0.0%";
        document.getElementById("kpi-eval-success-rate").innerText = succRate;
        document.getElementById("kpi-eval-success-sub").innerText = `${successful.length} of ${history.length} episodes solved`;

        const stepsList = successful.map(h => h.steps).filter(s => s > 0);
        const avgSteps = stepsList.length > 0 ? (stepsList.reduce((a,b) => a+b,0) / stepsList.length).toFixed(1) : "N/A";
        document.getElementById("kpi-eval-avg-steps").innerText = avgSteps;

        const evalRews = evalHistory.map(e => e.evaluation_reward).filter(r => r !== undefined && r !== null);
        document.getElementById("kpi-eval-reward").innerText = evalRews.length > 0 ? evalRews[evalRews.length - 1].toFixed(3) : "N/A";

        const timeouts = history.filter(h => h.timeout === true).length;
        document.getElementById("kpi-eval-timeout-rate").innerText = history.length > 0 ? ((timeouts / history.length) * 100).toFixed(1) + "%" : "0.0%";
    }

    // Chart 1: Loss
    const episodes = history.map(h => h.episode);
    const lossValues = history.map(h => h.loss !== undefined ? h.loss : null);
    const lossMA = computeMovingAverage(lossValues, maWindow);
    renderLossChart(episodes, lossValues, lossMA);

    // Chart 2: Reward
    const rewardValues = history.map(h => h.reward);
    const rewardMA = computeMovingAverage(rewardValues, maWindow);
    renderRewardChart(episodes, rewardValues, rewardMA);

    // Chart 3: Success Rate
    const successValues = history.map(h => h.goal_reached ? 100 : (h.goal_reached === false ? 0 : null));
    const successMA = computeMovingAverage(successValues, maWindow);
    renderSuccessRateChart(episodes, successValues, successMA);

    // Diagnostic interpretation
    updateDiagnosticAnalysis(lossValues, rewardValues, successValues, hasGoal);

    // Evaluation Section Charts (4, 5, 6, 7)
    renderEvaluationSectionCharts(history, evalHistory, hasGoal);
}

function updateDiagnosticAnalysis(losses, rewards, successes, hasGoal) {
    const titleEl = document.getElementById("diagnostic-title");
    const descEl = document.getElementById("diagnostic-desc");
    const diagBox = document.getElementById("diagnostic-box");

    const lossStatus = document.getElementById("diag-loss-status");
    const rewardStatus = document.getElementById("diag-reward-status");
    const successStatus = document.getElementById("diag-success-status");
    const termStatus = document.getElementById("diag-term-status");

    let isLossDecreasing = false;
    if (losses.length >= 10) {
        const firstHalf = losses.slice(0, Math.floor(losses.length / 2)).filter(v => v !== null);
        const secondHalf = losses.slice(Math.floor(losses.length / 2)).filter(v => v !== null);
        const avg1 = firstHalf.reduce((a, b) => a + b, 0) / (firstHalf.length || 1);
        const avg2 = secondHalf.reduce((a, b) => a + b, 0) / (secondHalf.length || 1);
        if (avg2 < avg1 * 0.9) isLossDecreasing = true;
    }

    let isRewardImproving = false;
    if (rewards.length >= 10) {
        const first10 = rewards.slice(0, 10).reduce((a, b) => a + b, 0) / 10;
        const last10 = rewards.slice(-10).reduce((a, b) => a + b, 0) / 10;
        if (last10 > first10 + 0.5) isRewardImproving = true;
    }

    if (isLossDecreasing) lossStatus.innerText = "Decreasing ↓ (Optimizing)";
    else lossStatus.innerText = "Fluctuating / Early";

    if (isRewardImproving) {
        rewardStatus.innerText = "Improving ↑";
        rewardStatus.className = "diag-val green";
        document.getElementById("reward-trend-badge").innerText = "TREND: IMPROVING ↑";
    } else {
        rewardStatus.innerText = "Fluctuating / Plateaued ↔";
        rewardStatus.className = "diag-val yellow";
        document.getElementById("reward-trend-badge").innerText = "TREND: PLATEAUED ↔";
    }

    if (!hasGoal) {
        successStatus.innerText = "N/A (Not Logged)";
        successStatus.className = "diag-val yellow";
    } else {
        const recentSucc = successes.slice(-10).filter(s => s === 100).length;
        if (recentSucc > 5) {
            successStatus.innerText = "High Success ↑";
            successStatus.className = "diag-val green";
        } else {
            successStatus.innerText = "Low / Goal Unreached ↓";
            successStatus.className = "diag-val red";
        }
    }

    termStatus.innerText = "Max Step Timeout";

    if (isLossDecreasing && !isRewardImproving) {
        diagBox.className = "diagnostic-box warning-box";
        titleEl.innerText = "Optimization vs Behavioral Performance Disconnect";
        descEl.innerText = "Neural network loss is decreasing (model optimization improving), but behavioral rewards and goal success rates have not improved consistently. This confirms that decreasing loss alone does NOT imply maze-solving success.";
    } else if (isLossDecreasing && isRewardImproving) {
        diagBox.className = "diagnostic-box success-box";
        titleEl.innerText = "Training Improvements Translating into Performance";
        descEl.innerText = "Both neural network optimization (decreasing loss) and environmental behavior (increasing reward) are improving in tandem.";
    }
}

function renderLossChart(episodes, rawLoss, maLoss) {
    const ctx = document.getElementById("chart-loss").getContext("2d");
    if (chartLoss) chartLoss.destroy();

    chartLoss = new Chart(ctx, {
        type: "line",
        data: {
            labels: episodes,
            datasets: [
                {
                    label: "Raw Loss",
                    data: rawLoss,
                    borderColor: "rgba(251, 146, 60, 0.35)",
                    borderWidth: 1,
                    pointRadius: 0,
                    fill: false
                },
                {
                    label: `${maWindow}-Ep Moving Avg Loss`,
                    data: maLoss,
                    borderColor: "#fb923c",
                    borderWidth: 2.5,
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: "Episode" } },
                y: { title: { display: true, text: "VDN Loss" }, beginAtZero: true }
            }
        }
    });
}

function renderRewardChart(episodes, rawReward, maReward) {
    const ctx = document.getElementById("chart-reward").getContext("2d");
    if (chartReward) chartReward.destroy();

    chartReward = new Chart(ctx, {
        type: "line",
        data: {
            labels: episodes,
            datasets: [
                {
                    label: "Raw Return",
                    data: rawReward,
                    borderColor: "rgba(56, 189, 248, 0.35)",
                    borderWidth: 1,
                    pointRadius: 0,
                    fill: false
                },
                {
                    label: `${maWindow}-Ep Moving Avg Reward`,
                    data: maReward,
                    borderColor: "#38bdf8",
                    borderWidth: 2.5,
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: "Episode" } },
                y: { title: { display: true, text: "Total Team Return" } }
            }
        }
    });
}

function renderSuccessRateChart(episodes, rawSuccess, maSuccess) {
    const ctx = document.getElementById("chart-success-rate").getContext("2d");
    if (chartSuccessRate) chartSuccessRate.destroy();

    chartSuccessRate = new Chart(ctx, {
        type: "line",
        data: {
            labels: episodes,
            datasets: [
                {
                    label: "Goal Reached (0 / 100%)",
                    data: rawSuccess,
                    borderColor: "rgba(74, 222, 128, 0.25)",
                    borderWidth: 1,
                    pointRadius: 1,
                    fill: false
                },
                {
                    label: `${maWindow}-Ep Success Rate (%)`,
                    data: maSuccess,
                    borderColor: "#4ade80",
                    borderWidth: 2.5,
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: "Episode" } },
                y: { title: { display: true, text: "Success Rate (%)" }, min: 0, max: 100 }
            }
        }
    });
}

function renderEvaluationSectionCharts(history, evalHistory, hasGoal) {
    // Chart 4: Eval Success Rate
    const ctx4 = document.getElementById("chart-eval-success").getContext("2d");
    if (chartEvalSuccess) chartEvalSuccess.destroy();
    chartEvalSuccess = new Chart(ctx4, {
        type: "bar",
        data: {
            labels: evalHistory.length > 0 ? evalHistory.map(e => `Ep ${e.episode}`) : ["Eval 1", "Eval 2"],
            datasets: [{
                label: "Eval Success Rate (%)",
                data: evalHistory.length > 0 ? evalHistory.map(e => e.success_rate || 0) : [0, 0],
                backgroundColor: "#4ade80"
            }]
        },
        options: { responsive: true, maintainAspectRatio: false, scales: { y: { min: 0, max: 100 } } }
    });

    // Chart 5: Steps to goal
    const ctx5 = document.getElementById("chart-eval-steps").getContext("2d");
    if (chartEvalSteps) chartEvalSteps.destroy();
    const successful = history.filter(h => h.goal_reached === true);
    chartEvalSteps = new Chart(ctx5, {
        type: "line",
        data: {
            labels: successful.map(h => `Ep ${h.episode}`),
            datasets: [{
                label: "Steps to Goal",
                data: successful.map(h => h.steps),
                borderColor: "#38bdf8",
                backgroundColor: "rgba(56, 189, 248, 0.2)",
                fill: true
            }]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });

    // Chart 6: Eval Reward
    const ctx6 = document.getElementById("chart-eval-reward").getContext("2d");
    if (chartEvalReward) chartEvalReward.destroy();
    chartEvalReward = new Chart(ctx6, {
        type: "line",
        data: {
            labels: evalHistory.map(e => `Ep ${e.episode}`),
            datasets: [{
                label: "Eval Reward (ε=0)",
                data: evalHistory.map(e => e.evaluation_reward || 0),
                borderColor: "#c084fc",
                fill: false
            }]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });

    // Chart 7: Success vs Failure vs Timeout Breakdown
    const ctx7 = document.getElementById("chart-eval-breakdown").getContext("2d");
    if (chartEvalBreakdown) chartEvalBreakdown.destroy();
    const succCount = history.filter(h => h.goal_reached === true).length;
    const timeoutCount = history.filter(h => h.timeout === true || h.goal_reached === false).length;
    const unknownCount = !hasGoal ? history.length : 0;

    chartEvalBreakdown = new Chart(ctx7, {
        type: "doughnut",
        data: {
            labels: hasGoal ? ["Success (Goal Reached)", "Timeout (Max Steps)"] : ["Goal Unlogged"],
            datasets: [{
                data: hasGoal ? [succCount, timeoutCount] : [unknownCount],
                backgroundColor: hasGoal ? ["#4ade80", "#f87171"] : ["#334155"]
            }]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });
}

function renderAllNodesDashboard(allNodes) {
    const tbody = document.getElementById("node-comparison-tbody");
    tbody.innerHTML = "";

    const n1 = allNodes["N1"] || {};
    const n2 = allNodes["N2"] || {};
    const n3 = allNodes["N3"] || {};

    const nodesArr = [
        { id: "N1", variant: "Checkpoints / Simple", data: n1 },
        { id: "N2", variant: "Obstacles & Walls", data: n2 },
        { id: "N3", variant: "Key & Door Pair", data: n3 }
    ];

    nodesArr.forEach(item => {
        const hist = item.data.training_history || [];
        const latest = hist.length > 0 ? hist[hist.length - 1] : {};
        const lossStr = latest.loss !== undefined && latest.loss !== null ? latest.loss.toFixed(4) : "N/A";
        const rews = hist.map(h => h.reward);
        const avgRew = rews.length > 0 ? (rews.slice(-10).reduce((a,b)=>a+b,0) / Math.min(10, rews.length)).toFixed(3) : "N/A";

        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td><strong>${item.id}</strong></td>
            <td>${item.variant}</td>
            <td>${lossStr}</td>
            <td>${avgRew}</td>
            <td>${item.data.has_goal_logged ? "Active" : "N/A — Not Logged"}</td>
            <td>N/A</td>
            <td>N/A</td>
            <td>100.0%</td>
        `;
        tbody.appendChild(tr);
    });

    // Comparative Multi-series Charts
    const ctxCompRew = document.getElementById("chart-compare-rewards").getContext("2d");
    if (chartCompareRewards) chartCompareRewards.destroy();
    chartCompareRewards = new Chart(ctxCompRew, {
        type: "line",
        data: {
            labels: (n1.training_history || []).map(h => h.episode),
            datasets: [
                { label: "N1 (Simple)", data: (n1.training_history || []).map(h => h.reward), borderColor: "#38bdf8" },
                { label: "N2 (Obstacles)", data: (n2.training_history || []).map(h => h.reward), borderColor: "#fb923c" },
                { label: "N3 (Key/Door)", data: (n3.training_history || []).map(h => h.reward), borderColor: "#c084fc" }
            ]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });

    const ctxCompLoss = document.getElementById("chart-compare-loss").getContext("2d");
    if (chartCompareLoss) chartCompareLoss.destroy();
    chartCompareLoss = new Chart(ctxCompLoss, {
        type: "line",
        data: {
            labels: (n1.training_history || []).map(h => h.episode),
            datasets: [
                { label: "N1 Loss", data: (n1.training_history || []).map(h => h.loss), borderColor: "#38bdf8" },
                { label: "N2 Loss", data: (n2.training_history || []).map(h => h.loss), borderColor: "#fb923c" },
                { label: "N3 Loss", data: (n3.training_history || []).map(h => h.loss), borderColor: "#c084fc" }
            ]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });
}
