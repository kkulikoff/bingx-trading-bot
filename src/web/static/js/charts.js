/**
 * Charts JavaScript for BingX Trading Bot
 * Handles creation and updating of various charts used in the application
 */

// Chart colors
const CHART_COLORS = {
    primary: '#3498db',
    success: '#27ae60',
    danger: '#e74c3c',
    warning: '#f39c12',
    purple: '#9b59b6',
    asphalt: '#34495e',
    clouds: '#ecf0f1'
};

// Chart configuration
const CHART_CONFIG = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            labels: {
                color: CHART_COLORS.clouds
            }
        },
        tooltip: {
            mode: 'index',
            intersect: false,
            backgroundColor: 'rgba(44, 62, 80, 0.9)',
            titleColor: CHART_COLORS.clouds,
            bodyColor: CHART_COLORS.clouds,
            borderColor: CHART_COLORS.primary,
            borderWidth: 1
        }
    },
    scales: {
        x: {
            grid: {
                color: 'rgba(255, 255, 255, 0.1)'
            },
            ticks: {
                color: CHART_COLORS.clouds
            }
        },
        y: {
            grid: {
                color: 'rgba(255, 255, 255, 0.1)'
            },
            ticks: {
                color: CHART_COLORS.clouds
            }
        }
    }
};

// Create equity curve chart
function createEquityChart(canvasId, equityData) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Prepare data
    const labels = equityData.map(point => {
        return point.time ? new Date(point.time).toLocaleDateString() : '';
    });
    
    const data = equityData.map(point => point.balance);
    
    // Create chart
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Equity Curve',
                data: data,
                borderColor: CHART_COLORS.primary,
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                fill: true,
                tension: 0.1,
                pointBackgroundColor: CHART_COLORS.primary,
                pointBorderColor: CHART_COLORS.clouds,
                pointHoverBackgroundColor: CHART_COLORS.clouds,
                pointHoverBorderColor: CHART_COLORS.primary,
                pointRadius: 3,
                pointHoverRadius: 5
            }]
        },
        options: {
            ...CHART_CONFIG,
            plugins: {
                ...CHART_CONFIG.plugins,
                legend: {
                    display: false
                }
            },
            scales: {
                ...CHART_CONFIG.scales,
                y: {
                    ...CHART_CONFIG.scales.y,
                    ticks: {
                        ...CHART_CONFIG.scales.y.ticks,
                        callback: function(value) {
                            return '$' + value.toLocaleString();
                        }
                    }
                }
            }
        }
    });
}

// Update equity chart
function updateEquityChart(equityData) {
    if (!window.equityChart) {
        window.equityChart = createEquityChart('equity-chart', equityData);
        return;
    }
    
    const labels = equityData.map(point => {
        return point.time ? new Date(point.time).toLocaleDateString() : '';
    });
    
    const data = equityData.map(point => point.balance);
    
    window.equityChart.data.labels = labels;
    window.equityChart.data.datasets[0].data = data;
    window.equityChart.update();
}

// Create allocation chart
function createAllocationChart(canvasId, assets) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Calculate total value and prepare data
    const totalValue = assets.reduce((sum, asset) => sum + (asset.total || 0), 0);
    
    const labels = [];
    const data = [];
    const backgroundColors = [];
    
    assets.forEach((asset, index) => {
        if (asset.total > 0) {
            const value = asset.total;
            const allocation = (value / totalValue * 100);
            
            if (allocation >= 1) { // Only show assets with >1% allocation
                labels.push(asset.asset);
                data.push(allocation);
                backgroundColors.push(getColorByIndex(index));
            }
        }
    });
    
    // Create chart
    return new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: backgroundColors,
                borderColor: CHART_COLORS.asphalt,
                borderWidth: 2
            }]
        },
        options: {
            ...CHART_CONFIG,
            cutout: '60%',
            plugins: {
                ...CHART_CONFIG.plugins,
                tooltip: {
                    ...CHART_CONFIG.plugins.tooltip,
                    callbacks: {
                        label: function(context) {
                            return context.label + ': ' + context.raw.toFixed(2) + '%';
                        }
                    }
                }
            }
        }
    });
}

// Update allocation chart
function updateAllocationChart(assets) {
    if (!window.allocationChart) {
        window.allocationChart = createAllocationChart('allocation-chart', assets);
        return;
    }
    
    // Calculate total value and prepare data
    const totalValue = assets.reduce((sum, asset) => sum + (asset.total || 0), 0);
    
    const labels = [];
    const data = [];
    const backgroundColors = [];
    
    assets.forEach((asset, index) => {
        if (asset.total > 0) {
            const value = asset.total;
            const allocation = (value / totalValue * 100);
            
            if (allocation >= 1) {
                labels.push(asset.asset);
                data.push(allocation);
                backgroundColors.push(getColorByIndex(index));
            }
        }
    });
    
    window.allocationChart.data.labels = labels;
    window.allocationChart.data.datasets[0].data = data;
    window.allocationChart.data.datasets[0].backgroundColor = backgroundColors;
    window.allocationChart.update();
}

// Create performance chart
function createPerformanceChart(canvasId, performanceData) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Prepare data for different metrics
    const metrics = [
        {
            label: 'Win Rate',
            data: performanceData.win_rate || 0,
            color: CHART_COLORS.success
        },
        {
            label: 'Sharpe Ratio',
            data: performanceData.sharpe_ratio || 0,
            color: CHART_COLORS.primary
        },
        {
            label: 'Profit Factor',
            data: performanceData.profit_factor || 0,
            color: CHART_COLORS.purple
        },
        {
            label: 'Max Drawdown',
            data: performanceData.max_drawdown || 0,
            color: CHART_COLORS.danger
        }
    ];
    
    // Create chart
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: metrics.map(m => m.label),
            datasets: [{
                label: 'Performance Metrics',
                data: metrics.map(m => m.data),
                backgroundColor: metrics.map(m => m.color),
                borderColor: metrics.map(m => m.color),
                borderWidth: 1
            }]
        },
        options: {
            ...CHART_CONFIG,
            indexAxis: 'y',
            plugins: {
                ...CHART_CONFIG.plugins,
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    ...CHART_CONFIG.scales.x,
                    display: true
                },
                y: {
                    ...CHART_CONFIG.scales.y,
                    display: true
                }
            }
        }
    });
}

// Update performance chart
function updatePerformanceChart(performanceData) {
    if (!window.performanceChart) {
        window.performanceChart = createPerformanceChart('performance-chart', performanceData);
        return;
    }
    
    const metrics = [
        performanceData.win_rate || 0,
        performanceData.sharpe_ratio || 0,
        performanceData.profit_factor || 0,
        performanceData.max_drawdown || 0
    ];
    
    window.performanceChart.data.datasets[0].data = metrics;
    window.performanceChart.update();
}

// Create signal confidence chart
function createSignalChart(canvasId, signals) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Prepare data
    const recentSignals = signals.slice(-10); // Last 10 signals
    const labels = recentSignals.map(s => s.symbol);
    const data = recentSignals.map(s => s.confidence * 100);
    const backgroundColors = recentSignals.map(s => {
        return s.direction === 'LONG' ? CHART_COLORS.success : CHART_COLORS.danger;
    });
    
    // Create chart
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Signal Confidence (%)',
                data: data,
                backgroundColor: backgroundColors,
                borderColor: backgroundColors,
                borderWidth: 1
            }]
        },
        options: {
            ...CHART_CONFIG,
            plugins: {
                ...CHART_CONFIG.plugins,
                legend: {
                    display: false
                }
            },
            scales: {
                ...CHART_CONFIG.scales,
                y: {
                    ...CHART_CONFIG.scales.y,
                    min: 0,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            }
        }
    });
}

// Update signal chart
function updateSignalChart(signals) {
    if (!window.signalChart) {
        window.signalChart = createSignalChart('signal-chart', signals);
        return;
    }
    
    const recentSignals = signals.slice(-10);
    const labels = recentSignals.map(s => s.symbol);
    const data = recentSignals.map(s => s.confidence * 100);
    const backgroundColors = recentSignals.map(s => {
        return s.direction === 'LONG' ? CHART_COLORS.success : CHART_COLORS.danger;
    });
    
    window.signalChart.data.labels = labels;
    window.signalChart.data.datasets[0].data = data;
    window.signalChart.data.datasets[0].backgroundColor = backgroundColors;
    window.signalChart.update();
}

// Helper function to get a color by index
function getColorByIndex(index) {
    const colors = [
        CHART_COLORS.primary,
        CHART_COLORS.success,
        CHART_COLORS.danger,
        CHART_COLORS.warning,
        CHART_COLORS.purple,
        '#1abc9c', // Turquoise
        '#d35400', // Pumpkin
        '#c0392b', // Pomegranate
        '#16a085', // Green Sea
        '#8e44ad'  // Wisteria
    ];
    
    return colors[index % colors.length];
}

// Initialize all charts
function initCharts() {
    console.log('Initializing charts...');
    
    // Check if Chart.js is available
    if (typeof Chart === 'undefined') {
        console.error('Chart.js is not loaded');
        return;
    }
    
    // Register any custom chart types or plugins here
    
    console.log('Charts initialized');
}

// Export functions for global access
window.charts = {
    init: initCharts,
    createEquityChart: createEquityChart,
    updateEquityChart: updateEquityChart,
    createAllocationChart: createAllocationChart,
    updateAllocationChart: updateAllocationChart,
    createPerformanceChart: createPerformanceChart,
    updatePerformanceChart: updatePerformanceChart,
    createSignalChart: createSignalChart,
    updateSignalChart: updateSignalChart
};

// Initialize charts when document is ready
$(document).ready(function() {
    initCharts();
});