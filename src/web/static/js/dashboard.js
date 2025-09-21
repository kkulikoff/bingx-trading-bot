/**
 * Dashboard JavaScript for BingX Trading Bot
 * Handles real-time updates and interactions for the main dashboard
 */

// Global variables
let dashboardData = {
    signals: [],
    portfolio: {},
    performance: {},
    systemStatus: {}
};

let charts = {
    equity: null,
    allocation: null,
    performance: null
};

let updateInterval;

// Initialize dashboard
function initDashboard() {
    console.log('Initializing dashboard...');
    
    // Load initial data
    loadDashboardData();
    
    // Set up auto-refresh
    updateInterval = setInterval(loadDashboardData, 30000); // Update every 30 seconds
    
    // Set up event listeners
    $('#refresh-dashboard').on('click', loadDashboardData);
    $('#auto-refresh-toggle').on('change', toggleAutoRefresh);
    
    // Initialize tooltips
    $('[data-toggle="tooltip"]').tooltip();
    
    console.log('Dashboard initialized');
}

// Load all dashboard data
function loadDashboardData() {
    console.log('Loading dashboard data...');
    
    // Show loading state
    $('#dashboard-loading').show();
    $('#dashboard-content').hide();
    
    // Update last updated timestamp
    $('#last-updated').text('Updating...');
    
    // Load data from multiple endpoints in parallel
    Promise.all([
        $.get('/api/signals?limit=5'),
        $.get('/api/portfolio'),
        $.get('/api/performance'),
        $.get('/api/system/status')
    ]).then(function(responses) {
        // Process responses
        dashboardData.signals = responses[0].success ? responses[0].signals : [];
        dashboardData.portfolio = responses[1].success ? responses[1].portfolio : {};
        dashboardData.performance = responses[2].success ? responses[2].performance : {};
        dashboardData.systemStatus = responses[3].success ? responses[3].status : {};
        
        // Update UI with new data
        updateSignalsSection();
        updatePortfolioSection();
        updatePerformanceSection();
        updateSystemStatusSection();
        
        // Update charts
        updateCharts();
        
        // Hide loading state, show content
        $('#dashboard-loading').hide();
        $('#dashboard-content').show();
        
        // Update last updated timestamp
        $('#last-updated').text(new Date().toLocaleTimeString());
        
        console.log('Dashboard data updated');
    }).catch(function(error) {
        console.error('Error loading dashboard data:', error);
        $('#dashboard-loading').html('<div class="alert alert-danger">Error loading data. Please try again.</div>');
    });
}

// Update signals section
function updateSignalsSection() {
    const signalsContainer = $('#recent-signals');
    
    if (dashboardData.signals.length === 0) {
        signalsContainer.html('<div class="no-data">No recent signals</div>');
        return;
    }
    
    let signalsHtml = '';
    
    dashboardData.signals.forEach(signal => {
        const directionClass = signal.direction === 'LONG' ? 'signal-long' : 'signal-short';
        const directionIcon = signal.direction === 'LONG' ? '↗' : '↘';
        const confidencePercent = Math.round(signal.confidence * 100);
        
        signalsHtml += `
            <div class="signal-card ${directionClass}">
                <div class="signal-header">
                    <div class="signal-symbol">${signal.symbol}</div>
                    <div class="signal-direction">${directionIcon} ${signal.direction}</div>
                </div>
                <div class="signal-details">
                    <div class="signal-price">Price: $${signal.price.toFixed(2)}</div>
                    <div class="signal-confidence">
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
                        </div>
                        <span>${confidencePercent}%</span>
                    </div>
                </div>
                <div class="signal-meta">
                    <span class="signal-time">${new Date(signal.timestamp).toLocaleTimeString()}</span>
                    <span class="signal-tf">${signal.timeframe}</span>
                </div>
            </div>
        `;
    });
    
    signalsContainer.html(signalsHtml);
}

// Update portfolio section
function updatePortfolioSection() {
    const portfolio = dashboardData.portfolio;
    
    if (!portfolio.balance) {
        return;
    }
    
    // Update balance and metrics
    $('#total-balance').text(`$${portfolio.balance.toFixed(2)}`);
    $('#positions-count').text(portfolio.current_positions || 0);
    
    // Update performance metrics if available
    if (dashboardData.performance.backtest_results) {
        const results = dashboardData.performance.backtest_results;
        $('#total-return').text(`${results.total_return ? results.total_return.toFixed(2) : 0}%`);
        $('#win-rate').text(`${results.win_rate ? results.win_rate.toFixed(2) : 0}%`);
        $('#sharpe-ratio').text(results.sharpe_ratio ? results.sharpe_ratio.toFixed(2) : 'N/A');
    }
}

// Update performance section
function updatePerformanceSection() {
    const performance = dashboardData.performance;
    
    if (!performance.system_metrics) {
        return;
    }
    
    const metrics = performance.system_metrics;
    
    $('#uptime').text(formatUptime(metrics.uptime));
    $('#signals-generated').text(metrics.signals_generated || 0);
    $('#trades-executed').text(metrics.trades_executed || 0);
    $('#api-requests').text(metrics.api_requests || 0);
}

// Update system status section
function updateSystemStatusSection() {
    const status = dashboardData.systemStatus;
    
    if (!status.is_running !== undefined) {
        const statusElement = $('#system-status');
        const statusText = status.is_running ? 'Running' : 'Stopped';
        const statusClass = status.is_running ? 'status-running' : 'status-stopped';
        
        statusElement.text(statusText).removeClass('status-running status-stopped').addClass(statusClass);
    }
    
    if (status.start_time) {
        $('#start-time').text(new Date(status.start_time).toLocaleString());
    }
    
    if (status.uptime) {
        $('#system-uptime').text(formatUptime(status.uptime));
    }
}

// Update all charts
function updateCharts() {
    updateEquityChart();
    updateAllocationChart();
    updatePerformanceChart();
}

// Format uptime for display
function formatUptime(seconds) {
    if (!seconds) return '0s';
    
    const days = Math.floor(seconds / (3600 * 24));
    const hours = Math.floor((seconds % (3600 * 24)) / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    
    let result = '';
    if (days > 0) result += `${days}d `;
    if (hours > 0) result += `${hours}h `;
    if (mins > 0) result += `${mins}m`;
    
    return result || `${Math.floor(seconds)}s`;
}

// Toggle auto-refresh
function toggleAutoRefresh() {
    if ($('#auto-refresh-toggle').is(':checked')) {
        updateInterval = setInterval(loadDashboardData, 30000);
        console.log('Auto-refresh enabled');
    } else {
        clearInterval(updateInterval);
        console.log('Auto-refresh disabled');
    }
}

// Export functions for global access
window.dashboard = {
    init: initDashboard,
    loadData: loadDashboardData,
    toggleAutoRefresh: toggleAutoRefresh
};

// Initialize dashboard when document is ready
$(document).ready(function() {
    initDashboard();
});