/**
 * app.js — PneumoScan AI shared utilities
 */

// History badge: always reflects true count from localStorage
function syncHistoryBadge() {
  const hist  = JSON.parse(localStorage.getItem('prediction_history') || '[]');
  const badge = document.getElementById('hist-badge');
  if (!badge) return;
  badge.textContent = hist.length;
  badge.classList.toggle('hidden', hist.length === 0);
}

// Alias kept for any old references
const updateHistoryBadge = syncHistoryBadge;

// Format bytes helper
function formatBytes(bytes) {
  if (bytes < 1024)      return bytes + ' B';
  if (bytes < 1048576)   return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}
