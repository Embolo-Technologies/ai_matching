#!/usr/bin/env python3
import csv
import json
import os
import sys

# Define directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_HTML = os.path.join(SCRIPT_DIR, "search_app.html")

# Candidate paths for the CSV
CSV_PATHS = [
    "/Users/admin/Downloads/item_export_2026-05-30_09-08-22.csv",  # Prefer newer 42MB version
    "/Users/admin/Downloads/Item_export_2026-05-29_17-33-30.csv"
]

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chemist Product Search & Match</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --container-bg: rgba(17, 24, 39, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
            --accent-cyan: #06b6d4;
            --accent-purple: #a855f7;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(at 0% 0%, rgba(6, 182, 212, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(168, 85, 247, 0.15) 0px, transparent 50%);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            width: 100%;
            max-width: 900px;
            background: var(--container-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 24px;
            padding: 35px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
            animation: fadeIn 0.6s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        header {
            margin-bottom: 30px;
            text-align: center;
        }

        h1 {
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            background: linear-gradient(to right, var(--accent-cyan), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }

        header p {
            color: var(--text-secondary);
            font-size: 0.95rem;
        }

        .search-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 25px;
        }

        @media (max-width: 600px) {
            .search-grid {
                grid-template-columns: 1fr;
            }
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .input-group label {
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .input-wrapper {
            position: relative;
        }

        input {
            width: 100%;
            padding: 14px 16px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 1rem;
            transition: all 0.3s ease;
        }

        input:focus {
            outline: none;
            border-color: var(--accent-cyan);
            background: rgba(255, 255, 255, 0.06);
            box-shadow: 0 0 15px rgba(6, 182, 212, 0.2);
        }

        /* Match Result Card Styles */
        .match-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 25px;
            transition: all 0.3s ease;
        }

        .match-card.success {
            background: rgba(16, 185, 129, 0.06);
            border-color: rgba(16, 185, 129, 0.3);
            box-shadow: 0 0 15px rgba(16, 185, 129, 0.1);
        }

        .match-card.fail {
            background: rgba(239, 68, 68, 0.04);
            border-color: rgba(239, 68, 68, 0.2);
        }

        .match-card-header {
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            margin-bottom: 10px;
        }

        .match-card.success .match-card-header {
            color: var(--accent-green);
        }

        .match-card.fail .match-card-header {
            color: var(--accent-red);
        }

        .match-title {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 6px;
        }

        .match-meta {
            font-size: 0.9rem;
            color: var(--text-secondary);
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }

        .match-meta span {
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .stats {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 15px;
            padding: 0 5px;
        }

        .results-container {
            border: 1px solid var(--border-color);
            border-radius: 16px;
            background: rgba(0, 0, 0, 0.2);
            overflow: hidden;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.95rem;
        }

        th {
            background: rgba(255, 255, 255, 0.02);
            padding: 14px 16px;
            font-weight: 600;
            color: var(--text-secondary);
            border-bottom: 1px solid var(--border-color);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        td {
            padding: 14px 16px;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-primary);
        }

        tr:last-child td {
            border-bottom: none;
        }

        tr {
            transition: background-color 0.2s ease;
        }

        tr.has-match:hover {
            background: rgba(255, 255, 255, 0.02);
        }

        .code-pill {
            background: rgba(6, 182, 212, 0.1);
            color: var(--accent-cyan);
            padding: 4px 8px;
            border-radius: 6px;
            font-family: monospace;
            font-size: 0.85rem;
            border: 1px solid rgba(6, 182, 212, 0.2);
        }

        .badge-green {
            background: rgba(16, 185, 129, 0.1);
            color: var(--accent-green);
            border: 1px solid rgba(16, 185, 129, 0.2);
            padding: 3px 6px;
            border-radius: 4px;
            font-size: 0.8rem;
        }

        .no-results {
            padding: 30px;
            text-align: center;
            color: var(--text-secondary);
        }

        /* Scrollbar styles */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.1);
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.2);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Chemist Product Search & Match</h1>
            <p id="catalog-status">Loading offline catalog...</p>
        </header>

        <div class="search-grid">
            <div class="input-group">
                <label for="name-input">Medicine Name</label>
                <div class="input-wrapper">
                    <input type="text" id="name-input" placeholder="e.g. Pantica injection, Calpol, Revital" autocomplete="off">
                </div>
            </div>
            <div class="input-group">
                <label for="pack-input">Pack Size (Optional)</label>
                <div class="input-wrapper">
                    <input type="text" id="pack-input" placeholder="e.g. 10 TAB, 15 CAP, 1*10ML" autocomplete="off">
                </div>
            </div>
        </div>

        <!-- Confirm Match Output Card -->
        <div class="match-card" id="match-card-container">
            <div class="match-card-header" id="match-card-header">Match Result</div>
            <div id="match-card-content">
                <div style="color: var(--text-secondary); font-size: 0.95rem;">Type a medicine name to check for a confirmed match.</div>
            </div>
        </div>

        <div class="stats">
            <div id="results-count">0 items found</div>
            <div id="latency">Search time: 0ms</div>
        </div>

        <div class="results-container">
            <table id="results-table">
                <thead>
                    <tr>
                        <th style="width: 12%;">Code</th>
                        <th style="width: 48%;">Name</th>
                        <th style="width: 25%;">Company</th>
                        <th style="width: 15%;">Pack</th>
                    </tr>
                </thead>
                <tbody id="results-body">
                    <tr>
                        <td colspan="4" class="no-results">Type a medicine name to begin searching.</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>

    <script>
        // Embedded Unified Catalog Database
        // Columns: [code, name, brand/compname, pack, strength]
        const CATALOG = %CATALOG_JSON%;

        // Initialize UI Elements
        document.getElementById('catalog-status').textContent = `Local Catalog Loaded: ${CATALOG.length.toLocaleString()} medicines available offline.`;

        const nameInput = document.getElementById('name-input');
        const packInput = document.getElementById('pack-input');
        
        const matchCardContainer = document.getElementById('match-card-container');
        const matchCardHeader = document.getElementById('match-card-header');
        const matchCardContent = document.getElementById('match-card-content');
        
        const resultsBody = document.getElementById('results-body');
        const resultsCount = document.getElementById('results-count');
        const latencyLabel = document.getElementById('latency');

        // Fast string normalization
        function normalizeText(text) {
            if (!text) return "";
            text = text.toLowerCase()
                       .replace(/-/g, " ")
                       .replace(/\\//g, " ");
            
            // Remove common formulations
            text = text.replace(/\\b(tabs?|tablets?|inj(ection)?s?|caps?(ules?)?|susp(ension)?|syp|syrup|drops?|cream|ointment|gel|liq(uid)?|sol(ution)?)\\b/g, "");
            text = text.replace(/[^a-z0-9\\s]/g, "");
            return text.split(/\\s+/).filter(Boolean).join(" ");
        }

        // Tokenize pharmaceutical names including squashed unit blobs (e.g. "PANTOP40")
        function meiliTokenize(text) {
            const rawTokens = normalizeText(text).split(" ");
            const finalTokens = [];
            
            for (const tok of rawTokens) {
                // Split number from unit (e.g. "5mg" -> ["5", "mg"])
                const parts = tok.match(/([a-z]+|[0-9]+)/g);
                if (parts) {
                    for (const p of parts) {
                        let cleaned = p;
                        // Normalize common units
                        if (cleaned === "cc" || cleaned === "grm") cleaned = "ml";
                        if (cleaned === "g") cleaned = "gm";
                        finalTokens.push(cleaned);
                    }
                }
            }
            return finalTokens;
        }

        // Fast Dice Coefficient for word similarity
        function diceCoefficient(str1, str2) {
            if (str1 === str2) return 1.0;
            if (str1.length < 2 || str2.length < 2) return 0.0;

            const makeBigrams = str => {
                const bigrams = new Set();
                for (let i = 0; i < str.length - 1; i++) {
                    bigrams.add(str.substr(i, 2));
                }
                return bigrams;
            };

            const bigrams1 = makeBigrams(str1);
            const bigrams2 = makeBigrams(str2);
            let intersection = 0;

            for (const bigram of bigrams1) {
                if (bigrams2.has(bigram)) intersection++;
            }

            return (2.0 * intersection) / (bigrams1.size + bigrams2.size);
        }

        // Strict guardrail validation (matches validate_match in matcher.py)
        function validateMatch(queryName, queryPack, candidate) {
            const queryCombined = (queryName + " " + queryPack).trim();
            const qClean = normalizeText(queryCombined);
            const cNameClean = normalizeText(candidate[1]);
            const cBrandClean = normalizeText(candidate[2]);
            
            const qWords = qClean.split(" ");
            if (qWords.length === 0) return false;
            const qBrand = qWords[0];
            
            // 1. Brand similarity check
            const qBrandFlat = qBrand.replace(/ /g, "");
            const cNameFlat = cNameClean.replace(/ /g, "");
            const cBrandFlat = cBrandClean.replace(/ /g, "");
            
            let brandMatch = (
                cNameClean.includes(qBrand) ||
                cBrandClean.includes(qBrand) ||
                cNameFlat.includes(qBrandFlat) ||
                cBrandFlat.includes(qBrandFlat) ||
                qBrandFlat.includes(cNameFlat) ||
                qBrandFlat.includes(cBrandFlat) ||
                diceCoefficient(qBrand, cNameClean) >= 0.75
            );
            if (!brandMatch) return false;
            
            // 2. Strength match check
            // Extract numbers from the query name (avoiding packaging size like 10TAB)
            const qNums = (normalizeText(queryName).match(/\\d+/g) || []);
            const cNums = ((candidate[1] + " " + candidate[4]).match(/\\d+/g) || []);
            
            if (qNums.length > 0) {
                let matchedNum = false;
                for (const qNum of qNums) {
                    if (['1', '2', '3', '4', '5', '6', '8', '10'].includes(qNum) && qNum.length === 1) {
                        continue;
                    }
                    if (cNums.includes(qNum)) {
                        matchedNum = true;
                        break;
                    }
                }
                if (cNums.length === 0) return false;
                if (!matchedNum) return false;
            }
            
            // 3. Formulation match check (tablet vs injection)
            const queryLower = queryCombined.toLowerCase();
            const candNameLower = candidate[1].toLowerCase();
            
            const queryIsInj = ['inj', 'injection', 'injections'].some(w => queryLower.includes(w));
            const candIsTab = ['tab', 'tabs', 'tablet', 'tablets'].some(w => candNameLower.includes(w));
            if (queryIsInj && candIsTab) return false;
            
            const queryIsTab = ['tab', 'tabs', 'tablet', 'tablets'].some(w => queryLower.includes(w));
            const candIsInj = ['inj', 'injection', 'injections'].some(w => candNameLower.includes(w));
            if (queryIsTab && candIsInj) return false;
            
            return true;
        }

        // Strict Heuristics (matches heuristic_match in matcher.py)
        function heuristicMatch(queryName, queryPack, candidates) {
            if (candidates.length === 0) return null;
            
            const bestCand = candidates[0];
            const candidate = bestCand.item;
            
            const qClean = normalizeText(queryName);
            const cNameClean = normalizeText(candidate[1]);
            const cBrandClean = normalizeText(candidate[2]);
            
            // Extract numbers from query name
            const qNums = (normalizeText(queryName).match(/\\d+/g) || []);
            const cNums = ((candidate[1] + " " + candidate[4]).match(/\\d+/g) || []);
            
            // 1. Strict Brand Match
            const qWords = qClean.split(" ");
            if (qWords.length === 0) return null;
            const qBrand = qWords[0];
            
            const qBrandFlat = qBrand.replace(/ /g, "");
            const cNameFlat = cNameClean.replace(/ /g, "");
            const cBrandFlat = cBrandClean.replace(/ /g, "");
            
            let brandExact = (
                qBrandFlat === cNameFlat ||
                qBrandFlat === cBrandFlat ||
                cNameFlat.includes(qBrandFlat) ||
                cBrandFlat.includes(qBrandFlat) ||
                diceCoefficient(qBrand, cNameClean) >= 0.78
            );
            
            // 2. Strict Strength Match
            let numsMatch = false;
            if (qNums.length > 0) {
                const qStrengths = qNums.filter(n => n.length > 1 || !['1', '2', '3', '4', '5', '6', '8', '10'].includes(n));
                const cStrengths = cNums.filter(n => n.length > 1 || !['1', '2', '3', '4', '5', '6', '8', '10'].includes(n));
                
                if (qStrengths.length > 0) {
                    numsMatch = qStrengths.every(n => cStrengths.includes(n));
                } else {
                    numsMatch = (cStrengths.length === 0);
                }
            } else {
                numsMatch = (cNums.filter(n => n.length > 1).length === 0);
            }
            
            // 3. Strict Formulation Match
            let formulationMatch = true;
            const queryLower = (queryName + " " + queryPack).toLowerCase();
            const candNameLower = candidate[1].toLowerCase();
            
            const queryIsInj = ['inj', 'injection', 'injections'].some(w => queryLower.includes(w));
            const candIsTab = ['tab', 'tabs', 'tablet', 'tablets'].some(w => candNameLower.includes(w));
            if (queryIsInj && candIsTab) formulationMatch = false;
            
            const queryIsTab = ['tab', 'tabs', 'tablet', 'tablets'].some(w => queryLower.includes(w));
            const candIsInj = ['inj', 'injection', 'injections'].some(w => candNameLower.includes(w));
            if (queryIsTab && candIsInj) formulationMatch = false;
            
            if (brandExact && numsMatch && formulationMatch) {
                if (validateMatch(queryName, queryPack, candidate)) {
                    return candidate;
                }
            }
            return null;
        }

        // Fuzzy Scorer logic matching searcher.py
        function calculateScore(queryTokens, queryPack, item) {
            const itemName = item[1];
            const itemBrand = item[2];
            const itemPack = item[3];
            
            const itemTokens = meiliTokenize(itemName);
            
            // Calculate matching token count & exactness
            let matchedCount = 0;
            let exactCount = 0;
            let prefixCount = 0;
            
            for (const qTok of queryTokens) {
                let found = false;
                for (const iTok of itemTokens) {
                    if (qTok === iTok) {
                        exactCount++;
                        matchedCount++;
                        found = true;
                        break;
                    } else if (iTok.startsWith(qTok)) {
                        prefixCount++;
                        matchedCount++;
                        found = true;
                        break;
                    }
                }
                if (!found && qTok.length >= 3) {
                    for (const iTok of itemTokens) {
                        if (iTok.length >= 3 && diceCoefficient(qTok, iTok) >= 0.7) {
                            matchedCount += 0.8;
                            break;
                        }
                    }
                }
            }
            
            // Front Consecutive matches
            let frontMatch = 0;
            const limit = Math.min(queryTokens.length, itemTokens.length);
            for (let i = 0; i < limit; i++) {
                if (queryTokens[i] === itemTokens[i]) {
                    frontMatch++;
                } else {
                    break;
                }
            }
            
            // Pack match score
            let packScore = 50;
            if (queryPack && itemPack) {
                const qPackNorm = normalizeText(queryPack);
                const iPackNorm = normalizeText(itemPack);
                if (qPackNorm === iPackNorm) {
                    packScore = 100;
                } else if (iPackNorm.includes(qPackNorm) || qPackNorm.includes(iPackNorm)) {
                    packScore = 80;
                }
            }
            
            const matchingRatio = queryTokens.length > 0 ? (matchedCount / queryTokens.length) : 0;
            const exactRatio = queryTokens.length > 0 ? (exactCount / queryTokens.length) : 0;
            
            const nameScore = (matchingRatio * 60) + (exactRatio * 20) + (frontMatch * 20);
            const finalScore = (nameScore * 0.75) + (packScore * 0.25);
            
            return {
                finalScore: finalScore,
                matchedCount: matchedCount,
                frontMatch: frontMatch
            };
        }

        // Main search and match execution
        function doSearchAndMatch() {
            const queryName = nameInput.value.trim();
            const queryPack = packInput.value.trim();
            
            if (queryName.length < 2) {
                resultsBody.innerHTML = `<tr><td colspan="4" class="no-results">Type at least 2 characters to search.</td></tr>`;
                resultsCount.textContent = "0 items found";
                
                // Clear match card
                matchCardContainer.className = "match-card";
                matchCardHeader.textContent = "Match Result";
                matchCardContent.innerHTML = `<div style="color: var(--text-secondary); font-size: 0.95rem;">Type a medicine name to check for a confirmed match.</div>`;
                return;
            }

            const t0 = performance.now();
            const queryTokens = meiliTokenize(queryName);
            const matches = [];
            
            for (let i = 0; i < CATALOG.length; i++) {
                const item = CATALOG[i];
                const itemNameLower = item[1].toLowerCase();
                const firstQueryWord = queryTokens[0];
                
                if (itemNameLower.indexOf(firstQueryWord) !== -1 || (item[2] && item[2].toLowerCase().indexOf(firstQueryWord) !== -1)) {
                    const scoreData = calculateScore(queryTokens, queryPack, item);
                    if (scoreData.finalScore >= 35) {
                        matches.push({
                            item: item,
                            score: scoreData.finalScore,
                            matchedCount: scoreData.matchedCount,
                            frontMatch: scoreData.frontMatch
                        });
                    }
                }
            }
            
            // Sort matches
            matches.sort((a, b) => {
                if (b.matchedCount !== a.matchedCount) {
                    return b.matchedCount - a.matchedCount;
                }
                if (b.frontMatch !== a.frontMatch) {
                    return b.frontMatch - a.frontMatch;
                }
                return b.score - a.score;
            });
            
            // Execute strict matching heuristics (Phase 2)
            const matchedProduct = heuristicMatch(queryName, queryPack, matches);
            
            // Update Confirm Match Status Card UI
            if (matchedProduct) {
                matchCardContainer.className = "match-card success";
                matchCardHeader.textContent = "Confirmed Match Found";
                matchCardContent.innerHTML = `
                    <div class="match-title">${matchedProduct[1]}</div>
                    <div class="match-meta">
                        <span><strong>Code:</strong> <span class="code-pill">${matchedProduct[0]}</span></span>
                        <span><strong>Company:</strong> ${matchedProduct[2] || 'GENERAL'}</span>
                        <span><strong>Pack:</strong> <span class="badge-green">${matchedProduct[3] || 'GENERAL'}</span></span>
                    </div>
                `;
            } else {
                matchCardContainer.className = "match-card fail";
                matchCardHeader.textContent = "No Confirmed Match";
                matchCardContent.innerHTML = `
                    <div style="font-size: 0.95rem; font-weight: 500; color: var(--accent-red); margin-bottom: 4px;">
                        No product matches the strict brand, formulation, or strength rules.
                    </div>
                    <div style="color: var(--text-secondary); font-size: 0.85rem;">
                        Inspect the fuzzy candidates list below for options.
                    </div>
                `;
            }
            
            const t1 = performance.now();
            latencyLabel.textContent = `Search time: ${Math.round(t1 - t0)}ms`;
            
            // Render ranked list below
            const topResults = matches.slice(0, 15);
            resultsCount.textContent = `${matches.length.toLocaleString()} items found`;
            
            if (topResults.length === 0) {
                resultsBody.innerHTML = `<tr><td colspan="4" class="no-results">No matching medicines found. Check spelling or try fewer words.</td></tr>`;
                return;
            }
            
            resultsBody.innerHTML = topResults.map(r => `
                <tr class="has-match">
                    <td><span class="code-pill">${r.item[0]}</span></td>
                    <td style="font-weight: 500;">${r.item[1]}</td>
                    <td style="color: var(--text-secondary);">${r.item[2] || 'GENERAL'}</td>
                    <td><span class="badge-green">${r.item[3] || 'GENERAL'}</span></td>
                </tr>
            `).join('');
        }

        // Attach event listeners for instant typing search
        nameInput.addEventListener('input', doSearchAndMatch);
        packInput.addEventListener('input', doSearchAndMatch);
    </script>
</body>
</html>
"""

def main():
    print()
    print("  === Bundling Offline Search & Match App ===")
    
    csv_path = None
    for p in CSV_PATHS:
        if os.path.exists(p):
            csv_path = p
            break
            
    if not csv_path:
        print(f"  [Error] Unified CSV database not found.")
        sys.exit(1)
        
    print(f"  Reading CSV catalog from: {csv_path}")
    catalog_list = []
    
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            code = r.get("code", "").strip()
            name = r.get("name", "").strip()
            compname = r.get("compname", "").strip()
            pack = r.get("pack", "").strip()
            strength = r.get("strength", "").strip()
            
            if code and name:
                catalog_list.append([code, name, compname, pack, strength])
                
    print(f"  Extracted {len(catalog_list):,} product records.")
    
    # Dump JSON array
    catalog_json = json.dumps(catalog_list)
    print(f"  Compressed JSON database size: {len(catalog_json) / (1024*1024):.2f} MB")
    
    # Inject database into template
    html_content = HTML_TEMPLATE.replace("%CATALOG_JSON%", catalog_json)
    
    print(f"  Writing standalone app to: {OUTPUT_HTML}")
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print("  ✓ App successfully compiled!")
    print(f"  You can now copy search_app.html to any PC/SSD and double-click to run!")
    print()

if __name__ == "__main__":
    main()
