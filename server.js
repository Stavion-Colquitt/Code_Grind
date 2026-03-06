/**
 * Code Grind — Unified Server
 * 
 * Everything in one process, two dependencies (express + dotenv):
 *   - Static frontend at /
 *   - In-memory BM25 search over knowledge JSON files
 *   - RAG-grounded AI companion endpoint
 *   - DeepSeek for code modes (review, explain, optimize)
 *   - Claude for interview prep mode
 * 
 * Deploy: npm install && node server.js
 */

require('dotenv').config();
const express = require('express');
const path = require('path');
const fs = require('fs');

const app = express();
app.use(express.json({ limit: '1mb' }));

app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.header('Access-Control-Allow-Headers', 'Content-Type');
    if (req.method === 'OPTIONS') return res.sendStatus(200);
    next();
});

app.use(express.static(path.join(__dirname, 'public')));

// Serve frontend at /codegrind when proxied through main server
app.get('/codegrind', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// ============================================================
// IN-MEMORY BM25 SEARCH ENGINE
// ============================================================

const KNOWLEDGE_DIR = path.join(__dirname, 'knowledge');

// Storage
const chunks = [];              // { id, filename, title, text, category }
const invertedIndex = Object.create(null);  // term -> [{ chunkIdx, tf }]
let avgDocLen = 0;

// Simple stemmer (suffix strip — good enough for search)
function stem(word) {
    if (word.length < 4) return word;
    return word
        .replace(/ies$/, 'y')
        .replace(/ied$/, 'y')
        .replace(/(s|es|ed|ing|tion|ment|ness|able|ible|ful|less|ous|ive|ly)$/, '')
        || word;
}

function tokenize(text) {
    return text.toLowerCase()
        .replace(/[^\w\s]/g, ' ')
        .split(/\s+/)
        .filter(w => w.length > 2)
        .map(stem);
}

function chunkDocument(content, filename) {
    const result = [];
    const lower = filename.toLowerCase();
    let category = 'general';
    if (lower.includes('challenge')) category = 'challenge';
    if (lower.includes('pattern')) category = 'pattern';
    if (lower.includes('dsp') || lower.includes('audio')) category = 'audio-dsp';
    if (lower.includes('spear')) category = 'grounding';

    if (Array.isArray(content)) {
        content.forEach((item, idx) => {
            const title = item.title || item.name || `Item ${idx + 1}`;
            const text = typeof item === 'string' ? item : JSON.stringify(item, null, 2);
            result.push({ id: `${filename}-${idx}`, filename, title: String(title).slice(0, 200), text, category });
        });
    } else if (content.sections) {
        content.sections.forEach((section, idx) => {
            result.push({
                id: `${filename}-section-${idx}`,
                filename,
                title: section.title || `Section ${idx + 1}`,
                text: section.content || JSON.stringify(section, null, 2),
                category
            });
        });
    } else if (typeof content === 'object') {
        for (const [key, value] of Object.entries(content)) {
            if (value && typeof value === 'object') {
                result.push({
                    id: `${filename}-${key}`,
                    filename,
                    title: `${filename.replace('.json', '')}: ${key}`,
                    text: JSON.stringify(value, null, 2),
                    category
                });
            }
        }
    }

    if (result.length === 0) {
        result.push({
            id: filename,
            filename,
            title: content.title || filename.replace('.json', ''),
            text: JSON.stringify(content, null, 2),
            category
        });
    }

    return result;
}

function buildIndex() {
    chunks.length = 0;
    for (const key of Object.keys(invertedIndex)) delete invertedIndex[key];
    avgDocLen = 0;

    const files = fs.readdirSync(KNOWLEDGE_DIR).filter(f => f.endsWith('.json'));
    console.log('[RAG] Indexing knowledge files...');

    for (const file of files) {
        try {
            const content = JSON.parse(fs.readFileSync(path.join(KNOWLEDGE_DIR, file), 'utf8'));
            const docs = chunkDocument(content, file);
            console.log(`  ${file}: ${docs.length} chunks`);
            for (const doc of docs) chunks.push(doc);
        } catch (e) {
            console.error(`  ${file}: ERROR — ${e.message}`);
        }
    }

    // Build inverted index with term frequencies
    let totalLen = 0;
    for (let i = 0; i < chunks.length; i++) {
        const terms = tokenize(chunks[i].title + ' ' + chunks[i].text);
        totalLen += terms.length;

        const freq = {};
        for (const t of terms) freq[t] = (freq[t] || 0) + 1;

        for (const [term, count] of Object.entries(freq)) {
            if (!invertedIndex[term]) invertedIndex[term] = [];
            invertedIndex[term].push({ idx: i, tf: count, docLen: terms.length });
        }
    }

    avgDocLen = chunks.length > 0 ? totalLen / chunks.length : 1;
    console.log(`[RAG] Indexed ${chunks.length} chunks, ${Object.keys(invertedIndex).length} unique terms\n`);
    return chunks.length;
}

// BM25 scoring
function search(query, limit = 8) {
    const terms = tokenize(query);
    if (!terms.length) return [];

    const N = chunks.length;
    const k1 = 1.5, b = 0.75;
    const scores = {};

    for (const term of terms) {
        const postings = invertedIndex[stem(term)] || invertedIndex[term] || [];
        if (!postings.length) continue;

        const df = postings.length;
        const idf = Math.log((N - df + 0.5) / (df + 0.5) + 1);

        for (const { idx, tf, docLen } of postings) {
            const tfNorm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * docLen / avgDocLen));
            scores[idx] = (scores[idx] || 0) + idf * tfNorm;
        }
    }

    return Object.entries(scores)
        .sort(([, a], [, b]) => b - a)
        .slice(0, limit)
        .map(([idx, score]) => ({ ...chunks[idx], score }));
}

function formatForPrompt(results, maxChars = 5000) {
    if (!results.length) return '';

    let output = '\n## RELEVANT KNOWLEDGE:\n';
    let chars = output.length;

    for (const r of results) {
        const entry = `\n### ${r.title}\n${r.text}\n`;
        if (chars + entry.length > maxChars) {
            const remaining = maxChars - chars - 50;
            if (remaining > 200) output += `\n### ${r.title}\n${r.text.slice(0, remaining)}...\n`;
            break;
        }
        output += entry;
        chars += entry.length;
    }

    return output;
}

function getRAGContext(title, category, mode) {
    const queries = [title];
    if (category) queries.push(category);
    if (category && category.toLowerCase().includes('audio')) queries.push('audio DSP real-time signal processing');
    if (mode === 'optimize') queries.push('complexity optimal time space');
    if (mode === 'interview') queries.push('interview behavioral STAR technical');

    const seen = new Set();
    const all = [];

    for (const q of queries) {
        for (const r of search(q, 5)) {
            if (!seen.has(r.id)) { seen.add(r.id); all.push(r); }
        }
    }

    all.sort((a, b) => b.score - a.score);
    return formatForPrompt(all.slice(0, 8));
}

// ============================================================
// SPEAR SYSTEM PROMPT
// ============================================================

const SPEAR_SYSTEM = `You are the Code Grind AI Companion — an embedded tutor inside a coding challenge app.

ABOUT THE USER:
- Developer with audio engineering background, self-taught programmer
- Preparing for technical interviews at top tech companies
- Learns through hands-on building
- Be direct, technical, concise, encouraging. Never condescending.

GROUNDING RULES (CRITICAL):
- ONLY reference challenges, solutions, concepts, patterns from the RELEVANT KNOWLEDGE section
- If knowledge doesn't contain info, say so — never invent details
- Never fabricate test cases, solutions, or complexity analysis
- The app has 64 challenges: Tier 1 (fundamentals), Tier 2 (intermediate), Tier 3 (advanced + audio DSP)

RESPONSE FORMAT:
- Under 300 words. Use code blocks for code. Bold key terms.
- Lead with most important finding, then explain, then next steps
- Never repeat the challenge prompt. No filler.

AUDIO DSP:
When reviewing audio challenges, connect code to real-world applications and interview relevance.

PATTERN RECOGNITION:
Always name the algorithm pattern. Connect across challenges.`;

const MODE_PROMPTS = {
    review: 'CODE REVIEW: Find bugs, logic errors, edge cases missed. Be specific. Reference the expected solution from knowledge. If correct, say so and suggest improvements.',
    explain: 'EXPLAIN LOGIC: Walk through step by step with a concrete example from the tests. Explain WHY each step works. Connect to the underlying pattern. Relate to audio concepts for DSP challenges.',
    optimize: 'OPTIMIZE: State time and space complexity. Compare to optimal from knowledge. For audio challenges, mention real-world constraints (latency, buffer, real-time).',
    interview: 'INTERVIEW PREP: Evaluate as a technical interviewer. Correctness, style, edge cases, complexity awareness. What would impress vs concern. Follow-up questions. Remember: plain text editor, no autocomplete, 35 min.'
};

// ============================================================
// LLM ROUTING — DeepSeek (code) + Claude (interview)
// ============================================================

async function callDeepSeek(systemPrompt, userMessage) {
    const key = process.env.DEEPSEEK_API_KEY;
    if (!key) throw new Error('DEEPSEEK_API_KEY not configured');

    const response = await fetch('https://api.deepseek.com/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${key}` },
        body: JSON.stringify({
            model: process.env.DEEPSEEK_MODEL || 'deepseek-chat',
            max_tokens: 1024,
            temperature: 0.3,
            messages: [
                { role: 'system', content: systemPrompt },
                { role: 'user', content: userMessage }
            ]
        })
    });

    if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.error?.message || `DeepSeek API error (${response.status})`);
    }

    const data = await response.json();
    return { text: data.choices[0]?.message?.content || '', model: data.model, usage: data.usage, provider: 'deepseek' };
}

async function callClaude(systemPrompt, userMessage) {
    const key = process.env.ANTHROPIC_API_KEY;
    if (!key) throw new Error('ANTHROPIC_API_KEY not configured');

    const response = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'x-api-key': key, 'anthropic-version': '2023-06-01' },
        body: JSON.stringify({
            model: process.env.CLAUDE_MODEL || 'claude-sonnet-4-20250514',
            max_tokens: 1024,
            system: systemPrompt,
            messages: [{ role: 'user', content: userMessage }]
        })
    });

    if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.error?.message || `Claude API error (${response.status})`);
    }

    const data = await response.json();
    return { text: data.content.map(b => b.text || '').join('\n'), model: data.model, usage: data.usage, provider: 'claude' };
}

async function callLLM(mode, systemPrompt, userMessage) {
    const hasDS = !!process.env.DEEPSEEK_API_KEY;
    const hasCL = !!process.env.ANTHROPIC_API_KEY;

    if (mode === 'interview' && hasCL) return callClaude(systemPrompt, userMessage);
    if (hasDS) return callDeepSeek(systemPrompt, userMessage);
    if (hasCL) return callClaude(systemPrompt, userMessage);

    throw new Error('No API keys configured. Add DEEPSEEK_API_KEY or ANTHROPIC_API_KEY to .env');
}

// ============================================================
// ENDPOINTS
// ============================================================

app.post('/codegrind/ask', async (req, res) => {
    try {
        const { challengeTitle, challengeCategory, challengePrompt, challengeSignature, language, userCode, testResults, mode = 'review' } = req.body;

        if (!challengeTitle || !userCode) return res.status(400).json({ error: 'challengeTitle and userCode required' });

        const langName = language || 'JavaScript';
        const langExt = {'JavaScript':'javascript','Python':'python','C++':'cpp','SQL':'sql'}[langName] || 'text';
        console.log(`[ASK] "${challengeTitle}" [${mode}] [${langName}]`);
        const ragContext = getRAGContext(challengeTitle, challengeCategory, mode);
        const systemPrompt = SPEAR_SYSTEM + (ragContext || '');

        const modeInstruction = MODE_PROMPTS[mode] || MODE_PROMPTS.review;
        const userMessage = `Challenge: "${challengeTitle}" (${challengeCategory || 'General'})
Problem: ${challengePrompt || 'N/A'}
Language: ${langName}
Signature: ${challengeSignature || 'N/A'}

Code:
\`\`\`${langExt}
${userCode}
\`\`\`

Tests:
${testResults || 'No tests run.'}

${modeInstruction}`;

        const result = await callLLM(mode, systemPrompt, userMessage);
        console.log(`[ASK] → ${result.provider} (${result.text.length} chars)`);

        res.json({ response: result.text, ragSources: !!ragContext, model: result.model, provider: result.provider, usage: result.usage });
    } catch (error) {
        console.error('[ASK] Error:', error.message);
        res.status(500).json({ error: error.message });
    }
});

app.post('/codegrind/search', (req, res) => {
    const { query, limit = 5 } = req.body;
    if (!query) return res.status(400).json({ error: 'query required' });

    const results = search(query, limit);
    res.json({
        query,
        results: results.map(r => ({ title: r.title, category: r.category, score: r.score.toFixed(3), preview: r.text.slice(0, 200) })),
        promptContext: formatForPrompt(results)
    });
});

app.post('/codegrind/reindex', (req, res) => {
    try { res.json({ status: 'ok', chunks: buildIndex() }); }
    catch (e) { res.status(500).json({ error: e.message }); }
});

app.get('/codegrind/health', (req, res) => {
    res.json({
        status: 'ok',
        hasDeepSeek: !!process.env.DEEPSEEK_API_KEY,
        hasAnthropic: !!process.env.ANTHROPIC_API_KEY,
        deepseekModel: process.env.DEEPSEEK_MODEL || 'deepseek-chat',
        claudeModel: process.env.CLAUDE_MODEL || 'claude-sonnet-4-20250514',
        knowledgeChunks: chunks.length,
        indexTerms: Object.keys(invertedIndex).length,
        routing: 'DeepSeek → code modes | Claude → interview prep',
        timestamp: new Date().toISOString()
    });
});

// ============================================================
// MULTI-LANGUAGE TEST RUNNER
// ============================================================

const { execFile } = require('child_process');
const os = require('os');

// Auto camelCase → snake_case conversion for dynamic/AI-generated names
function camelToSnake(name) {
    return name.replace(/([A-Z])/g, '_$1').toLowerCase();
}

// camelCase → snake_case mapping for all function/class names
const JS_TO_PY = {
    findMax:'find_max', reverseStr:'reverse_str', countOccurrences:'count_occurrences',
    fizzBuzz:'fizz_buzz', twoSum:'two_sum', twoSumFast:'two_sum_fast',
    maxSubarraySum:'max_subarray_sum', buildTree:'build_tree', maxDepth:'max_depth',
    inOrder:'in_order', isPalindrome:'is_palindrome', sumArray:'sum_array',
    countVowels:'count_vowels', removeDuplicates:'remove_duplicates',
    reverseInPlace:'reverse_in_place', isAnagram:'is_anagram',
    secondLargest:'second_largest', capitalizeWords:'capitalize_words',
    runningSum:'running_sum', missingNumber:'missing_number',
    isPowerOfTwo:'is_power_of_two', rotateArray:'rotate_array',
    isValid:'is_valid', mergeSorted:'merge_sorted', binarySearch:'binary_search',
    buildList:'build_list', reverseList:'reverse_list',
    lengthOfLongest:'length_of_longest', groupAnagrams:'group_anagrams',
    productExceptSelf:'product_except_self', hasCycle:'has_cycle',
    mergeSort:'merge_sort', hasPathSum:'has_path_sum', maxSubarray:'max_subarray',
    firstUnique:'first_unique', insertBST:'insert_bst', levelOrder:'level_order',
    climbStairs:'climb_stairs', movingAverage:'moving_average',
    zeroCrossingRate:'zero_crossing_rate', frameEnergy:'frame_energy',
    invertTree:'invert_tree', isValidBST:'is_valid_bst', coinChange:'coin_change',
    bfsDistance:'bfs_distance', detectPeaks:'detect_peaks',
    numIslands:'num_islands', topoSort:'topo_sort', rmsLevel:'rms_level',
    noiseGate:'noise_gate', ampToDb:'amp_to_db', dbToAmp:'db_to_amp',
    getMin:'get_min', MyQueue:'MyQueue', MinStack:'MinStack', CircBuf:'CircBuf',
    TreeNode:'TreeNode', ListNode:'ListNode'
};

const JS_TO_CPP = Object.assign({}, JS_TO_PY, { swap: 'swap_vars' });

const PY_SETUP = `
import json, math, sys

class TreeNode:
    def __init__(self, v):
        self.val = v
        self.left = None
        self.right = None

class ListNode:
    def __init__(self, v):
        self.val = v
        self.next = None

def build_tree():
    r = TreeNode(1)
    r.left = TreeNode(2); r.right = TreeNode(3)
    r.left.left = TreeNode(4); r.left.right = TreeNode(5)
    return r

def build_list(a):
    if not a: return None
    h = ListNode(a[0]); c = h
    for v in a[1:]: c.next = ListNode(v); c = c.next
    return h

def list_to_arr(h):
    r = []
    while h: r.append(h.val); h = h.next
    return r

def tree_vals(r):
    if not r: return []
    return [r.val] + tree_vals(r.left) + tree_vals(r.right)

def serialize(v):
    if v is None: return None
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)):
        if math.isinf(v): return "-Infinity" if v < 0 else "Infinity"
        return v
    if isinstance(v, str): return v
    if isinstance(v, list): return [serialize(x) for x in v]
    if isinstance(v, dict):
        return {str(k): serialize(vv) for k, vv in v.items()}
    if isinstance(v, TreeNode): return v.val
    if isinstance(v, ListNode): return list_to_arr(v)
    return str(v)
`;

function translateCallToPython(call) {
    let py = call;

    // Handle IIFE patterns: (function(){...})()
    if (py.startsWith('(function') || py.startsWith('(()')) {
        py = translateIIFEToPython(py);
        return py;
    }

    // Math.round → round, Math.abs → abs
    py = py.replace(/Math\.round/g, 'round');
    py = py.replace(/Math\.abs/g, 'abs');

    // new ClassName → ClassName
    py = py.replace(/\bnew\s+/g, '');

    // true/false/null
    py = py.replace(/\btrue\b/g, 'True');
    py = py.replace(/\bfalse\b/g, 'False');
    py = py.replace(/\bnull\b/g, 'None');

    // Replace known camelCase function names (longest first to avoid partial matches)
    const sorted = Object.entries(JS_TO_PY).sort((a,b) => b[0].length - a[0].length);
    for (const [js, py_name] of sorted) {
        py = py.replace(new RegExp('\\b' + js + '\\b', 'g'), py_name);
    }

    // Auto-convert remaining camelCase function names to snake_case (for AI-generated challenges)
    py = py.replace(/\b([a-z][a-zA-Z0-9]*)\s*\(/g, (match, name) => {
        // Skip if already snake_case or a Python builtin
        if (!name.match(/[A-Z]/) || ['round','abs','len','int','str','float','list','dict','set','range','print','sorted','reversed','enumerate','zip','map','filter','min','max','sum','any','all','type','isinstance','hasattr','getattr'].includes(name)) return match;
        const snake = name.replace(/([A-Z])/g, '_$1').toLowerCase();
        return snake + '(';
    });

    // Convert JS object literals: {A:['B'],C:[]} → {"A":["B"],"C":[]}
    // Quote unquoted keys in object literals
    py = py.replace(/\{([^}]+)\}/g, (match, inner) => {
        // Check if this looks like an object literal (has key:value pairs)
        if (inner.includes(':')) {
            const quoted = inner.replace(/(\b[A-Za-z_]\w*)\s*:/g, '"$1":');
            return '{' + quoted + '}';
        }
        return match;
    });

    // .length → len() — must come after name replacements
    py = py.replace(/(\w+)\.length/g, 'len($1)');

    return py;
}

function translateIIFEToPython(iife) {
    // Strip wrapper: (function(){...})() → extract body
    let body = iife.replace(/^\((?:function\s*\(\)|(?:\(\)\s*=>))\s*\{/, '').replace(/\}\s*\)\s*\(\)\s*$/, '');

    // Translate JS statements to Python
    body = body.replace(/\bconst\s+/g, '');
    body = body.replace(/\blet\s+/g, '');
    body = body.replace(/\bvar\s+/g, '');
    body = body.replace(/;/g, '\n');
    body = body.replace(/\bnew\s+/g, '');
    body = body.replace(/\btrue\b/g, 'True');
    body = body.replace(/\bfalse\b/g, 'False');
    body = body.replace(/\bnull\b/g, 'None');

    const sorted = Object.entries(JS_TO_PY).sort((a,b) => b[0].length - a[0].length);
    for (const [js, py_name] of sorted) {
        body = body.replace(new RegExp('\\b' + js + '\\b', 'g'), py_name);
    }

    // Turn "return X" into just X (the eval will capture it)
    const lines = body.split('\n').map(l => l.trim()).filter(Boolean);
    const pyLines = [];
    for (const line of lines) {
        if (line.startsWith('return ')) {
            pyLines.push('__result__ = ' + line.replace('return ', ''));
        } else {
            pyLines.push(line);
        }
    }

    // Wrap in a local function to allow multi-statement eval
    return `(lambda: [${pyLines.map(l => l.startsWith('__result__') ? l : l).join(', ')}])`;
}

function buildPythonScript(userCode, tests) {
    let script = PY_SETUP + '\n# USER CODE\n' + userCode + '\n\nresults = []\n';

    for (const test of tests) {
        const pyCall = translateCallToPython(test.call);

        // Check if this is a translated IIFE (multi-statement)
        if (test.call.startsWith('(function') || test.call.startsWith('(()')) {
            script += buildPythonIIFETest(test);
        } else {
            script += `
try:
    __r = ${pyCall}
    results.append({"pass": True, "result": serialize(__r), "error": None})
except Exception as e:
    results.append({"pass": False, "result": None, "error": str(e)})
`;
        }
    }

    script += '\nprint(json.dumps(results))\n';
    return script;
}

function buildPythonIIFETest(test) {
    let body = test.call.replace(/^\((?:function\s*\(\)|(?:\(\)\s*=>))\s*\{/, '').replace(/\}\s*\)\s*\(\)\s*$/, '');

    // Translate
    body = body.replace(/\bconst\s+/g, '').replace(/\blet\s+/g, '').replace(/\bvar\s+/g, '');
    body = body.replace(/\bnew\s+/g, '');
    body = body.replace(/\btrue\b/g, 'True').replace(/\bfalse\b/g, 'False').replace(/\bnull\b/g, 'None');

    const sorted = Object.entries(JS_TO_PY).sort((a,b) => b[0].length - a[0].length);
    for (const [js, py_name] of sorted) {
        body = body.replace(new RegExp('\\b' + js + '\\b', 'g'), py_name);
    }

    // .length → len()
    body = body.replace(/(\w+)\.length/g, 'len($1)');

    const stmts = body.split(';').map(s => s.trim()).filter(Boolean);
    let pyBlock = 'try:\n';
    for (const stmt of stmts) {
        if (stmt.startsWith('return ')) {
            pyBlock += '    __r = ' + stmt.replace('return ', '') + '\n';
        } else {
            pyBlock += '    ' + stmt + '\n';
        }
    }
    pyBlock += '    results.append({"pass": True, "result": serialize(__r), "error": None})\n';
    pyBlock += 'except Exception as e:\n';
    pyBlock += '    results.append({"pass": False, "result": None, "error": str(e)})\n';
    return pyBlock;
}

// ---- C++ Test Runner (parser-based) ----

// Tokenize a JS expression into tokens for proper conversion
function tokenizeJSExpr(expr) {
    const tokens = [];
    let i = 0;
    while (i < expr.length) {
        if (/\s/.test(expr[i])) { i++; continue; }
        // String literal (single or double quote)
        if (expr[i] === "'" || expr[i] === '"') {
            const q = expr[i]; let s = ''; i++;
            while (i < expr.length && expr[i] !== q) { s += expr[i]; i++; }
            i++; // skip closing quote
            tokens.push({ type: 'string', value: s });
        }
        // Number
        else if (/[\d]/.test(expr[i]) || (expr[i] === '-' && /[\d]/.test(expr[i+1] || ''))) {
            let n = ''; if (expr[i] === '-') { n = '-'; i++; }
            while (i < expr.length && /[\d.eE\-+]/.test(expr[i])) { n += expr[i]; i++; }
            tokens.push({ type: 'number', value: n });
        }
        // Identifier or keyword
        else if (/[a-zA-Z_$]/.test(expr[i])) {
            let id = '';
            while (i < expr.length && /[a-zA-Z0-9_$]/.test(expr[i])) { id += expr[i]; i++; }
            tokens.push({ type: 'ident', value: id });
        }
        // Punctuation
        else {
            tokens.push({ type: 'punct', value: expr[i] }); i++;
        }
    }
    return tokens;
}

// Convert a JS expression string to valid C++ expression string
function jsToCppExpr(expr) {
    // Handle the full expression, converting arrays, strings, function names
    let cpp = '';
    const tokens = tokenizeJSExpr(expr);

    for (let i = 0; i < tokens.length; i++) {
        const t = tokens[i];
        const prev = tokens[i-1];
        const next = tokens[i+1];

        if (t.type === 'string') {
            cpp += '"' + t.value + '"';
        }
        else if (t.type === 'number') {
            cpp += t.value;
        }
        else if (t.type === 'ident') {
            // JS keywords → C++
            if (t.value === 'true' || t.value === 'false') { cpp += t.value; continue; }
            if (t.value === 'null') { cpp += 'nullptr'; continue; }
            if (t.value === 'new') { cpp += 'new '; continue; }
            if (t.value === 'Math') {
                // Math.round, Math.abs, etc
                if (next && next.value === '.' && tokens[i+2]) {
                    const method = tokens[i+2].value;
                    if (method === 'round') cpp += 'round';
                    else if (method === 'abs') cpp += 'abs';
                    else if (method === 'floor') cpp += '(int)floor';
                    else if (method === 'ceil') cpp += '(int)ceil';
                    else cpp += method;
                    i += 2; // skip dot and method name
                    continue;
                }
            }
            // Function/variable name translation (fall back to auto camelCase→snake_case)
            const cppName = JS_TO_CPP[t.value] || (t.value.match(/[A-Z]/) ? camelToSnake(t.value) : t.value);
            cpp += cppName;
        }
        else if (t.type === 'punct') {
            if (t.value === '[') {
                // Is this array indexing or array literal?
                // Array literal if: start of expr, after ( , after =
                const isLiteral = !prev || prev.value === '(' || prev.value === ','
                    || prev.value === '=' || prev.value === '[' || prev.value === ':';

                if (isLiteral) {
                    // Collect everything until matching ]
                    const arrContent = collectBracketContent(tokens, i);
                    i = arrContent.endIdx;
                    const cppArr = convertArrayLiteral(arrContent.inner, tokens, arrContent.startIdx);
                    cpp += cppArr;
                } else {
                    // Indexing like arr[0]
                    cpp += '[';
                }
            }
            else if (t.value === '.') {
                // Smart dot conversion: pointer members use ->, stack methods use .
                const nextTok = tokens[i+1];
                if (nextTok && nextTok.type === 'ident') {
                    const pointerMembers = new Set(['val','left','right','next']);
                    if (pointerMembers.has(nextTok.value)) {
                        cpp += '->';
                    } else {
                        cpp += '.';
                    }
                } else {
                    cpp += '.';
                }
            }
            else {
                cpp += t.value;
            }
        }
    }
    // Post-processing: promote all vector<int> to vector<double> if any vector<double> exists
    if (cpp.includes('vector<double>') && cpp.includes('vector<int>')) {
        cpp = cpp.replace(/vector<int>/g, 'vector<double>');
    }
    return cpp;
}

// Collect tokens inside [ ... ] handling nesting
function collectBracketContent(tokens, startIdx) {
    let depth = 0;
    let i = startIdx;
    const inner = [];
    i++; // skip opening [
    while (i < tokens.length) {
        if (tokens[i].value === '[') { depth++; inner.push(tokens[i]); }
        else if (tokens[i].value === ']') {
            if (depth === 0) break;
            depth--; inner.push(tokens[i]);
        }
        else { inner.push(tokens[i]); }
        i++;
    }
    return { inner, endIdx: i, startIdx };
}

// Convert a collected array literal to a C++ vector initializer
function convertArrayLiteral(innerTokens, allTokens, outerStartIdx) {
    if (innerTokens.length === 0) return 'vector<int>{}';

    // Check if it's a 2D array (inner tokens contain [ ])
    const hasSubArrays = innerTokens.some(t => t.value === '[');
    if (hasSubArrays) {
        return convert2DArray(innerTokens);
    }

    // Check if elements contain complex expressions (dots, parens, etc.)
    const hasExpressions = innerTokens.some(t => t.value === '.' || t.value === '(' || t.value === ')');
    if (hasExpressions) {
        return convertExpressionArray(innerTokens);
    }

    // Detect element type
    const hasStrings = innerTokens.some(t => t.type === 'string');
    const hasDoubles = innerTokens.some(t => t.type === 'number' && t.value.includes('.'));

    // Convert elements
    const elements = [];
    for (const t of innerTokens) {
        if (t.value === ',') continue;
        if (t.type === 'string') elements.push('"' + t.value + '"');
        else if (t.type === 'number') elements.push(t.value);
        else if (t.type === 'ident') {
            if (t.value === 'true') elements.push('true');
            else if (t.value === 'false') elements.push('false');
            else elements.push(t.value);
        }
    }

    if (hasStrings) return `vector<string>{${elements.join(',')}}`;
    if (hasDoubles) return `vector<double>{${elements.join(',')}}`;
    return `vector<int>{${elements.join(',')}}`;
}

// Handle arrays with complex expression elements like [r.val, r.next.val]
function convertExpressionArray(innerTokens) {
    // Split into top-level comma-separated groups
    const groups = [];
    let current = [];
    let depth = 0;
    for (const t of innerTokens) {
        if (t.value === '(' || t.value === '[') depth++;
        if (t.value === ')' || t.value === ']') depth--;
        if (t.value === ',' && depth === 0) {
            if (current.length) groups.push(current);
            current = [];
        } else {
            current.push(t);
        }
    }
    if (current.length) groups.push(current);

    // Convert each group back to a string and run through jsToCppExpr
    const elements = groups.map(g => {
        const jsStr = g.map(t => {
            if (t.type === 'string') return '"' + t.value + '"';
            return t.value;
        }).join('');
        return jsToCppExpr(jsStr);
    });

    // Infer type from first element
    const firstGroup = groups[0];
    const hasStrings = firstGroup.some(t => t.type === 'string');
    if (hasStrings) return `vector<string>{${elements.join(',')}}`;
    return `vector<int>{${elements.join(',')}}`;
}

function convert2DArray(innerTokens) {
    // Detect if elements are chars/strings or ints
    const hasStrings = innerTokens.some(t => t.type === 'string');

    const rows = [];
    let depth = 0, currentRow = [];
    for (const t of innerTokens) {
        if (t.value === '[') {
            depth++;
            if (depth === 1) { currentRow = []; continue; }
        }
        if (t.value === ']') {
            depth--;
            if (depth === 0) { rows.push([...currentRow]); continue; }
        }
        if (depth === 0 && t.value === ',') continue;
        if (depth >= 1) currentRow.push(t);
    }

    // Check if inner elements are single chars
    const isSingleChars = hasStrings && rows.every(row =>
        row.filter(t => t.type === 'string').every(t => t.value.length === 1)
    );

    if (isSingleChars) {
        // vector<vector<char>>
        const rowStrs = rows.map(row => {
            const elems = row.filter(t => t.value !== ',').map(t =>
                t.type === 'string' ? `'${t.value}'` : t.value
            );
            return `{${elems.join(',')}}`;
        });
        return `vector<vector<char>>{${rowStrs.join(',')}}`;
    }

    if (hasStrings) {
        const rowStrs = rows.map(row => {
            const elems = row.filter(t => t.value !== ',').map(t =>
                t.type === 'string' ? `"${t.value}"` : t.value
            );
            return `{${elems.join(',')}}`;
        });
        return `vector<vector<string>>{${rowStrs.join(',')}}`;
    }

    // vector<vector<int>>
    const rowStrs = rows.map(row => {
        const elems = row.filter(t => t.value !== ',').map(t => t.value);
        return `{${elems.join(',')}}`;
    });
    return `vector<vector<int>>{${rowStrs.join(',')}}`;
}

// Handle graph object literals: {A:['B','C'], ...}
function convertGraphObj(expr) {
    // Parse {key:[vals], ...} → unordered_map
    const m = expr.match(/\{(.+)\}/);
    if (!m) return null;
    const entries = [];
    const re = /(\w+):\[([^\]]*)\]/g;
    let em;
    while ((em = re.exec(m[1]))) {
        const key = em[1];
        const vals = em[2] ? em[2].match(/'([^']+)'/g)?.map(v => `"${v.slice(1,-1)}"`) || [] : [];
        entries.push(`{"${key}", {${vals.join(',')}}}`);
    }
    return `unordered_map<string,vector<string>>{${entries.join(',')}}`;
}

const CPP_SETUP = `
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <stack>
#include <algorithm>
#include <cmath>
#include <climits>
#include <limits>
#include <sstream>
using namespace std;

struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int v) : val(v), left(nullptr), right(nullptr) {}
};

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int v) : val(v), next(nullptr) {}
};

TreeNode* build_tree() {
    TreeNode* r = new TreeNode(1);
    r->left = new TreeNode(2); r->right = new TreeNode(3);
    r->left->left = new TreeNode(4); r->left->right = new TreeNode(5);
    return r;
}

ListNode* build_list(vector<int> a) {
    if (a.empty()) return nullptr;
    ListNode* h = new ListNode(a[0]); ListNode* c = h;
    for (int i = 1; i < (int)a.size(); i++) { c->next = new ListNode(a[i]); c = c->next; }
    return h;
}

// JSON serializers
string to_json(int v) { return to_string(v); }
string to_json(long long v) { return to_string(v); }
string to_json(bool v) { return v ? "true" : "false"; }
string to_json(double v) {
    if (isinf(v)) return v < 0 ? "\\"-Infinity\\"" : "\\"Infinity\\"";
    ostringstream oss; oss << v;
    string s = oss.str();
    if (s.find('.') == string::npos && s.find('e') == string::npos) s += ".0";
    return s;
}
string to_json(const string& v) { return "\\"" + v + "\\""; }

string to_json(const vector<int>& v) {
    string s = "[";
    for (int i = 0; i < (int)v.size(); i++) { if (i) s += ","; s += to_string(v[i]); }
    return s + "]";
}
string to_json(const vector<double>& v) {
    string s = "[";
    for (int i = 0; i < (int)v.size(); i++) { if (i) s += ","; s += to_json(v[i]); }
    return s + "]";
}
string to_json(const vector<string>& v) {
    string s = "[";
    for (int i = 0; i < (int)v.size(); i++) { if (i) s += ","; s += to_json(v[i]); }
    return s + "]";
}
string to_json(const vector<vector<int>>& v) {
    string s = "[";
    for (int i = 0; i < (int)v.size(); i++) { if (i) s += ","; s += to_json(v[i]); }
    return s + "]";
}
string to_json(const vector<vector<string>>& v) {
    string s = "[";
    for (int i = 0; i < (int)v.size(); i++) { if (i) s += ","; s += to_json(v[i]); }
    return s + "]";
}
string to_json(const unordered_map<int,int>& m) {
    string s = "{";
    bool first = true;
    for (auto& p : m) { if (!first) s += ","; first = false; s += "\\"" + to_string(p.first) + "\\":" + to_string(p.second); }
    return s + "}";
}
string to_json(const unordered_map<string,int>& m) {
    string s = "{";
    bool first = true;
    for (auto& p : m) { if (!first) s += ","; first = false; s += "\\"" + p.first + "\\":" + to_string(p.second); }
    return s + "}";
}

string list_to_json(ListNode* h) {
    string s = "[";
    bool first = true;
    while (h) { if (!first) s += ","; first = false; s += to_string(h->val); h = h->next; }
    return s + "]";
}

string tree_preorder(TreeNode* r) {
    if (!r) return "[]";
    string s = "[" + to_string(r->val);
    if (r->left || r->right) {
        s += "," + tree_preorder(r->left) + "," + tree_preorder(r->right);
    }
    return s + "]";
}

// Pointer overloads for auto-serialization
string to_json(TreeNode* r) { return r ? to_string(r->val) : "null"; }
string to_json(ListNode* h) {
    string s = "["; bool first = true;
    while (h) { if (!first) s += ","; first = false; s += to_string(h->val); h = h->next; }
    return s + "]";
}
string to_json(const vector<vector<char>>& v) {
    string s = "[";
    for (int i = 0; i < (int)v.size(); i++) {
        if (i) s += ",";
        s += "[";
        for (int j = 0; j < (int)v[i].size(); j++) {
            if (j) s += ",";
            s += string("\\"") + v[i][j] + "\\"";
        }
        s += "]";
    }
    return s + "]";
}
string to_json(const vector<bool>& v) {
    string s = "[";
    for (int i = 0; i < (int)v.size(); i++) { if (i) s += ","; s += v[i] ? "true" : "false"; }
    return s + "]";
}
string to_json(char c) { return string("\\"") + c + "\\""; }
`;


// Convert a JS test call to a C++ test block
function buildCppTest(call, testIdx) {
    // Handle IIFE patterns
    if (call.startsWith('(function') || call.startsWith('(()')) {
        return buildCppIIFEBlock(call, testIdx);
    }

    // Handle graph object arguments: bfsDistance({A:['B','C'],...}, 'A')
    if (call.includes(':{') || call.match(/\(\s*\{[A-Za-z]/)) {
        return buildCppGraphCall(call, testIdx);
    }

    // Pre-declare any nested array arguments to avoid rvalue-to-ref binding issues

    // First check for 2D string/char grids: [["1","0"],["0","1"]]
    let predecl = '';
    let modCall = call;
    let arrIdx = 0;
    const charGridRe = /\[\s*(\[["'][^"']*["'](?:\s*,\s*["'][^"']*["'])*\](?:\s*,\s*\[["'][^"']*["'](?:\s*,\s*["'][^"']*["'])*\])*)\s*\]/;
    const cgm = modCall.match(charGridRe);
    if (cgm) {
        const varName = `__grid${testIdx}`;
        const rows = cgm[1].match(/\[([^\]]+)\]/g);
        predecl += `        vector<vector<string>> ${varName} = {\n`;
        rows.forEach(row => {
            const vals = row.match(/["']([^"']*)["']/g).map(v => `"${v.slice(1,-1)}"`);
            predecl += `            {${vals.join(',')}},\n`;
        });
        predecl += `        };\n`;
        modCall = modCall.replace(cgm[0], varName);
    }

    // Then check for numeric nested arrays: [[0,1],[2,3]]
    const nestedArrRe = /\[(\[[^\]]*\](?:\s*,\s*\[[^\]]*\])*)\]/g;
    let nm;
    while ((nm = nestedArrRe.exec(modCall)) !== null) {
        const varName = `__arr${testIdx}_${arrIdx++}`;
        const inner = nm[1].replace(/\[([^\]]*)\]/g, '{$1}');
        predecl += `        vector<vector<int>> ${varName} = {${inner}};\n`;
        modCall = modCall.replace(nm[0], varName);
    }

    // Standard function call — use the parser
    const cppExpr = jsToCppExpr(modCall);
    return predecl + `        auto __r${testIdx} = ${cppExpr};
        cout << "{\\"index\\":${testIdx},\\"result\\":" << to_json(__r${testIdx}) << "}" << endl;\n`;
}

function buildCppGraphCall(call, idx) {
    // bfsDistance({A:['B','C'],B:['D'],C:[],D:[]}, 'A')
    const m = call.match(/(\w+)\(\{(.+?)\},\s*['"](\w+)['"]\)/);
    if (!m) return `        cout << "{\\"index\\":${idx},\\"error\\":\\"parse_graph\\"}" << endl;\n`;

    const fname = JS_TO_CPP[m[1]] || (m[1].match(/[A-Z]/) ? camelToSnake(m[1]) : m[1]);
    const entries = [];
    const re = /(\w+):\[([^\]]*)\]/g;
    let em;
    while ((em = re.exec(m[2]))) {
        const key = em[1];
        const vals = em[2] ? em[2].match(/['"]([^'"]+)['"]/g)?.map(v => `"${v.slice(1,-1)}"`) || [] : [];
        entries.push(`{"${key}", {${vals.join(',')}}}`);
    }

    return `        unordered_map<string,vector<string>> __g${idx} = {${entries.join(',')}};
        auto __r${idx} = ${fname}(__g${idx}, "${m[3]}");
        // Serialize string->int map
        string __s${idx} = "{";
        bool __f${idx} = true;
        for (auto& p : __r${idx}) {
            if (!__f${idx}) __s${idx} += ",";
            __f${idx} = false;
            __s${idx} += "\\"" + p.first + "\\":" + to_string(p.second);
        }
        __s${idx} += "}";
        cout << "{\\"index\\":${idx},\\"result\\":" << __s${idx} << "}" << endl;\n`;
}

function buildCppIIFEBlock(call, idx) {
    let body = call.replace(/^\((?:function\s*\(\)|(?:\(\)\s*=>))\s*\{/, '').replace(/\}\s*\)\s*\(\)\s*$/, '');

    // Split into statements
    const stmts = body.split(';').map(s => s.trim()).filter(Boolean);
    let code = '';
    let hasResult = false;
    const declaredVars = new Set();

    for (const stmt of stmts) {
        if (stmt.startsWith('return ')) {
            const expr = stmt.slice(7);
            const cppExpr = jsToCppExpr(expr);
            code += `        auto __r${idx} = ${cppExpr};\n`;
            hasResult = true;
        } else {
            // Variable declaration or statement
            const hadDecl = /^(?:const|let|var)\s+/.test(stmt);
            let s = stmt.replace(/^(?:const|let|var)\s+/, '');

            // Parse into: varName = expression
            const eqIdx = s.indexOf('=');
            if (eqIdx > 0) {
                const varName = s.slice(0, eqIdx).trim();
                let valExpr = s.slice(eqIdx + 1).trim();

                // Check if this is a property assignment (contains dots)
                if (varName.includes('.')) {
                    // Property assignment like r.left=new TreeNode(3)
                    const cppLhs = jsToCppExpr(varName);
                    const cppRhs = jsToCppExpr(valExpr);
                    code += `        ${cppLhs} = ${cppRhs};\n`;
                } else {
                    const cppName = JS_TO_CPP[varName] || (varName.match(/[A-Z]/) ? camelToSnake(varName) : varName);
                    // Strip 'new' for stack-allocated class types (not TreeNode/ListNode which need heap)
                    const nodeTypes = /^new\s+(TreeNode|ListNode)\b/;
                    if (!nodeTypes.test(valExpr)) {
                        valExpr = valExpr.replace(/^new\s+/, '');
                    }
                    const cppVal = jsToCppExpr(valExpr);

                    if (declaredVars.has(cppName)) {
                        // Re-assignment, no auto
                        code += `        ${cppName} = ${cppVal};\n`;
                    } else if (hadDecl || !declaredVars.has(cppName)) {
                        // First declaration — use auto, but for nullptr use explicit type
                        if (valExpr === 'null' || valExpr === 'nullptr') {
                            code += `        TreeNode* ${cppName} = nullptr;\n`;
                        } else {
                            code += `        auto ${cppName} = ${cppVal};\n`;
                        }
                        declaredVars.add(cppName);
                    }
                }
            } else {
                // Bare statement like q.enqueue(1) or s.push(3)
                const cppStmt = jsToCppExpr(s);
                code += `        ${cppStmt};\n`;
            }
        }
    }

    if (hasResult) {
        code += `        cout << "{\\"index\\":${idx},\\"result\\":" << to_json(__r${idx}) << "}" << endl;\n`;
    } else {
        code += `        cout << "{\\"index\\":${idx},\\"result\\":null}" << endl;\n`;
    }

    return code;
}

function buildCppTestHarness(userCode, tests, challengeId) {
    let code = CPP_SETUP;

    // Remove duplicate struct definitions from user code
    let cleanCode = userCode;
    cleanCode = cleanCode.replace(/struct TreeNode\s*\{[^}]+\};/gs, '');
    cleanCode = cleanCode.replace(/struct ListNode\s*\{[^}]+\};/gs, '');

    code += '\n// USER CODE\n' + cleanCode + '\n\n';
    code += 'int main() {\n';

    let testCode = '';
    for (let i = 0; i < tests.length; i++) {
        const test = tests[i];
        testCode += `    // Test ${i}\n    {\n    try {\n`;
        testCode += buildCppTest(test.call, i);
        testCode += '    } catch(exception& e) {\n';
        testCode += `        cout << "{\\"index\\":${i},\\"error\\":\\"" << e.what() << "\\"}" << endl;\n`;
        testCode += '    } catch(...) {\n';
        testCode += `        cout << "{\\"index\\":${i},\\"error\\":\\"runtime error\\"}" << endl;\n`;
        testCode += '    }\n    }\n';
    }

    // Post-processing: if user code expects vector<double>, promote all vector<int> in tests
    if (cleanCode.includes('vector<double>') && testCode.includes('vector<int>')) {
        testCode = testCode.replace(/vector<int>/g, 'vector<double>');
    }

    code += testCode;
    code += '    return 0;\n}\n';
    return code;
}
// ---- Execution functions ----

function runPython(script, timeout = 10000) {
    return new Promise((resolve, reject) => {
        const tmpFile = path.join(os.tmpdir(), `cg_test_${Date.now()}.py`);
        fs.writeFileSync(tmpFile, script);

        execFile('python3', [tmpFile], { timeout }, (error, stdout, stderr) => {
            try { fs.unlinkSync(tmpFile); } catch(e) {}
            if (error) {
                if (error.killed) return reject(new Error('Timeout: code took too long'));
                return reject(new Error(stderr || error.message));
            }
            resolve(stdout.trim());
        });
    });
}

function runCpp(source, timeout = 15000) {
    return new Promise((resolve, reject) => {
        const tmpSrc = path.join(os.tmpdir(), `cg_test_${Date.now()}.cpp`);
        const tmpBin = tmpSrc.replace('.cpp', '');
        fs.writeFileSync(tmpSrc, source);

        execFile('g++', ['-std=c++17', '-O2', '-o', tmpBin, tmpSrc], { timeout: 10000 }, (compErr, compOut, compStderr) => {
            try { fs.unlinkSync(tmpSrc); } catch(e) {}
            if (compErr) {
                // Extract useful compilation errors
                const errMsg = compStderr.replace(new RegExp(tmpSrc, 'g'), 'code.cpp').split('\n').slice(0, 10).join('\n');
                return reject(new Error('Compilation error:\n' + errMsg));
            }

            execFile(tmpBin, [], { timeout }, (runErr, stdout, stderr) => {
                try { fs.unlinkSync(tmpBin); } catch(e) {}
                if (runErr) {
                    if (runErr.killed) return reject(new Error('Timeout: code took too long'));
                    return reject(new Error(stderr || runErr.message));
                }
                resolve(stdout.trim());
            });
        });
    });
}

// ---- Compare results ----

function deepEq(a, b) {
    if (a === b) return true;
    // Coerce string/number comparisons (AI sometimes mismatches types in test expectations)
    if (typeof a === 'number' && typeof b === 'string') { b = Number(b); if (!isNaN(b)) return Math.abs(a - b) < 1e-6; return false; }
    if (typeof a === 'string' && typeof b === 'number') { a = Number(a); if (!isNaN(a)) return Math.abs(a - b) < 1e-6; return false; }
    if (typeof a !== typeof b) return false;
    if (typeof a === 'number' && typeof b === 'number') return Math.abs(a - b) < 1e-6;
    if (Array.isArray(a) && Array.isArray(b)) {
        if (a.length !== b.length) return false;
        for (let i = 0; i < a.length; i++) if (!deepEq(a[i], b[i])) return false;
        return true;
    }
    if (typeof a === 'object' && a !== null && b !== null) {
        const ka = Object.keys(a).sort(), kb = Object.keys(b).sort();
        if (ka.length !== kb.length) return false;
        for (let i = 0; i < ka.length; i++) {
            if (ka[i] !== kb[i]) return false;
            if (!deepEq(a[ka[i]], b[kb[i]])) return false;
        }
        return true;
    }
    return false;
}

// ---- Endpoint ----

app.post('/codegrind/run-tests', async (req, res) => {
    const { language, code, tests, challengeId, setup } = req.body;

    if (!language || !code || !tests) {
        return res.status(400).json({ error: 'language, code, and tests required' });
    }

    console.log(`[TEST] Challenge #${challengeId} [${language}] ${tests.length} tests`);

    try {
        let results;

        if (language === 'py') {
            const script = buildPythonScript(code, tests);
            const raw = await runPython(script);
            let parsed;
            try { parsed = JSON.parse(raw); } catch(e) {
                return res.json({ results: tests.map((t, i) => ({
                    index: i, pass: false, error: 'Output parse error: ' + raw.substring(0, 200)
                }))});
            }
            results = parsed.map((r, i) => {
                if (r.error) return { index: i, pass: false, result: null, error: r.error, expected: tests[i].expect };
                const pass = deepEq(r.result, tests[i].expect);
                return { index: i, pass, result: r.result, expected: tests[i].expect, error: null };
            });
        } else if (language === 'cpp') {
            const source = buildCppTestHarness(code, tests, challengeId);
            const raw = await runCpp(source);
            const lines = raw.split('\n').filter(Boolean);
            results = tests.map((t, i) => {
                const line = lines.find(l => {
                    try { return JSON.parse(l).index === i; } catch(e) { return false; }
                });
                if (!line) return { index: i, pass: false, result: null, error: 'No output for test', expected: t.expect };
                try {
                    const parsed = JSON.parse(line);
                    if (parsed.error) return { index: i, pass: false, result: null, error: parsed.error, expected: t.expect };
                    const pass = deepEq(parsed.result, t.expect);
                    return { index: i, pass, result: parsed.result, expected: t.expect, error: null };
                } catch(e) {
                    return { index: i, pass: false, result: null, error: 'Parse error: ' + line, expected: t.expect };
                }
            });
        } else {
            return res.status(400).json({ error: 'Unsupported language: ' + language });
        }

        res.json({ results });
    } catch (e) {
        console.error(`[TEST ERROR]`, e.message);
        // Return all tests as failed with the error
        res.json({
            results: tests.map((t, i) => ({
                index: i, pass: false, result: null, error: e.message, expected: t.expect
            }))
        });
    }
});

// ============================================================
// AI CHALLENGE GENERATION
// ============================================================

const TIER_CATS = {
    1: ["Variables & Logic", "Arrays", "Strings", "Logic"],
    2: ["Hash Maps", "Sliding Window", "Binary Trees", "Two Pointers", "Stacks", "Sorting", "Binary Search", "Linked Lists", "Recursion"],
    3: ["Dynamic Programming", "Audio DSP", "Graphs", "Binary Trees", "Stacks"]
};

const TIER_DIFFICULTY = {
    1: "easy (LeetCode Easy equivalent). Simple loops, conditionals, basic array/string manipulation. No data structures beyond arrays/objects.",
    2: "medium (LeetCode Easy-Medium). Hash maps, two pointers, sliding window, basic trees, linked lists, stacks, sorting algorithms, binary search.",
    3: "hard (LeetCode Medium). Dynamic programming, graph traversal (BFS/DFS), topological sort, audio DSP algorithms (filters, FFT concepts, signal processing), advanced tree operations."
};

// ---- LEETCODE PROBLEM POOL ----
// Curated free problems by tier, mapped to LeetCode slugs
const LC_POOL = {
    1: [ // Easy
        "two-sum","valid-parentheses","merge-two-sorted-lists","best-time-to-buy-and-sell-stock",
        "valid-palindrome","linked-list-cycle","reverse-linked-list","contains-duplicate",
        "valid-anagram","missing-number","move-zeroes","power-of-three","fizz-buzz",
        "reverse-string","intersection-of-two-arrays-ii","first-unique-character-in-a-string",
        "majority-element","roman-to-integer","palindrome-number","plus-one",
        "sqrt-x","climbing-stairs","remove-duplicates-from-sorted-array","single-number",
        "maximum-subarray","merge-sorted-array","pascal-s-triangle","remove-element",
        "search-insert-position","length-of-last-word","add-binary","excel-sheet-column-title",
        "happy-number","isomorphic-strings","count-primes","power-of-two",
        "implement-queue-using-stacks","number-of-1-bits","reverse-bits","hamming-distance",
        "find-the-difference","ransom-note","longest-palindrome","third-maximum-number",
        "add-strings","number-of-segments-in-a-string","arranging-coins","find-all-numbers-disappeared-in-an-array",
        "assign-cookies","keyboard-row","base-7","relative-ranks","detect-capital"
    ],
    2: [ // Medium
        "add-two-numbers","longest-substring-without-repeating-characters","container-with-most-water",
        "3sum","letter-combinations-of-a-phone-number","remove-nth-node-from-end-of-list",
        "generate-parentheses","next-permutation","search-in-rotated-sorted-array",
        "find-first-and-last-position-of-element-in-sorted-array","combination-sum",
        "rotate-image","group-anagrams","spiral-matrix","jump-game","merge-intervals",
        "unique-paths","minimum-path-sum","sort-colors","subsets","word-search",
        "binary-tree-inorder-traversal","validate-binary-search-tree",
        "binary-tree-level-order-traversal","construct-binary-tree-from-preorder-and-inorder-traversal",
        "flatten-binary-tree-to-linked-list","best-time-to-buy-and-sell-stock-ii",
        "word-break","copy-list-with-random-pointer","single-number-ii",
        "clone-graph","gas-station","number-of-islands","course-schedule",
        "implement-trie-prefix-tree","kth-largest-element-in-an-array","product-of-array-except-self",
        "search-a-2d-matrix-ii","meeting-rooms-ii","top-k-frequent-elements",
        "decode-string","queue-reconstruction-by-height","partition-equal-subset-sum",
        "find-all-anagrams-in-a-string","subarray-sum-equals-k","task-scheduler",
        "daily-temperatures","asteroid-collision","set-matrix-zeroes"
    ],
    3: [ // Hard-ish (LeetCode Medium-Hard)
        "longest-palindromic-substring","regular-expression-matching","merge-k-sorted-lists",
        "trapping-rain-water","maximum-subarray","edit-distance","minimum-window-substring",
        "largest-rectangle-in-histogram","maximal-rectangle","word-ladder",
        "longest-consecutive-sequence","word-break-ii","maximum-product-subarray",
        "house-robber-ii","coin-change","longest-increasing-subsequence",
        "perfect-squares","russian-doll-envelopes","count-of-smaller-numbers-after-self",
        "reconstruct-itinerary","serialize-and-deserialize-binary-tree",
        "median-of-two-sorted-arrays","burst-balloons","super-ugly-number"
    ]
};

async function fetchLeetCodeProblem(slug) {
    // Primary: LeetCode's own GraphQL API (most reliable)
    try {
        console.log(`[LC] Fetching via GraphQL: ${slug}`);
        const resp = await fetch('https://leetcode.com/graphql', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Referer': 'https://leetcode.com' },
            body: JSON.stringify({
                query: `query questionData($titleSlug: String!) {
                    question(titleSlug: $titleSlug) {
                        title titleSlug difficulty content
                        exampleTestcaseList
                        topicTags { name }
                        hints
                    }
                }`,
                variables: { titleSlug: slug }
            }),
            signal: AbortSignal.timeout(10000)
        });
        if (resp.ok) {
            const json = await resp.json();
            const q = json?.data?.question;
            if (q && q.content) {
                console.log(`[LC] GraphQL success: ${q.title}`);
                return q;
            }
        }
    } catch (e) {
        console.log(`[LC] GraphQL failed: ${e.message}`);
    }

    // Fallback: alfa-leetcode third-party API
    try {
        const url = `https://alfa-leetcode-api.onrender.com/select?titleSlug=${slug}`;
        console.log(`[LC] Fetching fallback: ${url}`);
        const resp = await fetch(url, { signal: AbortSignal.timeout(10000) });
        if (resp.ok) {
            const data = await resp.json();
            // This API returns problem HTML in "question" field (string, not object)
            if (data.question || data.content) {
                const normalized = {
                    title: data.questionTitle || data.title || slug,
                    titleSlug: data.titleSlug || slug,
                    difficulty: data.difficulty || 'Medium',
                    content: data.question || data.content,
                    topicTags: data.topicTags || [],
                    hints: data.hints || [],
                    exampleTestcaseList: data.exampleTestcases
                        ? data.exampleTestcases.split('\n') : []
                };
                console.log(`[LC] Fallback API success: ${normalized.title}`);
                return normalized;
            }
        }
    } catch (e) {
        console.log(`[LC] Fallback API failed: ${e.message}`);
    }

    return null;
}

function stripHtml(html) {
    return html
        .replace(/<pre>\s*/g, '\n```\n')
        .replace(/<\/pre>/g, '\n```\n')
        .replace(/<code>/g, '`').replace(/<\/code>/g, '`')
        .replace(/<strong[^>]*>/g, '**').replace(/<\/strong>/g, '**')
        .replace(/<em>/g, '*').replace(/<\/em>/g, '*')
        .replace(/<li>/g, '- ').replace(/<\/li>/g, '\n')
        .replace(/<p[^>]*>/g, '\n').replace(/<\/p>/g, '\n')
        .replace(/<[^>]+>/g, '')
        .replace(/&nbsp;/g, ' ')
        .replace(/&lt;/g, '<').replace(/&gt;/g, '>')
        .replace(/&amp;/g, '&').replace(/&quot;/g, '"')
        .replace(/&#39;/g, "'")
        .replace(/\n{3,}/g, '\n\n')
        .trim();
}

const LC_CONVERT_SYSTEM = `You are a coding challenge adapter for Code Grind. You convert LeetCode problems into a JSON format that loads into a code editor.

CRITICAL FORMAT RULES:
- Return ONLY valid JSON (one object, NOT an array). No markdown, no backticks, no text outside the JSON.
- The "sig" field is the EMPTY function signature shown in the editor. It has NO logic — just the shell with params.
- The "solution" field is the COMPLETE working function with the same name as sig.
- Function names: camelCase in JavaScript, snake_case in Python and C++.
- The JS function name in "sig", "solution", and every "tests[].call" MUST be identical.
- Test "call" strings are JavaScript expressions, e.g. "twoSum([2,7,11,15], 9)"
- Test "expect" values must be EXACT. Mentally run the solution to verify each one.
- Use ONLY simple types: arrays of ints/strings, strings, numbers, booleans. No TreeNode, ListNode, or custom classes.
- If the problem uses linked lists or trees, ADAPT it to use plain arrays.
- Keep test inputs small: arrays under 10 elements, numbers under 10000.
- No external libraries. Python: no type hints. C++: include headers, use "using namespace std;".
- The "prompt" should be a clean rewrite — NOT a copy of raw LeetCode HTML.`;

const LC_TAG_TO_CAT = {
    'Array': 'Arrays', 'String': 'Strings', 'Hash Table': 'Hash Maps',
    'Two Pointers': 'Two Pointers', 'Sliding Window': 'Sliding Window',
    'Binary Search': 'Binary Search', 'Sorting': 'Sorting',
    'Stack': 'Stacks', 'Queue': 'Stacks', 'Monotonic Stack': 'Stacks',
    'Recursion': 'Recursion', 'Dynamic Programming': 'Dynamic Programming',
    'Graph': 'Graphs', 'Breadth-First Search': 'Graphs', 'Depth-First Search': 'Graphs',
    'Topological Sort': 'Graphs', 'Tree': 'Binary Trees', 'Binary Tree': 'Binary Trees',
    'Binary Search Tree': 'Binary Trees', 'Linked List': 'Linked Lists',
    'Math': 'Logic', 'Bit Manipulation': 'Logic', 'Greedy': 'Logic',
    'Backtracking': 'Recursion', 'Divide and Conquer': 'Recursion',
    'Heap (Priority Queue)': 'Sorting', 'Matrix': 'Arrays',
    'Simulation': 'Logic', 'Counting': 'Hash Maps', 'Prefix Sum': 'Arrays',
};

const TIER_TO_LC_DIFFICULTY = { 1: 'Easy', 2: 'Medium', 3: 'Hard' };

function guessCat(lcProblem) {
    const tags = (lcProblem.topicTags || []).map(t => t.name || t);
    for (const tag of tags) {
        if (LC_TAG_TO_CAT[tag]) return LC_TAG_TO_CAT[tag];
    }
    return 'Logic';
}

function buildLCConvertPrompt(lcProblem, tier) {
    const desc = stripHtml(lcProblem.content || '');
    const title = lcProblem.title || lcProblem.questionTitle || 'Unknown';
    const difficulty = lcProblem.difficulty || TIER_TO_LC_DIFFICULTY[tier] || 'Medium';
    const tags = (lcProblem.topicTags || []).map(t => t.name || t).join(', ');
    const hints = (lcProblem.hints || []).slice(0, 2).map(stripHtml).join(' | ');
    const suggestedCat = guessCat(lcProblem);

    return `Convert this LeetCode problem for Tier ${tier} (${TIER_TO_LC_DIFFICULTY[tier] || 'Medium'}).

PROBLEM: ${title} | DIFFICULTY: ${difficulty} | TAGS: ${tags}
SUGGESTED CATEGORY: ${suggestedCat}
${hints ? 'HINTS: ' + hints : ''}

DESCRIPTION:
${desc.substring(0, 2500)}

Return this EXACT JSON structure (single object, NOT array):
{
  "title": "${title}",
  "cat": "${suggestedCat}",
  "prompt": "Clear 2-4 sentence description of what to implement, inputs, outputs, constraints.",
  "hint": "Helpful hint without giving away the answer.",
  "concept": "One sentence: what CS concept is tested.",
  "js": {
    "sig": "function camelName(p1, p2) { }",
    "solution": "function camelName(p1, p2) {\\n  // working code\\n  return result;\\n}"
  },
  "py": {
    "sig": "def snake_name(p1, p2):",
    "solution": "def snake_name(p1, p2):\\n    # working code\\n    return result"
  },
  "cpp": {
    "sig": "return_type snake_name(params) { }",
    "solution": "#include <vector>\\nusing namespace std;\\n\\nreturn_type snake_name(params) {\\n    return result;\\n}"
  },
  "tests": [
    {"call": "camelName(input1)", "expect": output1},
    {"call": "camelName(input2)", "expect": output2},
    {"call": "camelName(input3)", "expect": output3}
  ]
}

CRITICAL: The function name in js.sig, js.solution, and every tests[].call MUST be identical.
Use the LeetCode examples as test cases. Exactly 3 tests.`;
}


// Shuffle and pick N random items from array
function pickRandom(arr, n) {
    const shuffled = [...arr].sort(() => Math.random() - 0.5);
    return shuffled.slice(0, n);
}

// ---- GENERATION PROMPTS (AI-ORIGINAL MODE) ----

const GEN_SYSTEM_PROMPT = `You are a coding challenge generator for an interview prep tool. Generate unique, well-tested coding challenges.

CRITICAL RULES:
- Return ONLY valid JSON, no markdown, no backticks, no commentary
- Every test expectation must be EXACTLY correct — verify by mentally running the solution
- Function names must be camelCase in JS, snake_case in Python, snake_case in C++
- Tests must use simple, deterministic inputs with unambiguous outputs
- Solutions must be complete and correct
- Do NOT use any external libraries — only built-in language features
- For Python solutions, do NOT include type hints in the function body
- Keep test inputs small (arrays under 10 elements, numbers under 1000)
- Each challenge must be completely self-contained (no setup code, no helper classes)
- Use only simple types: arrays of ints/strings, strings, numbers, booleans, objects/dicts
- Do NOT generate challenges requiring TreeNode, ListNode, or any class definitions`;

function buildGenPrompt(tier, categories, count, existingTitles) {
    return `Generate ${count} coding challenges for Tier ${tier}.

Difficulty: ${TIER_DIFFICULTY[tier]}
Categories to use (pick from these): ${categories.join(', ')}
Do NOT duplicate these existing titles: ${existingTitles.join(', ')}

Return a JSON array of objects with this EXACT structure:
[
  {
    "title": "Challenge Title",
    "cat": "Category Name",
    "prompt": "Clear description of what to implement. Be specific about inputs, outputs, and edge cases.",
    "hint": "A helpful hint without giving away the solution.",
    "concept": "One sentence about the underlying CS concept.",
    "js": {
      "sig": "function funcName(arg1, arg2) { }",
      "solution": "function funcName(arg1, arg2) {\\n  // complete working solution\\n}"
    },
    "py": {
      "sig": "def func_name(arg1, arg2):",
      "solution": "def func_name(arg1, arg2):\\n    # complete working solution"
    },
    "cpp": {
      "sig": "return_type func_name(param_type param) { }",
      "solution": "return_type func_name(param_type param) {\\n    // complete working solution\\n}"
    },
    "tests": [
      {"call": "funcName(arg1, arg2)", "expect": expected_value},
      {"call": "funcName(arg3, arg4)", "expect": expected_value2},
      {"call": "funcName(arg5, arg6)", "expect": expected_value3}
    ]
  }
]

Each challenge MUST have exactly 3 tests. The "call" field uses the JS function name.
C++ solutions should include necessary headers (vector, string, etc.) and use 'using namespace std;'.
Make challenges creative and practical — not just textbook exercises.`;
}

function validateChallenge(ch) {
    const errors = [];

    // Ensure ch.js exists with sig and solution
    if (!ch.js) {
        // AI may have put sig/solution at top level instead of nested under js
        if (ch.sig && ch.solution) {
            ch.js = { sig: ch.sig, solution: ch.solution };
        } else {
            errors.push('Missing js.sig and js.solution');
            return errors;
        }
    }
    if (!ch.js.sig && ch.js.solution) {
        // Extract function name from solution to build sig
        const fnMatch = ch.js.solution.match(/function\s+(\w+)\s*\(([^)]*)\)/);
        if (fnMatch) {
            ch.js.sig = `function ${fnMatch[1]}(${fnMatch[2]}) { }`;
        }
    }
    if (!ch.js.solution) {
        errors.push('Missing js.solution');
        return errors;
    }

    // Ensure function name in tests matches the solution
    const solFnMatch = ch.js.solution.match(/function\s+(\w+)/);
    if (solFnMatch && ch.tests?.length) {
        const solFnName = solFnMatch[1];
        const testFnMatch = ch.tests[0].call.match(/^(\w+)\(/);
        if (testFnMatch && testFnMatch[1] !== solFnName) {
            // Fix test calls to use the correct function name
            const oldName = testFnMatch[1];
            console.log(`[VALIDATE] Fixing function name: ${oldName} → ${solFnName}`);
            for (const test of ch.tests) {
                test.call = test.call.replace(new RegExp('^' + oldName + '\\('), solFnName + '(');
            }
            // Also fix sig if it uses the wrong name
            if (ch.js.sig) {
                ch.js.sig = ch.js.sig.replace(new RegExp('function\\s+' + oldName), 'function ' + solFnName);
            }
        }
    }

    // Run the JS solution against each test to verify expectations
    try {
        for (let i = 0; i < ch.tests.length; i++) {
            const test = ch.tests[i];
            const full = ch.js.solution + '\n(' + test.call + ')';
            let result;
            try {
                result = eval(full);
            } catch (e) {
                errors.push(`Test ${i}: JS eval error: ${e.message}`);
                continue;
            }
            if (!deepEq(result, test.expect)) {
                // Fix the expectation to match actual result
                console.log(`[VALIDATE] Fixing test ${i} expect: ${JSON.stringify(test.expect)} → ${JSON.stringify(result)}`);
                ch.tests[i].expect = JSON.parse(JSON.stringify(result));
                ch.tests[i]._fixed = true;
            }
        }
    } catch (e) {
        errors.push(`Solution eval error: ${e.message}`);
    }
    return errors;
}

app.post('/codegrind/generate-challenges', async (req, res) => {
    const { tier, count = 5, source = 'ai' } = req.body;

    if (!tier || tier < 1 || tier > 3) {
        return res.status(400).json({ error: 'Valid tier (1-3) required' });
    }

    const hasCL = !!process.env.ANTHROPIC_API_KEY;
    const hasDS = !!process.env.DEEPSEEK_API_KEY;
    if (!hasCL && !hasDS) {
        return res.status(500).json({ error: 'No API keys configured' });
    }

    const actualCount = Math.min(Math.max(count, 3), 10);
    const existingTitles = (req.body.existingTitles || []).slice(0, 30);

    // ---- LEETCODE MODE ----
    if (source === 'leetcode') {
        console.log(`[LC] Fetching ${actualCount} Tier ${tier} LeetCode problems...`);

        try {
            const pool = LC_POOL[tier] || LC_POOL[1];
            const slugs = pickRandom(pool, actualCount + 5); // grab extras in case some fail
            const valid = [];

            for (const slug of slugs) {
                if (valid.length >= actualCount) break;

                // Skip if title already exists
                const titleGuess = slug.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
                if (existingTitles.some(t => t.toLowerCase() === titleGuess.toLowerCase())) continue;

                // Fetch from LeetCode API
                const lcData = await fetchLeetCodeProblem(slug);
                if (!lcData || !lcData.content) {
                    console.log(`[LC] Skipping ${slug}: no content`);
                    continue;
                }

                const problem = lcData;
                problem.titleSlug = problem.titleSlug || slug;

                // Have AI convert to Code Grind format
                const convertPrompt = buildLCConvertPrompt(problem, tier);
                let rawResult;
                try {
                    if (hasCL) rawResult = await callClaude(LC_CONVERT_SYSTEM, convertPrompt);
                    else rawResult = await callDeepSeek(LC_CONVERT_SYSTEM, convertPrompt);
                } catch (e) {
                    console.log(`[LC] AI conversion failed for ${slug}: ${e.message}`);
                    continue;
                }

                // Parse
                const raw = rawResult.text || rawResult;
                let cleaned = raw.trim();
                if (cleaned.startsWith('```')) cleaned = cleaned.replace(/^```(?:json)?\n?/, '').replace(/\n?```$/, '');
                let ch;
                try {
                    ch = JSON.parse(cleaned);
                    if (Array.isArray(ch)) ch = ch[0]; // in case AI returns array
                } catch (e) {
                    const objMatch = cleaned.match(/\{[\s\S]*\}/);
                    if (objMatch) { try { ch = JSON.parse(objMatch[0]); } catch(e2) { continue; } }
                    else continue;
                }

                if (!ch || !ch.tests?.length) continue;
                // Normalize: AI sometimes puts sig/solution at top level
                if (!ch.js && ch.sig && ch.solution) {
                    ch.js = { sig: ch.sig, solution: ch.solution };
                }
                if (!ch.js?.solution) continue;

                // Add LeetCode URL to prompt
                if (ch.leetcode_url || problem.titleSlug) {
                    const url = ch.leetcode_url || `https://leetcode.com/problems/${problem.titleSlug}`;
                    ch.prompt = (ch.prompt || '') + `\n\n🔗 LeetCode: ${url}`;
                }

                // Validate
                const errors = validateChallenge(ch);
                if (errors.length > 0) {
                    console.log(`[LC] Validation issues for "${ch.title}":`, errors.join('; '));
                }
                const allOk = ch.tests.every(t => t.expect !== undefined);
                if (allOk) {
                    ch._source = 'leetcode';
                    valid.push(ch);
                    console.log(`[LC] ✓ ${ch.title}`);
                } else {
                    console.log(`[LC] ✗ ${ch.title} (validation failed)`);
                }
            }

            console.log(`[LC] Validated ${valid.length}/${actualCount} LeetCode challenges`);

            if (valid.length === 0) {
                return res.status(500).json({ error: 'No LeetCode challenges passed validation. Try again or use AI-generated mode.' });
            }

            return res.json({ challenges: valid, tier, source: 'leetcode' });

        } catch (e) {
            console.error('[LC ERROR]', e.message);
            return res.status(500).json({ error: e.message });
        }
    }

    // ---- AI-ORIGINAL MODE ----
    const categories = TIER_CATS[tier];
    console.log(`[GEN] Generating ${actualCount} Tier ${tier} challenges...`);

    try {
        const userPrompt = buildGenPrompt(tier, categories, actualCount, existingTitles);
        let rawResult;
        if (hasCL) rawResult = await callClaude(GEN_SYSTEM_PROMPT, userPrompt);
        else rawResult = await callDeepSeek(GEN_SYSTEM_PROMPT, userPrompt);

        const raw = rawResult.text || rawResult;
        let cleaned = raw.trim();
        if (cleaned.startsWith('```')) cleaned = cleaned.replace(/^```(?:json)?\n?/, '').replace(/\n?```$/, '');

        let challenges;
        try {
            challenges = JSON.parse(cleaned);
        } catch (e) {
            const arrMatch = cleaned.match(/\[[\s\S]*\]/);
            if (arrMatch) challenges = JSON.parse(arrMatch[0]);
            else return res.status(500).json({ error: 'Failed to parse AI response' });
        }

        if (!Array.isArray(challenges)) return res.status(500).json({ error: 'Expected array' });

        const valid = [];
        for (const ch of challenges) {
            if (!ch.title || !ch.tests?.length) continue;
            // Normalize: AI sometimes puts sig/solution at top level
            if (!ch.js && ch.sig && ch.solution) {
                ch.js = { sig: ch.sig, solution: ch.solution };
            }
            if (!ch.js?.solution) continue;
            validateChallenge(ch);
            if (ch.tests.every(t => t.expect !== undefined)) valid.push(ch);
        }

        console.log(`[GEN] Generated ${challenges.length}, validated ${valid.length}`);
        if (valid.length === 0) return res.status(500).json({ error: 'No challenges passed validation.' });

        res.json({ challenges: valid, tier, source: 'ai' });

    } catch (e) {
        console.error('[GEN ERROR]', e.message);
        res.status(500).json({ error: e.message });
    }
});

// ============================================================
// SINGLE LEETCODE CHALLENGE (for dedicated LC slot)
// ============================================================

app.post('/codegrind/leetcode-challenge', async (req, res) => {
    const { tier = 1, exclude = [] } = req.body;
    const t = Math.min(Math.max(tier, 1), 3);

    const hasCL = !!process.env.ANTHROPIC_API_KEY;
    const hasDS = !!process.env.DEEPSEEK_API_KEY;
    if (!hasCL && !hasDS) {
        return res.status(500).json({ error: 'No API keys configured' });
    }

    console.log(`[LC-SINGLE] Fetching Tier ${t} LeetCode problem...`);

    try {
        const pool = LC_POOL[t] || LC_POOL[1];
        // Filter out already-seen slugs
        const available = pool.filter(s => !exclude.includes(s));
        if (available.length === 0) {
            return res.status(404).json({ error: 'All problems in this tier have been used. Reset to try again.' });
        }

        const shuffled = [...available].sort(() => Math.random() - 0.5);

        for (const slug of shuffled.slice(0, 8)) {
            // Fetch from LC API
            const lcData = await fetchLeetCodeProblem(slug);
            if (!lcData || !lcData.content) { console.log(`[LC-SINGLE] Skip ${slug}: fetch failed`); continue; }

            const problem = lcData;
            problem.titleSlug = problem.titleSlug || slug;

            // Convert via AI
            const convertPrompt = buildLCConvertPrompt(problem, t);
            let rawResult;
            try {
                if (hasCL) rawResult = await callClaude(LC_CONVERT_SYSTEM, convertPrompt);
                else rawResult = await callDeepSeek(LC_CONVERT_SYSTEM, convertPrompt);
            } catch (e) {
                console.log(`[LC-SINGLE] AI failed for ${slug}: ${e.message}`);
                continue;
            }

            // Parse JSON
            const raw = rawResult.text || rawResult;
            let cleaned = raw.trim();
            if (cleaned.startsWith('```')) cleaned = cleaned.replace(/^```(?:json)?\n?/, '').replace(/\n?```$/, '');
            let ch;
            try {
                ch = JSON.parse(cleaned);
                if (Array.isArray(ch)) ch = ch[0];
            } catch (e) {
                const m = cleaned.match(/\{[\s\S]*\}/);
                if (m) { try { ch = JSON.parse(m[0]); } catch(e2) { continue; } }
                else continue;
            }

            if (!ch || !ch.tests?.length) continue;
            // Normalize: AI sometimes puts sig/solution at top level
            if (!ch.js && ch.sig && ch.solution) {
                ch.js = { sig: ch.sig, solution: ch.solution };
            }
            if (!ch.js?.solution) continue;

            // Validate by running JS
            const errors = validateChallenge(ch);
            if (errors.length > 0) console.log(`[LC-SINGLE] Fix: ${ch.title}:`, errors.join('; '));

            if (!ch.tests.every(t => t.expect !== undefined)) continue;

            // Add metadata
            ch._source = 'leetcode';
            ch._slug = slug;
            ch._lcUrl = `https://leetcode.com/problems/${slug}`;
            ch._lcDifficulty = problem.difficulty || TIER_TO_LC_DIFFICULTY[t];

            console.log(`[LC-SINGLE] ✓ ${ch.title} (${ch._lcDifficulty})`);
            return res.json({ challenge: ch, tier: t, slug });
        }

        return res.status(500).json({ error: 'Could not fetch/convert a valid problem. Try again.' });

    } catch (e) {
        console.error('[LC-SINGLE ERROR]', e.message);
        res.status(500).json({ error: e.message });
    }
});

// ============================================================
// START
// ============================================================

const PORT = process.env.PORT || 3100;

buildIndex();

app.listen(PORT, '0.0.0.0', () => {
    const ds = process.env.DEEPSEEK_API_KEY ? '✓' : '✗';
    const cl = process.env.ANTHROPIC_API_KEY ? '✓' : '✗';

    console.log(`⚡ Code Grind running on port ${PORT}`);
    console.log(`  Frontend:  http://localhost:${PORT}`);
    console.log(`  API:       POST /codegrind/ask`);
    console.log(`  DeepSeek:  ${ds} (code review/explain/optimize)`);
    console.log(`  Claude:    ${cl} (interview prep)`);
    console.log(`  Knowledge: ${chunks.length} chunks, ${Object.keys(invertedIndex).length} terms`);
    console.log(`  Routing:   DeepSeek for code → Claude for interviews\n`);
});
