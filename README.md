# Code Grind

A coding challenge app with a RAG-grounded AI companion. 64 challenges across 3 tiers — from fundamentals to advanced algorithms and audio DSP — with built-in code editing, test running, and AI-powered code review.

![Node.js](https://img.shields.io/badge/Node.js-20+-green) ![Express](https://img.shields.io/badge/Express-4.x-lightgrey) ![License](https://img.shields.io/badge/License-MIT-blue)

## Features

- **64 Coding Challenges** across 3 tiers:
  - **Tier 1** — Fundamentals (arrays, strings, recursion)
  - **Tier 2** — Intermediate (trees, graphs, dynamic programming)
  - **Tier 3** — Advanced + Audio DSP (convolution, noise gates, RMS metering)
- **Multi-language support** — JavaScript, Python, C++, SQL
- **In-browser code editor** with syntax highlighting and test runner
- **AI Companion** with 4 modes:
  - **Review** — Bug detection, logic errors, edge cases
  - **Explain** — Step-by-step walkthrough with examples
  - **Optimize** — Time/space complexity analysis
  - **Interview Prep** — Technical interview simulation
- **RAG-grounded responses** — BM25 search over a knowledge base ensures AI answers stay factual
- **Dual LLM routing** — DeepSeek for code analysis, Claude for interview prep
- **Progress tracking** with tier unlocking and streak counter
- **LeetCode integration** — Bonus slots for real LeetCode problems

## Quick Start

```bash
git clone https://github.com/Stavion-Colquitt/Code_Grind.git
cd Code_Grind
npm install
cp .env.example .env
# Add your API keys to .env
node server.js
```

Open `http://localhost:3050` in your browser.

## Configuration

Copy `.env.example` to `.env` and add your API keys:

```
DEEPSEEK_API_KEY=your_deepseek_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
PORT=3050
```

- **DeepSeek API Key** — Get one at [platform.deepseek.com](https://platform.deepseek.com)
- **Anthropic API Key** — Get one at [console.anthropic.com](https://console.anthropic.com)

## Docker

```bash
docker build -t codegrind .
docker run -p 3050:3050 --env-file .env codegrind
```

## Architecture

```
codegrind/
├── server.js              # Express server, BM25 search engine, LLM routing
├── public/
│   └── index.html         # Single-page app (editor, tests, AI chat)
├── knowledge/
│   ├── codegrind_challenges_t1.json   # Tier 1 challenges
│   ├── codegrind_challenges_t2.json   # Tier 2 challenges
│   ├── codegrind_challenges_t3.json   # Tier 3 challenges
│   ├── codegrind_patterns.json        # Algorithm pattern reference
│   ├── codegrind_dsp_audio.json       # Audio DSP knowledge base
│   └── codegrind_spear.json           # AI companion system directives
├── Dockerfile
├── .env.example
└── package.json
```

**Zero-framework frontend** — the entire UI is a single HTML file with vanilla JS. No build step, no bundler.

**Two dependencies** — `express` and `dotenv`. That's it.

## License

[MIT](LICENSE)
