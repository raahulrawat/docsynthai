<p align="center">
  <img src="assets/logo.png" width="180" />
</p>

<h1 align="center">DocSynthAI – Intelligent Document Processing MCP Server</h1>

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python->3.11-blue?logo=python&logoColor=white" />
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Platform-MCP-black?logo=openai&logoColor=white" />
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Docs-Available-brightgreen?logo=readthedocs&logoColor=white" />
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/License-MIT-yellow?logo=open-source-initiative&logoColor=white" />
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Status-Active-success?logo=github" />
  </a>
</p>


A modular, extensible, next-generation document understanding engine powered by MCP (Model Context Protocol) and Gemini Vision.

<section class="card">
  <h2>Quick Overview</h2>
  <p>
    <strong>DocSynthAI</strong> is a modular document understanding platform built on MCP. It provides:
  </p>
  <ul>
    <li>AI-powered document classification (Gemini Vision integration)</li>
    <li>Rule-based & general LLM classification modes</li>
    <li>STDIO MCP server and async STDIO client for local/dev integration</li>
    <li>Roadmap: Extraction → Validation → Knowledge Graph creation → HTTP/SSE transport</li>
  </ul>
</section>

<section class="card">
  <h2>Install & Setup</h2>

  <h3>1. Clone</h3>
  <pre><code>git clone https://github.com/raahulrawat/docsynthai.git </code></pre>
  
  <h3>2. Python packages</h3>
  <p>Install dependencies (recommended to use a virtualenv):</p>
  <pre><code>python -m venv .venv </code></pre>

  <section class="card">
  <h2>Run — MCP Server (STDIO) — Current</h2>
  <p>This starts the MCP server in STDIO mode (default/current). Clients connect over stdio pipes.</p>

  <h3>Start server (local)</h3>
  <pre><code>python server.py</code></pre>

  <h3>Start server (explicit stdio mode)</h3>
  <pre><code>DOCSYNTH_TRY_HTTP=0 python server.py</code></pre>

  <p>
    By default the server will load rules from <code>classifier_rules.json</code> if present
    and persist rules to that file. The server exposes the following MCP tools:
  </p>
  <ul>
    <li><code>setup_classifier</code></li>
    <li><code>create_rule</code></li>
    <li><code>get_all_rules</code></li>
    <li><code>delete_rule</code></li>
    <li><code>classify_document</code></li>
  </ul>

  <h3>Running the STDIO client demo</h3>
  <pre><code>python mcp_stdio_client.py</code></pre>
  <p>The demo will: launch the server subprocess, do MCP initialization, ask for API key, and let you classify a local image file.</p>
</section>

<section class="card">
  <h2>Run — HTTP & SSE (Next Release)</h2>
  <p>Planned in the next release:</p>
  <ul>
    <li><strong>HTTP Transport:</strong> <code>mcp.run(transport="http", host="0.0.0.0", port=8000)</code> — REST-like access to tools</li>
    <li><strong>SSE Transport:</strong> streaming support for long-running/extraction tasks</li>
  </ul>
  <p>When HTTP is enabled you will be able to run:</p>
  <pre><code>python server.py   # will detect DOCSYNTH_TRY_HTTP=1 and bind to host/port</code></pre>
  <p>Client libraries will be updated to support HTTP tool discovery and SSE streaming.</p>
</section>

<section class="card">
  <h2>Running MCP Server JSON (STDIO config)</h2>
  <p>Use this sample JSON for external orchestrators or MCP host configs (e.g., Cursor / IDE tool integrations):</p>

  <pre><code>
{
  "mcpServers": {
    "docsynth": {
      "command": "python",
      "args": [
        "server.py"
      ],
      "transport": {
        "type": "stdio"
      },
      "env": {}
    }
  }
}
    
  </code></pre>

  <p>
    Save as <code>.mcp/docsynth-mcp.json</code> or include in your MCP host configuration. This tells an MCP host to spawn <code>server.py</code> and connect via stdio.
  </p>
</section>

<section class="card">
  <h2>Classification Roadmap (current support)</h2>
  <p>Core pipeline stages we implement or plan to implement — each becomes an MCP tool.</p>

  <div class="cols">
    <div>
      <h3>Stage 1 — Classification (current)</h3>
      <ul>
        <li>Rule-based classification (user-defined rules)</li>
        <li>General LLM classification (Gemini Vision)</li>
        <li>Single-image & batch classification</li>
        <li>Strict JSON response format for downstream parsing</li>
      </ul>
    </div></div>
  <div class="cols">
    <div>
      <h3>Stage 2 — Extraction (next)</h3>
      <ul>
        <li>Key–Value pair extraction (KV)</li>
        <li>Table extraction → CSV/JSON</li>
        <li>Multi-page PDF → page images conversion (optional helper)</li>
        <li>Tool: <code>extract_document</code></li>
      </ul>
    </div>
  </div></div>

  <div class="cols" style="margin-top:12px;">
    <div>
      <h3>Stage 3 — Validation</h3>
      <ul>
        <li>Field-level validation (PAN/Aadhaar format, dates, totals)</li>
        <li>Cross-document validation (e.g., PAN ↔ Bank Statement)</li>
        <li>Rule-based & model-assisted validation</li>
        <li>Tool: <code>validate_document</code></li>
      </ul>
    </div></div>
  <div class="cols">
    <div>
      <h3>Stage 4 — Knowledge Graph Creation</h3>
      <ul>
        <li>Triplet extraction (subject, predicate, object)</li>
        <li>Ontology mapping & transformation</li>
        <li>Neo4j / Memgraph integrations</li>
        <li>Tool: <code>kg_insert</code>, <code>kg_generate_triplets</code></li>
      </ul>
    </div>
  </div></div>
</section>
<section class="card">
  <h2>Testing & Development Tips</h2>
  <ul>
    <li>Use <code>DocumentClassifier(mock_mode=True)</code> for fast local tests without Gemini API calls.</li>
    <li>Persist rules in <code>classifier_rules.json</code> to re-use definitions across restarts.</li>
    <li>To test HTTP mode when implemented, set <code>DOCSYNTH_TRY_HTTP=1</code> and pass <code>DOCSYNTH_HOST</code>/<code>DOCSYNTH_PORT</code>.</li>
    <li>Create pytest tests that launch the server subprocess via stdio and call the client (mock_mode recommended).</li>
  </ul>
</section>

<section class="card">
  <h2>Contributing</h2>
  <p>PRs welcome. Suggested first issues:</p>
  <ul>
    <li>HTTP transport adapter & docs</li>
    <li>SSE streaming for long extraction jobs</li>
    <li>PDF→image helper & multi-page handling</li>
    <li>KG connector for Neo4j</li>
  </ul>
  <p>Please follow the code style, add tests, and include changelog entries for breaking changes.</p>
</section>

<footer>
  <div>© DocSynthAI — Built for MCP experimentation and production prototyping</div>
</footer>


