require('dotenv').config();
const express = require('express');
const axios = require('axios');
const multer = require('multer');
const path = require('path');
const Database = require('better-sqlite3');

const app = express();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 8 * 1024 * 1024 } });

const PORT = process.env.PORT || 3000;
const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL || 'http://localhost:11434';
const TEXT_MODEL = process.env.TEXT_MODEL || 'qwen3:8b';
const VISION_MODEL = process.env.VISION_MODEL || 'qwen2.5vl:3b';
const TAVILY_API_KEY = process.env.TAVILY_API_KEY || '';

// Max characters allowed in a single user message/prompt, to keep requests small
// and predictable for the local Ollama server.
const MAX_MESSAGE_LENGTH = 1096;

// Max number of requests allowed to wait in the queue before we start rejecting
// new ones outright (protects against unbounded memory growth under heavy load).
const MAX_QUEUE_SIZE = parseInt(process.env.MAX_QUEUE_SIZE || '20', 10);

// How many requests are allowed to hit Ollama at the same time. Local LLM servers
// generally do NOT benefit from concurrent requests (they just contend for the
// same GPU/CPU), so we default to 1 — i.e. requests are queued and processed
// one-by-one, in order ("orders"), instead of overloading the server.
const OLLAMA_CONCURRENCY = parseInt(process.env.OLLAMA_CONCURRENCY || '1', 10);

// ---------- SQLite persistence ----------
// Chats survive server restarts and page reloads. A single file DB is fine here
// since the whole app is already serialized through the request queue above.
const DB_PATH = process.env.DB_PATH || path.join(__dirname, 'data', 'vekosiq.db');
require('fs').mkdirSync(path.dirname(DB_PATH), { recursive: true });
const db = new Database(DB_PATH);
db.pragma('journal_mode = WAL');

db.exec(`
  CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT 'New chat',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
  );
  CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources TEXT,
    created_at INTEGER NOT NULL
  );
  CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
`);

const stmts = {
  insertConv: db.prepare('INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)'),
  touchConv: db.prepare('UPDATE conversations SET updated_at = ? WHERE id = ?'),
  renameConv: db.prepare('UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?'),
  getConv: db.prepare('SELECT * FROM conversations WHERE id = ?'),
  listConvs: db.prepare('SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC'),
  deleteConv: db.prepare('DELETE FROM conversations WHERE id = ?'),
  insertMsg: db.prepare('INSERT INTO messages (conversation_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)'),
  listMsgs: db.prepare('SELECT role, content, sources, created_at FROM messages WHERE conversation_id = ? ORDER BY id ASC'),
};

function ensureConversation(id) {
  let conv = stmts.getConv.get(id);
  if (!conv) {
    const now = Date.now();
    stmts.insertConv.run(id, 'New chat', now, now);
    conv = stmts.getConv.get(id);
  }
  return conv;
}

function saveMessage(conversationId, role, content, sources) {
  stmts.insertMsg.run(conversationId, role, content, sources ? JSON.stringify(sources) : null, Date.now());
  stmts.touchConv.run(Date.now(), conversationId);
}

function maybeAutoTitle(conversationId, firstUserText) {
  const conv = stmts.getConv.get(conversationId);
  if (conv && conv.title === 'New chat' && firstUserText) {
    stmts.renameConv.run(firstUserText.slice(0, 32), Date.now(), conversationId);
  }
}

app.use(express.json({ limit: '10mb' }));
app.use(express.static(path.join(__dirname))); // serves index.html

// ---------- Simple in-memory request queue ----------
// Ensures at most OLLAMA_CONCURRENCY jobs run against Ollama at once. Extra jobs
// wait in FIFO order ("orders") instead of all firing at the same time and
// overloading the server. Each call to enqueue(fn) returns a promise that
// resolves/rejects with whatever fn() resolves/rejects with.
let activeJobs = 0;
const queue = [];

function runNext() {
  if (activeJobs >= OLLAMA_CONCURRENCY) return;
  const job = queue.shift();
  if (!job) return;
  activeJobs++;
  job.fn()
    .then(job.resolve, job.reject)
    .finally(() => {
      activeJobs--;
      runNext();
    });
}

function enqueue(fn) {
  if (queue.length >= MAX_QUEUE_SIZE) {
    return Promise.reject(new Error('Server is busy, please try again shortly.'));
  }
  return new Promise((resolve, reject) => {
    queue.push({ fn, resolve, reject });
    runNext();
  });
}

// Validates a piece of user-supplied text against MAX_MESSAGE_LENGTH.
// Returns an error string if invalid, otherwise null.
function checkLength(text, label = 'message') {
  if (typeof text !== 'string') return null; // let the required-field check handle this
  if (text.length > MAX_MESSAGE_LENGTH) {
    return `${label} is too long (${text.length} characters). Limit is ${MAX_MESSAGE_LENGTH} characters.`;
  }
  return null;
}

// Pulls the real error message out of an axios error, including the
// upstream Ollama/Tavily response body if present, instead of just
// the generic "Request failed with status code 500".
function extractErrorDetail(err) {
  if (err.response && err.response.data) {
    const data = err.response.data;
    if (typeof data === 'string') return data;
    if (data.error) return typeof data.error === 'string' ? data.error : JSON.stringify(data.error);
    return JSON.stringify(data);
  }
  return err.message;
}

// ---------- Tavily web search ----------
async function tavilySearch(query) {
  if (!TAVILY_API_KEY) throw new Error('TAVILY_API_KEY not set');
  const res = await axios.post('https://api.tavily.com/search', {
    api_key: TAVILY_API_KEY,
    query,
    search_depth: 'basic',
    max_results: 5
  });
  return res.data.results || [];
}

// ============================================================
// ---------- Conversation CRUD (persisted in SQLite) ----------
// ============================================================

// List all conversations (sidebar history), most recently updated first.
app.get('/api/conversations', (req, res) => {
  const rows = stmts.listConvs.all();
  res.json({ conversations: rows });
});

// Create a new (empty) conversation and return its id.
app.post('/api/conversations', (req, res) => {
  const id = 'c_' + Date.now() + '_' + Math.random().toString(36).slice(2, 9);
  const now = Date.now();
  stmts.insertConv.run(id, 'New chat', now, now);
  res.json({ id, title: 'New chat', created_at: now, updated_at: now });
});

// Fetch a single conversation with its full message history.
app.get('/api/conversations/:id', (req, res) => {
  const conv = stmts.getConv.get(req.params.id);
  if (!conv) return res.status(404).json({ error: 'Conversation not found' });
  const rows = stmts.listMsgs.all(req.params.id);
  const messages = rows.map(r => ({
    role: r.role,
    content: r.content,
    sources: r.sources ? JSON.parse(r.sources) : null,
    created_at: r.created_at
  }));
  res.json({ id: conv.id, title: conv.title, messages });
});

// Delete a conversation and all of its messages.
app.delete('/api/conversations/:id', (req, res) => {
  stmts.deleteConv.run(req.params.id);
  res.json({ ok: true });
});

// ---------- Text chat (Qwen3:8b via Ollama) — supports multi-turn history ----------
app.post('/api/chat', async (req, res) => {
  try {
    const { message, conversationId, useSearch } = req.body;
    if (!message) return res.status(400).json({ error: 'message required' });
    if (!conversationId) return res.status(400).json({ error: 'conversationId required' });

    const lengthError = checkLength(message, 'Message');
    if (lengthError) return res.status(400).json({ error: lengthError });

    ensureConversation(conversationId);

    // Pull prior turns from the DB instead of trusting the client to send them.
    const priorRows = stmts.listMsgs.all(conversationId);
    const messages = priorRows.map(r => ({ role: r.role, content: r.content }));

    let userContent = message;
    let usedSources = [];

    if (useSearch) {
      try {
        const results = await tavilySearch(message);
        usedSources = results.map(r => ({ title: r.title, url: r.url }));
        const context = results
          .map((r, i) => `[${i + 1}] ${r.title}: ${r.content}`)
          .join('\n');
        userContent = `Use the following web search results to help answer. Cite sources as [1], [2] etc where relevant.\n\n${context}\n\nQuestion: ${message}`;
      } catch (e) {
        // search failed silently, fall back to plain message
      }
    }

    messages.push({ role: 'user', content: userContent });

    const ollamaRes = await enqueue(() => axios.post(`${OLLAMA_BASE_URL}/api/chat`, {
      model: TEXT_MODEL,
      messages,
      stream: false
    }));

    const reply = ollamaRes.data.message?.content || '';

    // Persist the turn (store the raw user message, not the search-augmented one,
    // so re-loading the chat looks the same as what the user actually typed).
    maybeAutoTitle(conversationId, message);
    saveMessage(conversationId, 'user', message, null);
    saveMessage(conversationId, 'assistant', reply, usedSources.length ? usedSources : null);

    const conv = stmts.getConv.get(conversationId);
    res.json({ reply, sources: usedSources, title: conv.title });
  } catch (err) {
    const detail = extractErrorDetail(err);
    console.error('Chat error:', detail);
    res.status(503).json({ error: 'Text generation failed', detail });
  }
});

// ---------- Vision (Qwen2.5VL:3b via Ollama) ----------
// Step 1: vision model ONLY produces a neutral, factual description of the image (it never
// sees the user's actual question).
// Step 2: text model (Qwen3:8b) takes that description + the user's prompt/question and
// generates the real answer, just like a normal chat turn.
app.post('/api/vision', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'image required' });
    const userPrompt = req.body.prompt || 'Describe this image in detail.';
    const conversationId = req.body.conversationId;
    if (!conversationId) return res.status(400).json({ error: 'conversationId required' });

    const lengthError = checkLength(userPrompt, 'Prompt');
    if (lengthError) return res.status(400).json({ error: lengthError });

    ensureConversation(conversationId);

    const base64Image = req.file.buffer.toString('base64');

    // 1) Plain description from the vision model only.
    const visionRes = await enqueue(() => axios.post(`${OLLAMA_BASE_URL}/api/generate`, {
      model: VISION_MODEL,
      prompt: 'Describe exactly what is visible in this image in detail and objectively. Do not answer any question, do not speculate beyond what is visible — just describe the image.',
      images: [base64Image],
      stream: false
    }));

    const imageDescription = visionRes.data.response || '';

    // 2) Text model generates the actual answer using that description + the user's request.
    const composedMessage =
      `Here is a description of an image, generated by a vision model:\n\n` +
      `"""\n${imageDescription}\n"""\n\n` +
      `Using that description, respond to the user's request below.\n\n` +
      `User's request: ${userPrompt}`;

    const ollamaRes = await enqueue(() => axios.post(`${OLLAMA_BASE_URL}/api/chat`, {
      model: TEXT_MODEL,
      messages: [{ role: 'user', content: composedMessage }],
      stream: false
    }));

    const reply = ollamaRes.data.message?.content || '';

    maybeAutoTitle(conversationId, userPrompt);
    saveMessage(conversationId, 'user', userPrompt, null);
    saveMessage(conversationId, 'assistant', reply, null);

    const conv = stmts.getConv.get(conversationId);
    res.json({ reply, imageDescription, title: conv.title });
  } catch (err) {
    const detail = extractErrorDetail(err);
    console.error('Vision error:', detail);
    res.status(503).json({ error: 'Vision analysis failed', detail });
  }
});

// ---------- Plain web search endpoint ----------
app.post('/api/search', async (req, res) => {
  try {
    const { query } = req.body;
    if (!query) return res.status(400).json({ error: 'query required' });

    const lengthError = checkLength(query, 'Query');
    if (lengthError) return res.status(400).json({ error: lengthError });

    const results = await enqueue(() => tavilySearch(query));
    res.json({ results });
  } catch (err) {
    res.status(503).json({ error: 'Search failed', detail: err.message });
  }
});

app.get('/health', (req, res) => res.json({
  status: 'ok',
  queue: { waiting: queue.length, active: activeJobs, concurrency: OLLAMA_CONCURRENCY },
  maxMessageLength: MAX_MESSAGE_LENGTH
}));

app.listen(PORT, () => {
  console.log(`Vekosiq AI running on port ${PORT}`);
  console.log(`Ollama base URL: ${OLLAMA_BASE_URL}`);
  console.log(`SQLite DB: ${DB_PATH}`);
});
