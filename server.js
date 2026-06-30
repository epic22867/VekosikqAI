require('dotenv').config();
const express = require('express');
const axios = require('axios');
const multer = require('multer');
const path = require('path');

const app = express();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 8 * 1024 * 1024 } });

const PORT = process.env.PORT || 3000;
const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL || 'http://localhost:11434';
const TEXT_MODEL = process.env.TEXT_MODEL || 'qwen3:8b';
const VISION_MODEL = process.env.VISION_MODEL || 'qwen2.5vl:3b';
const TAVILY_API_KEY = process.env.TAVILY_API_KEY || '';

app.use(express.json({ limit: '10mb' }));
app.use(express.static(path.join(__dirname))); // serves index.html

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

// ---------- Text chat (Qwen3:8b via Ollama) ----------
app.post('/api/chat', async (req, res) => {
  try {
    const { message, useSearch } = req.body;
    if (!message) return res.status(400).json({ error: 'message required' });

    let context = '';
    if (useSearch) {
      try {
        const results = await tavilySearch(message);
        context = results
          .map((r, i) => `[${i + 1}] ${r.title}: ${r.content}`)
          .join('\n');
      } catch (e) {
        context = '';
      }
    }

    const prompt = context
      ? `Use the following web search results to help answer. Cite sources as [1], [2] etc where relevant.\n\n${context}\n\nQuestion: ${message}`
      : message;

    const ollamaRes = await axios.post(`${OLLAMA_BASE_URL}/api/generate`, {
      model: TEXT_MODEL,
      prompt,
      stream: false
    });

    res.json({ reply: ollamaRes.data.response, sources: context ? true : false });
  } catch (err) {
    console.error(err.message);
    res.status(500).json({ error: 'Text generation failed', detail: err.message });
  }
});

// ---------- Vision (Qwen2.5VL:3b via Ollama) ----------
app.post('/api/vision', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'image required' });
    const prompt = req.body.prompt || 'Describe this image in detail.';
    const base64Image = req.file.buffer.toString('base64');

    const ollamaRes = await axios.post(`${OLLAMA_BASE_URL}/api/generate`, {
      model: VISION_MODEL,
      prompt,
      images: [base64Image],
      stream: false
    });

    res.json({ reply: ollamaRes.data.response });
  } catch (err) {
    console.error(err.message);
    res.status(500).json({ error: 'Vision analysis failed', detail: err.message });
  }
});

// ---------- Plain web search endpoint ----------
app.post('/api/search', async (req, res) => {
  try {
    const { query } = req.body;
    if (!query) return res.status(400).json({ error: 'query required' });
    const results = await tavilySearch(query);
    res.json({ results });
  } catch (err) {
    res.status(500).json({ error: 'Search failed', detail: err.message });
  }
});

app.get('/health', (req, res) => res.json({ status: 'ok' }));

app.listen(PORT, () => {
  console.log(`Vekosiq AI running on port ${PORT}`);
  console.log(`Ollama base URL: ${OLLAMA_BASE_URL}`);
});
