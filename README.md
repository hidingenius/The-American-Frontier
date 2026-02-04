# American History AI

A lightweight, **100% local** AI that discusses the full history of America from pre-colonial times to the present day. International relations are mentioned only in the context of their impact on American events.

Powered by **Ollama** (local LLMs) + **Gradio** (chat UI). No cloud, no keys.

> "Exploring America's past, one question at a time." — Your AI Historian

## Features

- Covers all eras: Pre-colonial indigenous societies, colonial period, Revolution, Civil War, 20th/21st centuries, up to present day
- Strict focus: American events only; redirects off-topic queries
- Educational: Factual explanations, key figures, events, impacts
- Customizable: Tailor to eras, figures, or themes
- Stays on-topic: No non-American history

## Quick Start

### Prerequisites

1. Install [**Ollama**](https://ollama.com)
2. Pull a model:
   ```bash
   ollama pull llama3.2:3b  # Fast/small
   # Or: ollama pull llama3.1:8b for more detailed responses
