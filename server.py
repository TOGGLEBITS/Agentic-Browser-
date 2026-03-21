import asyncio
import json
import os
import urllib.parse
from typing import Optional

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from playwright.async_api import async_playwright

# ─── Configuration ──────────────────────────────────────────────────────────

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY environment variable.")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "stepfun/step-3.5-flash:free"

app = FastAPI(title="Antigravity Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── System Prompt ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an ultra-fast intent extraction AI.

Your only job is to analyze the user's objective and extract the core intent into a structured JSON object.

Supported actions:
- search
- search_and_play
- open_site
- fill_form

Return ONLY valid JSON:
{
"site": "youtube | google | github | etc",
"action": "search | search_and_play | open_site | fill_form",
"query": "the search term or relevant input"
}
"""

# ─── JSON Parser ───────────────────────────────────────────────────────────

def safe_parse(output: str) -> dict | None:
    if not output: return None
    try:
        return json.loads(output)
    except:
        output = output.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(output)
        except:
            return None


# ─── Intent Extractor ──────────────────────────────────────────────────────

async def extract_intent(objective: str) -> dict | None:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Antigravity Agent",
    }
    
    payload = {
        "model": MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Objective: {objective}"},
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as http_client:
            response = await http_client.post(OPENROUTER_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return safe_parse(data["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"[extract_intent] Error: {e}")
        return None


# ─── Fast Playwright Handlers ──────────────────────────────────────────────

async def execute_task(ws: WebSocket, page, intent: dict, speed: str):
    site = intent.get("site", "").lower()
    action = intent.get("action", "")
    query = intent.get("query", "")
    
    await ws.send_json({"type": "status", "step": 1, "msg": f"⚡ Executing {action} on {site}"})
    
    try:
        # ── YouTube Handler ──
        if site == "youtube":
            if action in ["search", "search_and_play"]:
                url_query = urllib.parse.quote_plus(query)
                await page.goto(f"https://www.youtube.com/results?search_query={url_query}", wait_until="domcontentloaded", timeout=15000)
                await ws.send_json({"type": "observation", "step": 1, "msg": f"👁️ Loaded YouTube search for '{query}'"})
                
                if action == "search_and_play":
                    await page.wait_for_selector('ytd-video-renderer a#thumbnail', state="visible", timeout=10000)
                    await page.click('ytd-video-renderer a#thumbnail')
                    await ws.send_json({"type": "observation", "step": 1, "msg": "🎥 Playing first video result"})
            else:
                await page.goto("https://www.youtube.com", wait_until="domcontentloaded", timeout=15000)
                
        # ── Google Handler ──
        elif site == "google":
            if action == "search":
                url_query = urllib.parse.quote_plus(query)
                await page.goto(f"https://www.google.com/search?q={url_query}", wait_until="domcontentloaded", timeout=15000)
                await ws.send_json({"type": "observation", "step": 1, "msg": f"👁️ Loaded Google search for '{query}'"})
            else:
                await page.goto("https://www.google.com", wait_until="domcontentloaded", timeout=15000)
                
        # ── Github Handler ──
        elif site == "github":
            if action == "search":
                url_query = urllib.parse.quote_plus(query)
                await page.goto(f"https://github.com/search?q={url_query}&type=repositories", wait_until="domcontentloaded", timeout=15000)
                await ws.send_json({"type": "observation", "step": 1, "msg": f"👁️ Loaded Github search for '{query}'"})
            else:
                await page.goto("https://github.com", wait_until="domcontentloaded", timeout=15000)
                
        # ── Generic Site Handler ──
        else:
            base_url = f"https://www.{site}.com" if "." not in site else f"https://{site}"
            if action == "search":
                # Fallback to loading the site and attempting a generic search box
                await page.goto(base_url, wait_until="domcontentloaded", timeout=15000)
                try:
                    await page.wait_for_selector('input[type="search"], input[name="q"], input[placeholder*="search" i]', state="visible", timeout=3000)
                    await page.fill('input[type="search"], input[name="q"], input[placeholder*="search" i]', query)
                    await page.keyboard.press("Enter")
                    await ws.send_json({"type": "observation", "step": 1, "msg": f"👁️ Attempted generic search on {site}"})
                except Exception:
                    await ws.send_json({"type": "error", "step": 1, "msg": f"⚠️ Could not find generic search field on {site}"})
            else:
                await page.goto(base_url, wait_until="domcontentloaded", timeout=15000)
                await ws.send_json({"type": "observation", "step": 1, "msg": f"👁️ Navigated to {base_url}"})
                
        await ws.send_json({"type": "success", "step": 1, "msg": "✅ Task Completed Successfully!"})
        
    except Exception as e:
        await ws.send_json({"type": "error", "step": 1, "msg": f"❌ Execution Failed: {str(e)[:100]}"})


# ─── WebSocket Endpoint ───────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    browser = None
    context = None
    page = None

    try:
        raw = await websocket.receive_text()
        payload = json.loads(raw)

        objective = payload.get("objective", "")
        speed = payload.get("speed", "normal")

        if not objective:
            await websocket.send_json({"type": "error", "step": 0, "msg": "❌ Missing target objective."})
            return

        await websocket.send_json({"type": "status", "step": 0, "msg": f"🚀 Start: {objective}"})
        await websocket.send_json({"type": "status", "step": 0, "msg": "🧠 Thinking (Analyzing Intent)..."})

        # STEP 1: AI PLANNING (ONLY ONCE)
        intent = await extract_intent(objective)
        
        if not intent:
            await websocket.send_json({"type": "error", "step": 0, "msg": "❌ AI failed to extract intent. Retrying once..."})
            intent = await extract_intent(objective)
            if not intent:
                await websocket.send_json({"type": "error", "step": 0, "msg": "❌ Critical AI Failure. Aborting."})
                return
                
        await websocket.send_json({"type": "thinking", "step": 0, "msg": f"Intent Decoded: {json.dumps(intent)}"})

        # STEP 2: DIRECT EXECUTION
        pw = await async_playwright().start()
        browser = await pw.chromium.launch(headless=False)
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await context.new_page()
        
        # Execute pure playwright handlers
        await execute_task(websocket, page, intent, speed)
        
        await websocket.send_json({"type": "status", "step": 1, "msg": "🛑 Finished."})
        
        # Keep browser open locally for verification
        await asyncio.sleep(300)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "step": 0, "msg": f"❌ Unhandled Error: {str(e)[:100]}"})
        except:
            pass
    finally:
        if browser:
            try:
                await browser.close()
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)