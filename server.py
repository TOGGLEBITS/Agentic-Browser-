import asyncio
import json
import base64
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from playwright.async_api import async_playwright
from google import genai
from google.genai import types

app = FastAPI()
client = genai.Client(api_key="AIzaSyC-dpTnMXerQNaKS6lgxQqtPKyN0_h8ci8")

SYSTEM_PROMPT = """
You are an autonomous browser agent. You will receive a screenshot and a simplified DOM structure of the current web page, along with the user's objective.
Analyze the screen and decide the single next action to take.

Respond ONLY with valid JSON in this exact format:
{
    "thought": "Brief explanation of your reasoning.",
    "action": "click" | "type" | "navigate" | "done",
    "target": "CSS selector (leave empty if navigating/done)",
    "value": "Text to type or URL (leave empty if clicking/done)"
}
"""

async def extract_dom(page):
    script = """
        () => {
            const elements = document.querySelectorAll('a, button, input, textarea, [role="button"]');
            return Array.from(elements).map(el => {
                return `<${el.tagName.toLowerCase()} class="${el.className}" id="${el.id}" name="${el.name}">${el.innerText.trim()}</${el.tagName.toLowerCase()}>`;
            }).join('\\n');
        }
    """
    return await page.evaluate(script)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        data = await websocket.receive_text()
        payload = json.loads(data)
        objective = payload.get("objective")
        start_url = payload.get("url")
        
        await websocket.send_json({"type": "log", "msg": f"🚀 Initializing Agent. Objective: {objective}"})

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True) # Set to False if you want to watch the browser pop up
            context = await browser.new_context(viewport={'width': 1280, 'height': 800})
            page = await context.new_page()
            
            await websocket.send_json({"type": "log", "msg": f"🌐 Navigating to {start_url}..."})
            await page.goto(start_url)
            await page.wait_for_load_state("networkidle")

            step_count = 0
            max_steps = 15

            while step_count < max_steps:
                step_count += 1
                await websocket.send_json({"type": "log", "msg": f"\n--- Step {step_count} ---"})
                
                # 1. OBSERVE
                await websocket.send_json({"type": "log", "msg": "👁️ Observing page state..."})
                screenshot_bytes = await page.screenshot(type='jpeg', quality=70)
                b64_screenshot = base64.b64encode(screenshot_bytes).decode('utf-8')
                dom_text = await extract_dom(page)
                
                # Send screenshot to frontend
                await websocket.send_json({"type": "image", "data": b64_screenshot})

                # 2. THINK
                await websocket.send_json({"type": "log", "msg": "🧠 Thinking (Gemini 2.5 Flash)..."})
                image_part = types.Part.from_bytes(data=screenshot_bytes, mime_type='image/jpeg')
                prompt_text = f"Objective: {objective}\n\nInteractive DOM:\n{dom_text}\n\nWhat is the next action?"
                
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[types.Content(role="user", parts=[types.Part.from_text(SYSTEM_PROMPT), image_part, types.Part.from_text(prompt_text)])],
                    config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0)
                )

                try:
                    decision = json.loads(response.text)
                    await websocket.send_json({"type": "log", "msg": f"💡 Thought: {decision['thought']}"})
                    await websocket.send_json({"type": "action", "msg": f"⚡ Action: {decision['action']} | Target: {decision['target']}"})
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "msg": "❌ Invalid JSON from AI. Retrying..."})
                    continue

                # 3. ACT
                action = decision.get("action")
                try:
                    if action == "navigate":
                        await page.goto(decision["value"])
                    elif action == "click":
                        await page.wait_for_selector(decision["target"], timeout=5000)
                        await page.click(decision["target"])
                    elif action == "type":
                        await page.wait_for_selector(decision["target"], timeout=5000)
                        await page.fill(decision["target"], decision["value"])
                    elif action == "done":
                        await websocket.send_json({"type": "success", "msg": "✅ Objective achieved!"})
                        break
                    
                    await page.wait_for_load_state("networkidle")
                    await asyncio.sleep(1) # Small pause for stability
                    
                except Exception as e:
                    await websocket.send_json({"type": "error", "msg": f"❌ Execution failed: {str(e)}"})

            await browser.close()
            await websocket.send_json({"type": "log", "msg": "🛑 Agent task finished."})

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Server error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)