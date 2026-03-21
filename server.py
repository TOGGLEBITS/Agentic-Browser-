"""
Antigravity Agentic Browser — Modular Backend Engine
=====================================================
FastAPI + WebSocket + Playwright + OpenRouter (Nemotron)

Architecture:
  AgentMemory   — Rolling window of last 5 actions + context
  AgentPlanner  — High-level plan generation & partial replanning
  AgentExecutor — Playwright browser actions with retry & highlighting
  AgentLoop     — Orchestrator: Observe → Think → Act → Update → Check
"""

import asyncio
import json
import os
from typing import Optional

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from playwright.async_api import async_playwright

# ─── Configuration ──────────────────────────────────────────────────────────

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY environment variable. Set it before running.")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "nvidia/nemotron-3-super-120b-a12b:free"
MAX_STEPS = 20
MAX_MEMORY = 5
MAX_DOM_ELEMENTS = 80
MAX_TEXT_LENGTH = 40
STUCK_THRESHOLD = 3

app = FastAPI(title="Antigravity Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── System Prompt ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an autonomous browser agent called Antigravity. You control a real browser to accomplish the user's objective.

You receive:
- The user's OBJECTIVE
- A filtered list of interactive DOM elements (indexed)
- Your MEMORY of previous actions and results
- Your current PLAN (if any)

## Rules
1. Think step-by-step before deciding an action.
2. Return exactly ONE action per response.
3. NEVER hallucinate CSS selectors. Only use selectors derived from the DOM provided.
4. Prefer selectors in this order: id > name > aria-label > visible text > class.
5. Avoid long compound class selectors — they are fragile.
6. Use "done" when the objective is clearly achieved.
7. Use "clarify" if the objective is ambiguous and you cannot proceed.
8. Use "scroll" to reveal more content if needed elements are not visible.
9. After typing into a search/input field, ALWAYS prefer "press_enter" over clicking a search button. This is more reliable.
10. If a previous click action failed, try "press_enter" or a different selector as fallback.

## Response Format
Respond ONLY with valid JSON. No extra text before or after the JSON object.
{
    "thought": "clear reasoning in 1-2 lines",
    "plan": ["remaining step 1", "remaining step 2"],
    "action": "click | type | press_enter | navigate | scroll | done | clarify",
    "target": "CSS selector or empty string",
    "value": "text to type, URL to navigate, scroll direction, or empty string"
}"""


# ─── OpenRouter Model Caller ───────────────────────────────────────────────

async def call_model(system_prompt: str, user_prompt: str) -> dict | None:
    """Call OpenRouter API and return parsed JSON response."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Antigravity Agent",
    }

    payload = {
        "model": MODEL_NAME,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=60.0) as http_client:
                response = await http_client.post(OPENROUTER_URL, headers=headers, json=payload)
                response.raise_for_status()

                data = response.json()
                content = data["choices"][0]["message"]["content"]

                # Strip markdown code fences if model wraps output
                content = content.strip()
                if content.startswith("```"):
                    lines = content.split("\n")
                    lines = [l for l in lines if not l.strip().startswith("```")]
                    content = "\n".join(lines).strip()

                return json.loads(content)

        except json.JSONDecodeError:
            if attempt == 0:
                print(f"[call_model] JSON parse failed (attempt {attempt+1}), retrying...")
                continue
            print(f"[call_model] JSON parse failed after retry. Raw: {content[:200] if 'content' in dir() else 'N/A'}")
            return None
        except httpx.HTTPStatusError as e:
            print(f"[call_model] HTTP error: {e.response.status_code} — {e.response.text[:200]}")
            return None
        except Exception as e:
            print(f"[call_model] Error: {e}")
            if attempt == 0:
                await asyncio.sleep(1)
                continue
            return None

    return None


# ─── Agent Memory ───────────────────────────────────────────────────────────

class AgentMemory:
    """Rolling window of recent actions for context."""

    def __init__(self, max_size: int = MAX_MEMORY):
        self.entries: list[dict] = []
        self.max_size = max_size

    def add(self, step: int, thought: str, action: str, target: str, value: str, result: str):
        self.entries.append({
            "step": step,
            "thought": thought[:80],
            "action": action,
            "target": target,
            "value": value[:50],
            "result": result[:60],
        })
        if len(self.entries) > self.max_size:
            self.entries.pop(0)

    def to_prompt(self) -> str:
        if not self.entries:
            return "No previous actions."
        lines = []
        for e in self.entries:
            lines.append(
                f"  Step {e['step']}: {e['action']}(\"{e['target']}\", \"{e['value']}\") → {e['result']}"
            )
        return "\n".join(lines)

    def get_last_n_actions(self, n: int = 3) -> list[tuple[str, str]]:
        return [(e["action"], e["target"]) for e in self.entries[-n:]]

    def clear(self):
        self.entries.clear()


# ─── Agent Planner ──────────────────────────────────────────────────────────

class AgentPlanner:
    """Generates and updates high-level plans."""

    def __init__(self):
        self.current_plan: list[str] = []

    async def generate_initial_plan(self, objective: str, dom_text: str, ws: WebSocket) -> list[str]:
        await ws.send_json({"type": "status", "step": 0, "msg": "📋 Generating execution plan..."})

        plan_system = "You are a planning assistant. Given an objective and a page DOM, generate a short plan (3-5 steps). Respond ONLY with a JSON array of strings. No extra text."
        plan_user = f'Objective: "{objective}"\n\nCurrent page DOM:\n{dom_text[:1500]}\n\nGenerate a plan as a JSON array.'

        try:
            result = await call_model(plan_system, plan_user)
            if isinstance(result, list):
                self.current_plan = result
            elif isinstance(result, dict) and "plan" in result:
                self.current_plan = result["plan"]
            else:
                self.current_plan = [f"Navigate and complete: {objective}"]
            await ws.send_json({"type": "plan", "step": 0, "msg": self.current_plan})
            return self.current_plan
        except Exception:
            self.current_plan = [f"Navigate and complete: {objective}"]
            await ws.send_json({"type": "plan", "step": 0, "msg": self.current_plan})
            return self.current_plan

    async def partial_replan(self, objective: str, memory_text: str, dom_text: str, ws: WebSocket, step: int) -> list[str]:
        await ws.send_json({"type": "status", "step": step, "msg": "🔄 Replanning based on current state..."})

        plan_system = "You are a planning assistant. Generate a revised plan (3-5 steps) for the REMAINING work. Respond ONLY with a JSON array of strings."
        plan_user = f'Objective: "{objective}"\n\nPrevious actions:\n{memory_text}\n\nCurrent page DOM:\n{dom_text[:1500]}\n\nGenerate a revised plan.'

        try:
            result = await call_model(plan_system, plan_user)
            if isinstance(result, list):
                self.current_plan = result
            elif isinstance(result, dict) and "plan" in result:
                self.current_plan = result["plan"]
            await ws.send_json({"type": "plan", "step": step, "msg": self.current_plan})
            return self.current_plan
        except Exception:
            return self.current_plan

    def update_from_response(self, plan: list[str]):
        if plan:
            self.current_plan = plan

    def plan_text(self) -> str:
        if not self.current_plan:
            return "No plan yet."
        return "\n".join(f"  {i+1}. {s}" for i, s in enumerate(self.current_plan))


# ─── CAPTCHA Detection ─────────────────────────────────────────────────────

async def detect_captcha(page) -> bool:
    """Check if a CAPTCHA is present on the page."""
    try:
        captcha_indicators = await page.evaluate("""
            () => {
                const html = document.body.innerText.toLowerCase();
                const hasCaptchaText = html.includes('captcha') || html.includes('i\'m not a robot') || html.includes('verify you are human') || html.includes('unusual traffic');
                const hasRecaptcha = !!document.querySelector('iframe[src*="recaptcha"], iframe[title*="recaptcha"], .g-recaptcha, #captcha, [class*="captcha"]');
                return hasCaptchaText || hasRecaptcha;
            }
        """)
        return captcha_indicators
    except Exception:
        return False


# ─── Agent Executor ─────────────────────────────────────────────────────────

class AgentExecutor:
    """Executes browser actions with retry, highlighting, fallback, and CAPTCHA awareness."""

    def __init__(self, page):
        self.page = page

    async def highlight_element(self, selector: str):
        try:
            await self.page.evaluate("""
                (sel) => {
                    const el = document.querySelector(sel);
                    if (el) {
                        el.style.outline = '3px solid #ff4444';
                        el.style.outlineOffset = '2px';
                        setTimeout(() => {
                            el.style.outline = '';
                            el.style.outlineOffset = '';
                        }, 1200);
                    }
                }
            """, selector)
        except Exception:
            pass

    async def execute_action(self, decision: dict, ws: WebSocket, step: int) -> str:
        """Execute a single action with retry + fallback logic."""
        action = decision.get("action", "")
        target = decision.get("target", "")
        value = decision.get("value", "")

        for attempt in range(2):
            try:
                if action == "navigate":
                    await self.page.goto(value, wait_until="domcontentloaded", timeout=15000)
                    await self._safe_wait()
                    return f"Navigated to {value}"

                elif action == "click":
                    try:
                        await self.page.wait_for_selector(target, timeout=4000, state="visible")
                        await self.highlight_element(target)
                        await self.page.click(target)
                    except Exception:
                        # ── Fallback: try pressing Enter instead ──
                        await ws.send_json({
                            "type": "status", "step": step,
                            "msg": f"⚠️ Click fallback: pressing Enter instead"
                        })
                        await self.page.keyboard.press("Enter")
                    await self._safe_wait()
                    return f"Clicked {target}"

                elif action == "type":
                    await self.page.wait_for_selector(target, timeout=4000, state="visible")
                    await self.highlight_element(target)
                    await self.page.click(target)
                    await self.page.fill(target, "")
                    await self.page.fill(target, value)
                    return f"Typed '{value}' into {target}"

                elif action == "press_enter":
                    await self.page.keyboard.press("Enter")
                    await self._safe_wait()
                    return "Pressed Enter"

                elif action == "scroll":
                    direction = value.lower() if value else "down"
                    delta = -500 if direction == "up" else 500
                    await self.page.evaluate(f"window.scrollBy(0, {delta})")
                    await asyncio.sleep(0.2)
                    return f"Scrolled {direction}"

                elif action == "done":
                    return "DONE"

                elif action == "clarify":
                    return f"CLARIFY: {value}"

                else:
                    return f"Unknown action: {action}"

            except Exception as e:
                error_msg = str(e)[:120]
                if attempt == 0:
                    await ws.send_json({
                        "type": "error", "step": step,
                        "msg": f"⚠️ Action failed (attempt 1): {error_msg}. Retrying..."
                    })
                    await asyncio.sleep(0.3)
                else:
                    return f"FAILED after retry: {error_msg}"

        return "FAILED: unknown error"

    async def _safe_wait(self):
        """Fast page load wait — domcontentloaded with short timeout."""
        try:
            await self.page.wait_for_load_state("domcontentloaded", timeout=3000)
        except Exception:
            pass
        await asyncio.sleep(0.2)


# ─── DOM Extraction ─────────────────────────────────────────────────────────

async def extract_dom(page, max_elements: int = MAX_DOM_ELEMENTS, max_text: int = MAX_TEXT_LENGTH) -> str:
    script = f"""
        () => {{
            const selectors = 'a, button, input, textarea, select, [role="button"], [role="link"], [role="tab"], [role="menuitem"], [contenteditable="true"]';
            const elements = document.querySelectorAll(selectors);
            const results = [];
            let idx = 0;
            for (const el of elements) {{
                if (idx >= {max_elements}) break;
                const rect = el.getBoundingClientRect();
                if (rect.width === 0 && rect.height === 0) continue;
                const tag = el.tagName.toLowerCase();
                const id = el.id ? ` id="${{el.id}}"` : '';
                const name = el.name ? ` name="${{el.name}}"` : '';
                const type = el.type ? ` type="${{el.type}}"` : '';
                const role = el.getAttribute('role') ? ` role="${{el.getAttribute('role')}}"` : '';
                const aria = el.getAttribute('aria-label') ? ` aria-label="${{el.getAttribute('aria-label')}}"` : '';
                const placeholder = el.placeholder ? ` placeholder="${{el.placeholder}}"` : '';
                const href = (tag === 'a' && el.href) ? ` href="${{el.href.substring(0, 60)}}"` : '';
                let text = (el.innerText || el.value || '').trim().substring(0, {max_text});
                if (text) text = `>${{text}}`;
                const visible = (rect.top >= 0 && rect.top < window.innerHeight) ? '' : ' [offscreen]';
                idx++;
                results.push(`[${{idx}}] <${{tag}}${{id}}${{name}}${{type}}${{role}}${{aria}}${{placeholder}}${{href}}${{visible}}${{text}}`);
            }}
            return results.join('\\n');
        }}
    """
    try:
        return await page.evaluate(script)
    except Exception:
        return "[DOM extraction failed]"


async def get_page_info(page) -> str:
    try:
        title = await page.title()
        url = page.url
        return f"Page: {title} | URL: {url}"
    except Exception:
        return "Page info unavailable"


# ─── Stuck Detection ────────────────────────────────────────────────────────

def detect_stuck(memory: AgentMemory) -> bool:
    recent = memory.get_last_n_actions(STUCK_THRESHOLD)
    if len(recent) < STUCK_THRESHOLD:
        return False
    return len(set(recent)) == 1


# ─── Agent Loop ─────────────────────────────────────────────────────────────

class AgentLoop:
    """Main orchestrator for the agent execution cycle."""

    def __init__(self, ws: WebSocket, page, objective: str, speed: str = "normal"):
        self.ws = ws
        self.page = page
        self.objective = objective
        self.speed = speed
        self.memory = AgentMemory()
        self.planner = AgentPlanner()
        self.executor = AgentExecutor(page)
        self.interrupted = False
        self.interrupt_prompt: Optional[str] = None
        self.consecutive_failures = 0

    async def run(self):
        step = 0

        # ── Initial Plan ──
        dom_text = await extract_dom(self.page)
        await self.planner.generate_initial_plan(self.objective, dom_text, self.ws)

        while step < MAX_STEPS and not self.interrupted:
            step += 1

            await self.ws.send_json({"type": "step_divider", "step": step, "msg": f"Step {step}/{MAX_STEPS}"})

            # ── 1. OBSERVE ──
            await self.ws.send_json({"type": "status", "step": step, "msg": "👁️ Observing page..."})
            dom_text = await extract_dom(self.page)
            page_info = await get_page_info(self.page)

            # ── CAPTCHA check ──
            if await detect_captcha(self.page):
                await self.ws.send_json({
                    "type": "error", "step": step,
                    "msg": "🛡️ CAPTCHA detected! Please solve it manually in the browser window. Waiting 30 seconds..."
                })
                await asyncio.sleep(30)
                # Re-check after wait
                if await detect_captcha(self.page):
                    await self.ws.send_json({"type": "error", "step": step, "msg": "🛡️ CAPTCHA still present. Skipping this step..."})
                    continue
                else:
                    await self.ws.send_json({"type": "status", "step": step, "msg": "✅ CAPTCHA solved! Resuming..."})
                    dom_text = await extract_dom(self.page)
                    page_info = await get_page_info(self.page)

            # ── Stuck detection ──
            if detect_stuck(self.memory):
                await self.ws.send_json({"type": "error", "step": step, "msg": "🔁 Stuck detected! Triggering replan..."})
                await self.planner.partial_replan(self.objective, self.memory.to_prompt(), dom_text, self.ws, step)
                self.consecutive_failures = 0

            # ── 2. THINK ──
            await self.ws.send_json({"type": "status", "step": step, "msg": "🧠 Thinking..."})
            decision = await self._think(dom_text, page_info)

            if decision is None:
                await self.ws.send_json({"type": "error", "step": step, "msg": "❌ Model returned invalid response. Retrying step..."})
                self.consecutive_failures += 1
                if self.consecutive_failures >= 3:
                    await self.ws.send_json({"type": "error", "step": step, "msg": "❌ Too many model failures. Stopping."})
                    break
                continue

            # ── Log thinking ──
            await self.ws.send_json({"type": "thinking", "step": step, "msg": decision.get("thought", "")})

            # ── Update plan ──
            response_plan = decision.get("plan", [])
            if response_plan:
                self.planner.update_from_response(response_plan)
                await self.ws.send_json({"type": "plan", "step": step, "msg": response_plan})

            # ── 3. EXECUTE ──
            action = decision.get("action", "")
            target = decision.get("target", "")
            value = decision.get("value", "")

            action_desc = f"{action}"
            if target:
                action_desc += f' → "{target}"'
            if value and action in ("type", "navigate"):
                action_desc += f' = "{value[:50]}"'

            await self.ws.send_json({"type": "action", "step": step, "msg": f"⚡ {action_desc}"})

            result = await self.executor.execute_action(decision, self.ws, step)

            # ── 4. Handle result ──
            if result == "DONE":
                await self.ws.send_json({"type": "success", "step": step, "msg": "✅ Objective achieved!"})
                self.memory.add(step, decision.get("thought", ""), action, target, value, "SUCCESS")
                return

            if result.startswith("CLARIFY:"):
                await self.ws.send_json({"type": "status", "step": step, "msg": f"❓ {result}"})
                self.memory.add(step, decision.get("thought", ""), action, target, value, result)
                return

            if result.startswith("FAILED"):
                await self.ws.send_json({"type": "error", "step": step, "msg": f"❌ {result}"})
                self.consecutive_failures += 1
                self.memory.add(step, decision.get("thought", ""), action, target, value, result)

                if self.consecutive_failures >= 2:
                    await self.ws.send_json({"type": "error", "step": step, "msg": "🔄 Multiple failures — replanning..."})
                    dom_text = await extract_dom(self.page)
                    await self.planner.partial_replan(self.objective, self.memory.to_prompt(), dom_text, self.ws, step)
                    self.consecutive_failures = 0
                continue

            # ── Success ──
            await self.ws.send_json({"type": "observation", "step": step, "msg": f"👁️ {result}"})
            self.memory.add(step, decision.get("thought", ""), action, target, value, result)
            self.consecutive_failures = 0

            # ── Speed delay ──
            if self.speed == "normal":
                await asyncio.sleep(0.3)
            elif self.speed == "fast":
                await asyncio.sleep(0.05)

        # ── Loop ended ──
        if self.interrupted:
            await self.ws.send_json({"type": "status", "step": step, "msg": "🛑 Execution interrupted by user."})
        else:
            await self.ws.send_json({"type": "status", "step": step, "msg": f"🛑 Reached max steps ({MAX_STEPS}). Stopping."})

    async def _think(self, dom_text: str, page_info: str) -> dict | None:
        user_prompt = "\n".join([
            f"OBJECTIVE: {self.objective}",
            f"\nPAGE INFO: {page_info}",
            f"\nCURRENT PLAN:\n{self.planner.plan_text()}",
            f"\nMEMORY (recent actions):\n{self.memory.to_prompt()}",
            f"\nINTERACTIVE DOM ELEMENTS:\n{dom_text}",
            "\nWhat is the NEXT best action?",
        ])
        return await call_model(SYSTEM_PROMPT, user_prompt)

    def interrupt(self, new_prompt: Optional[str] = None):
        self.interrupted = True
        self.interrupt_prompt = new_prompt


# ─── WebSocket Endpoint ────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    agent_loop: Optional[AgentLoop] = None
    browser = None

    try:
        raw = await websocket.receive_text()
        payload = json.loads(raw)

        objective = payload.get("objective", "")
        start_url = payload.get("url", "")
        speed = payload.get("speed", "normal")

        if not objective or not start_url:
            await websocket.send_json({"type": "error", "step": 0, "msg": "❌ Missing objective or URL."})
            return

        await websocket.send_json({"type": "status", "step": 0, "msg": "🚀 Initializing Agent..."})
        await websocket.send_json({"type": "status", "step": 0, "msg": f"🎯 Objective: {objective}"})
        await websocket.send_json({"type": "status", "step": 0, "msg": f"🤖 Model: {MODEL_NAME}"})

        pw = await async_playwright().start()
        browser = await pw.chromium.launch(headless=False)
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await context.new_page()

        await websocket.send_json({"type": "status", "step": 0, "msg": f"🌐 Navigating to {start_url}..."})
        await page.goto(start_url, wait_until="domcontentloaded", timeout=15000)
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=3000)
        except Exception:
            pass

        agent_loop = AgentLoop(websocket, page, objective, speed)

        loop_task = asyncio.create_task(agent_loop.run())
        listener_task = asyncio.create_task(_listen_for_interrupts(websocket, agent_loop))

        done_tasks, pending = await asyncio.wait(
            [loop_task, listener_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        if agent_loop.interrupted and agent_loop.interrupt_prompt:
            new_objective = agent_loop.interrupt_prompt
            await websocket.send_json({
                "type": "status", "step": 0,
                "msg": f"🔄 Restarting with new objective: {new_objective}"
            })

            agent_loop = AgentLoop(websocket, page, new_objective, speed)
            loop_task = asyncio.create_task(agent_loop.run())
            listener_task = asyncio.create_task(_listen_for_interrupts(websocket, agent_loop))

            done_tasks, pending = await asyncio.wait(
                [loop_task, listener_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        await websocket.send_json({"type": "status", "step": 0, "msg": "🛑 Agent session finished."})

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Server error: {e}")
        try:
            await websocket.send_json({"type": "error", "step": 0, "msg": f"❌ Server error: {str(e)[:100]}"})
        except Exception:
            pass
    finally:
        if browser:
            try:
                await browser.close()
            except Exception:
                pass


async def _listen_for_interrupts(ws: WebSocket, agent_loop: AgentLoop):
    try:
        while True:
            raw = await ws.receive_text()
            payload = json.loads(raw)
            if payload.get("type") == "interrupt":
                new_prompt = payload.get("prompt", None)
                agent_loop.interrupt(new_prompt)
                await ws.send_json({"type": "interrupt", "step": 0, "msg": "🛑 Interrupt received. Stopping..."})
                return
            elif payload.get("type") == "speed":
                agent_loop.speed = payload.get("speed", "normal")
    except (WebSocketDisconnect, Exception):
        return


# ─── Entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)