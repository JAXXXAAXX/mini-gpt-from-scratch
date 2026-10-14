"""Local panel: type a sentence or some words, read the continuation, type again.

Run:
    python interact.py

The page has one text box. Each submission is a fresh prompt for the active
checkpoint. The box is ready again after the continuation appears.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

import config
from generate import encode_prompt, generate_from_ids, load_model
from src.dialogue_data import format_chat_prompt, trim_assistant_reply
from src.tokenizer_utils import load_tokenizer

HOST = "127.0.0.1"
PORT = 8765
MAX_NEW_TOKENS = 80

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MiniGPT</title>
<style>
  :root {
    color-scheme: light;
    --ink: #1c1915;
    --muted: #6b645c;
    --line: #e4ddd4;
    --paper: #f7f4ef;
    --card: #fffdf9;
    --you: #efe7dc;
    --reply: #f3f7f2;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    min-height: 100vh;
    background: var(--paper);
    color: var(--ink);
    font: 17px/1.5 Georgia, "Iowan Old Style", Palatino, serif;
  }
  main {
    max-width: 720px;
    margin: 0 auto;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    padding: 28px 20px 24px;
  }
  h1 {
    margin: 0;
    font-size: 22px;
    font-weight: normal;
  }
  p.hint {
    margin: 6px 0 18px;
    color: var(--muted);
    font: 14px/1.4 system-ui, sans-serif;
  }
  #log {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .turn {
    border: 1px solid var(--line);
    background: var(--card);
    border-radius: 12px;
    padding: 12px 14px;
  }
  .label {
    display: block;
    margin-bottom: 4px;
    font: 12px/1.2 system-ui, sans-serif;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--muted);
  }
  .you { background: var(--you); }
  .reply { background: var(--reply); white-space: pre-wrap; }
  form {
    display: flex;
    gap: 8px;
    margin-top: 16px;
    position: sticky;
    bottom: 16px;
  }
  input[type="text"] {
    flex: 1;
    border: 1px solid var(--line);
    background: white;
    border-radius: 10px;
    padding: 12px 14px;
    font: 16px/1.4 system-ui, sans-serif;
    color: var(--ink);
  }
  button {
    border: 0;
    border-radius: 10px;
    padding: 0 16px;
    background: #2c3a2e;
    color: white;
    font: 15px/1 system-ui, sans-serif;
    cursor: pointer;
  }
  button:disabled, input:disabled { opacity: 0.6; cursor: wait; }
  .error { color: #8a2b2b; font: 14px/1.4 system-ui, sans-serif; }
</style>
</head>
<body>
<main>
  <h1>MiniGPT</h1>
  <p class="hint">Type a sentence. For reply-tuned checkpoints, the model answers as Assistant instead of continuing a children's story.</p>
  <div id="log"></div>
  <form id="composer">
    <input id="line" type="text" autocomplete="off" placeholder="Once upon a time" autofocus>
    <button id="send" type="submit">Continue</button>
  </form>
</main>
<script>
const log = document.getElementById("log");
const form = document.getElementById("composer");
const line = document.getElementById("line");
const send = document.getElementById("send");

function addTurn(kind, text) {
  const block = document.createElement("div");
  block.className = "turn " + kind;
  const label = document.createElement("span");
  label.className = "label";
  label.textContent = kind === "you" ? "You" : "Reply";
  const body = document.createElement("div");
  body.textContent = text;
  block.append(label, body);
  log.append(block);
  block.scrollIntoView({ block: "nearest" });
  return body;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = line.value.trim();
  if (!text) {
    line.focus();
    return;
  }
  addTurn("you", text);
  line.value = "";
  line.disabled = true;
  send.disabled = true;
  const pending = addTurn("reply", "…");
  try {
    const response = await fetch("/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const payload = await response.json();
    if (!response.ok) {
      pending.textContent = payload.error || "The model did not answer.";
      pending.classList.add("error");
    } else {
      pending.textContent = payload.continuation || "";
    }
  } catch (error) {
    pending.textContent = "The panel could not reach the model.";
    pending.classList.add("error");
  } finally {
    line.disabled = false;
    send.disabled = false;
    line.focus();
  }
});
</script>
</body>
</html>
"""


class Panel:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = load_tokenizer()
        self.model, self.checkpoint = load_model(config.BEST_CHECKPOINT, self.device)
        self.model.eval()

    def continue_text(self, text: str) -> str:
        prompt = text.strip()
        if not prompt:
            raise ValueError("Type a sentence or a few words first.")
        use_chat = bool(getattr(config, "USE_CHAT_TEMPLATE", False))
        model_prompt = format_chat_prompt(prompt) if use_chat else prompt
        temperature = 0.4 if use_chat else config.STORY_EVAL_TEMPERATURE
        prompt_ids = encode_prompt(self.tokenizer, model_prompt, self.device)[0].tolist()
        sample = generate_from_ids(
            self.model,
            self.tokenizer,
            prompt_ids=prompt_ids,
            prompt_text=model_prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=temperature,
            top_k=config.STORY_EVAL_TOP_K,
            seed=None,
            device=self.device,
        )
        return trim_assistant_reply(sample["continuation_text"])


def make_handler(panel: Panel):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path not in {"/", "/index.html"}:
                self.send_error(404)
                return
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if self.path != "/generate":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
                continuation = panel.continue_text(str(payload.get("text", "")))
                response = {"continuation": continuation}
                status = 200
            except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
                response = {"error": str(exc)}
                status = 400
            body = json.dumps(response).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            return

    return Handler


def main() -> None:
    panel = Panel()
    server = ThreadingHTTPServer((HOST, PORT), make_handler(panel))
    print(f"MiniGPT panel: http://{HOST}:{PORT}")
    print(f"Checkpoint: {config.BEST_CHECKPOINT}")
    print(f"Chat template: {getattr(config, 'USE_CHAT_TEMPLATE', False)}")
    print("Type a sentence in the page. Close this window to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
