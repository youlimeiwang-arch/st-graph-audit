"""Phone pushes (Bark) and tap-to-choose replies (via an ntfy.sh topic).

The Bark device key and the reply topic are read from ~/.staig_notify.json (STAIG_NOTIFY_KEY overrides the key)
and must never be stored inside the repository.

ask() sends one Bark push per option; tapping one opens https://ntfy.sh/<topic>/publish?message=<request>:<option>,
and ask() polls the topic until a reply for this request arrives. With multi=True every tap toggles an option and
a separate "完成" push submits the current set. Only the listed option ids are accepted and nothing else is ever
executed, so a stranger who guessed the topic could at most pick among the offered options.

Usage:
    python scripts/maintenance/notify.py "标题" "正文"
    python scripts/maintenance/notify.py --ask "标题" "问题" A="选项一" B="选项二" [--multi] [--timeout 3600] [--default A]
        (prints the chosen option id, or comma-separated ids with --multi; the default / nothing on timeout)
    from notify import send, ask
"""
from __future__ import annotations

import json
import os
import secrets
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

CONFIG = Path.home() / ".staig_notify.json"
DONE = "__done__"
RESET = "__reset__"


def _config() -> dict:
    cfg = {"server": "https://api.day.app", "reply_server": "https://ntfy.sh"}
    if CONFIG.exists():
        cfg.update(json.loads(CONFIG.read_text(encoding="utf-8")))
    if os.environ.get("STAIG_NOTIFY_KEY"):
        cfg["key"] = os.environ["STAIG_NOTIFY_KEY"]
    return cfg


def send(title: str, body: str, group: str = "STAIG", url: str | None = None) -> bool:
    cfg = _config()
    if not cfg.get("key"):
        print("[notify] no key configured; skipped", file=sys.stderr)
        return False
    payload = {"title": title, "body": body[:1500], "group": group}
    if url:
        payload["url"] = url
    req = urllib.request.Request(f"{cfg['server'].rstrip('/')}/{cfg['key']}", data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json; charset=utf-8"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            ok = r.status == 200
    except Exception as e:  # a failed push must never break a running job
        print(f"[notify] failed: {e}", file=sys.stderr)
        return False
    print(f"[notify] sent: {title}" if ok else "[notify] non-200 response", file=sys.stderr)
    return ok


def _replies(cfg: dict) -> list[tuple[str, str]]:
    """(message id, text) in publication order."""
    # since=all instead of a timestamp: the local clock is not trusted; requests are told apart by their random id
    url = f"{cfg['reply_server'].rstrip('/')}/{cfg['reply_topic']}/json?poll=1&since=all"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            lines = r.read().decode("utf-8").splitlines()
    except Exception as e:
        print(f"[notify] poll failed: {e}", file=sys.stderr)
        return []
    out = []
    for line in lines:
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if m.get("event") == "message":
            out.append((str(m.get("id", "")), str(m.get("message", "")).strip()))
    return out


def ask(title: str, question: str, options: dict[str, str], timeout_s: int = 3600,
        default=None, poll_s: int = 15, multi: bool = False):
    """Return the chosen option id (multi=True: list of ids, possibly empty), or default on timeout."""
    cfg = _config()
    if not cfg.get("reply_topic"):
        print("[notify] no reply_topic configured; returning default", file=sys.stderr)
        return default
    rid = secrets.token_hex(4)
    base = f"{cfg['reply_server'].rstrip('/')}/{cfg['reply_topic']}/publish"
    link = lambda k: f"{base}?message={urllib.parse.quote(f'{rid}:{k}')}"
    lines = "\n".join(f"{k}. {v}" for k, v in options.items())
    wait = f"{timeout_s // 60} 分钟" if timeout_s < 7200 else f"{timeout_s / 3600:.1f} 小时"
    dflt = ",".join(default) if isinstance(default, (list, tuple)) else default
    fallback = f"，超时默认选 {dflt}" if dflt else ""
    how = ("可以多选：点一条就选上（重复点不会取消）；点错了点「清空重选」；选好后点「完成」提交" if multi
           else "点下面对应选项的那条通知即可")
    send(f"需要你选择：{title}", f"{question}\n{lines}\n\n{how}（{wait}内有效{fallback}）。")
    for k, v in options.items():
        send(f"选 {k}：{v}"[:60], f"点这条 = 选择 {k}（{title}）", url=link(k))
    if multi:
        send(f"清空重选（{title}）"[:60], "点错了就点这条，已选的全部清空（只能用一次）", url=link(RESET))
        send(f"完成：提交选择（{title}）"[:60], "选好后点这条提交", url=link(DONE))

    # Taps are add-only and every distinct link counts once: iOS may re-open (reload) a link page later,
    # which publishes the same message again.
    chosen: list[str] = []
    seen: set[str] = set()
    used: set[str] = set()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for mid, msg in _replies(cfg):
            if mid in seen:
                continue
            seen.add(mid)
            r, _, opt = msg.partition(":")
            if r != rid:
                continue
            if not multi and opt in options:
                send(f"已收到你的选择：{opt}", f"{title}：{options[opt]}")
                return opt
            if not multi or msg in used:
                continue
            used.add(msg)
            if opt in options or opt == RESET:
                if opt == RESET:
                    chosen.clear()
                    used = {m for m in used if m.partition(":")[2] not in options}
                elif opt not in chosen:
                    chosen.append(opt)
                now = "、".join(f"{k}（{options[k]}）" for k in chosen) or "（还没选）"
                send(f"当前已选：{'、'.join(chosen) or '无'}", f"{title}\n{now}\n选好后点「完成」。")
            elif opt == DONE:
                picked = "、".join(f"{k}（{options[k]}）" for k in chosen) or "一个都不选"
                send(f"已提交：{'、'.join(chosen) or '无'}", f"{title}：{picked}")
                return chosen
        time.sleep(poll_s)
    send(f"选择超时：{title}", f"没有收到{'「完成」' if multi else '选择'}，"
                              f"{'按默认选 ' + dflt if dflt else '暂不处理，等你回到电脑前'}。")
    return default


def _cli_ask(argv: list[str]) -> int:
    timeout, default, multi, pos, opts = 3600, None, False, [], {}
    it = iter(argv)
    for a in it:
        if a == "--timeout":
            timeout = int(next(it))
        elif a == "--default":
            default = next(it)
        elif a == "--multi":
            multi = True
        elif "=" in a and len(pos) >= 2:
            k, v = a.split("=", 1)
            opts[k] = v
        else:
            pos.append(a)
    if len(pos) < 2 or not opts:
        print('usage: notify.py --ask "标题" "问题" A="选项一" B="选项二" [--multi] [--timeout 秒] [--default A]',
              file=sys.stderr)
        return 2
    if multi and default is not None:
        default = [d for d in default.split(",") if d]
    choice = ask(pos[0], pos[1], opts, timeout, default, multi=multi)
    if choice is None:
        return 1
    print(",".join(choice) if multi else choice)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--ask":
        sys.exit(_cli_ask(sys.argv[2:]))
    title = sys.argv[1] if len(sys.argv) > 1 else "STAIG"
    body = sys.argv[2] if len(sys.argv) > 2 else "通知测试"
    sys.exit(0 if send(title, body) else 1)
