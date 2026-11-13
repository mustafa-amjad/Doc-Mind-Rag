"""Browser cookie helpers for 'stay logged in'.
Reading is easy (the server receives cookies). Writing needs a tiny JavaScript snippet."""
import json  # safely puts values inside JavaScript

import streamlit as st

from docmind import config

PENDING_KEY = "_cookie_action"  # remembers "set" or "clear" until the next page draw


def read_token() -> str | None:
    """Cookie sent by the browser when this session started (None if there is none)."""
    try:
        token = st.context.cookies.get(config.COOKIE_NAME)
    except Exception:  # very old Streamlit or no context
        return None
    return token if isinstance(token, str) and token else None


def queue_set(token: str) -> None:
    st.session_state[PENDING_KEY] = ("set", token)  # written on the next run, after st.rerun()


def queue_clear() -> None:
    st.session_state[PENDING_KEY] = ("clear", None)


def apply_pending() -> None:
    """Call near the top of every run. Writes/clears the cookie once, then forgets the request."""
    action = st.session_state.pop(PENDING_KEY, None)
    if not action:
        return
    kind, token = action
    max_age = config.SESSION_DAYS * 86400 if kind == "set" else 0  # 0 = delete now
    value = json.dumps(token or "")  # JSON string = safe JS string
    name = json.dumps(config.COOKIE_NAME)
    st.html(  # runs in the page itself, so window.parent === window
        f"""<script>{{                                            // block scope: safe to run more than once
        const p = window.parent;                                   // the real page (also correct inside an iframe)
        const secure = p.location.protocol === "https:" ? "; Secure" : "";   // Secure flag only on https
        p.document.cookie = {name} + "=" + {value} + "; path=/; max-age={max_age}; SameSite=Lax" + secure;
        }}</script>""",
        unsafe_allow_javascript=True,
    )
