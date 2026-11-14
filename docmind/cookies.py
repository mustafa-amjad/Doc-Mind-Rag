"""'Stay signed in' for the Streamlit app.

The sign-in token lives in the browser's localStorage and reaches Python through a tiny
Streamlit component (custom components v2). Cookies are not used: Streamlit Community
Cloud strips custom cookies before they reach the app, which signed people out on every
refresh there. The token is a random key that expires on the server; only its hash is stored.
"""
import streamlit as st

STORE_KEY = "_signin_store"  # component state: {"token": "..."} once the browser has answered
PENDING_KEY = "_signin_action"  # ("set", token) or ("clear", None), sent with the next run

_STORE = st.components.v2.component(
    "docmind_signin_store",
    html="<span hidden></span>",
    js="""
export default function (component) {
  const { data, setStateValue } = component
  const KEY = "docmind_session"
  let token = ""
  try {
    if (data && data.action === "set" && data.token) localStorage.setItem(KEY, data.token)
    if (data && data.action === "clear") localStorage.removeItem(KEY)
    token = localStorage.getItem(KEY) || ""
  } catch (error) {
    token = ""  // storage blocked (private mode, policy): behave as signed out
  }
  if (!data || data.known !== token) setStateValue("token", token)
}
""",
)


def queue_set(token: str) -> None:
    """Save this token in the browser on the next run (after st.rerun())."""
    st.session_state[PENDING_KEY] = ("set", token)


def queue_clear() -> None:
    """Remove the saved token from the browser on the next run."""
    st.session_state[PENDING_KEY] = ("clear", None)


def sync() -> str | None:
    """Mount the store (call once, near the top of every run).
    Returns the saved token, "" when there is none, or None while the browser hasn't answered yet."""
    kind, token = st.session_state.pop(PENDING_KEY, None) or (None, None)
    known = (st.session_state.get(STORE_KEY) or {}).get("token")
    result = _STORE(key=STORE_KEY, data={"action": kind, "token": token, "known": known},
                    on_token_change=lambda: None, height=1)
    return getattr(result, "token", None)


def read_token() -> str | None:
    """The token the browser last reported (None if unknown)."""
    return (st.session_state.get(STORE_KEY) or {}).get("token") or None
