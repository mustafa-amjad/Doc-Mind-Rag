import { Link } from "react-router-dom";
import { useSession } from "../session";

export default function Landing() {
  const { user } = useSession();
  const start = user ? "/app" : "/signup";
  return (
    <div className="landing">
      <header className="wrap site-head">
        <Link className="logo" to="/">DocMind</Link>
        <nav aria-label="Site">
          <a href="#how">How it works</a>
          <a href="#features">Features</a>
          <a href="#security">Security</a>
          <a href="#pricing">Pricing</a>
        </nav>
        <div className="cta">
          {user ? <Link className="btn" to="/app">Open DocMind</Link> : <Link className="btn" to="/login">Sign in</Link>}
          <Link className="btn btn-dark" to={start}>Start free</Link>
        </div>
      </header>

      <section className="wrap hero-l">
        <span className="pill pill-plain" style={{ height: 30, fontSize: 13 }}><span style={{ width: 8, height: 8, borderRadius: 99, background: "var(--accent)" }} />Every answer cites its page</span>
        <h1>Answers you can <em>check.</em></h1>
        <p>Ask questions about your policies, contracts and reports. DocMind answers only from your documents, shows the exact page it used, and tells you when the answer isn’t there.</p>
        <div className="ctas">
          <Link className="btn btn-dark btn-lg" to={start}>Start free</Link>
          <a className="btn btn-lg" href="#how">See how it works</a>
        </div>
      </section>

      <section className="wrap" aria-label="Product preview" style={{ paddingTop: 24, paddingBottom: 96 }}>
        <div className="preview-frame">
          <div className="preview">
            <div className="p-side">
              <span className="serif" style={{ fontSize: 22 }}>DocMind</span>
              <span className="on">Ask</span>
              <span className="off">Library</span>
              <span className="off">Settings</span>
            </div>
            <div className="p-main">
              <span className="serif" style={{ fontSize: 26, lineHeight: 1.2 }}>How long do I have to return a jacket?</span>
              <span className="pill pill-accent" style={{ alignSelf: "flex-start" }}>✓ Cited from 1 source</span>
              <p style={{ margin: 0, fontSize: 15, lineHeight: 1.7 }}>You can return most items within <strong>30 days of delivery</strong>, unworn and unwashed with the original tags attached. <span className="cite" aria-hidden="true">1</span></p>
            </div>
            <div className="p-page">
              <div className="p-paper">
                <span style={{ font: "600 11px var(--sans)", color: "var(--muted)" }}>handbook.pdf · page 3</span>
                <span style={{ font: "600 15px Georgia, serif" }}>2. Returns and refunds</span>
                <span><mark>You can return most items within 30 days of delivery.</mark> Items must be unworn and unwashed, with the original tags attached.</span>
                <span>Refunds go back to the original payment method within 5 to 7 business days.</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="how" className="wrap" style={{ paddingBottom: 96, display: "flex", flexDirection: "column", gap: 40 }}>
        <h2 className="h2">Three steps. No prompt engineering.</h2>
        <div className="steps3">
          <div className="card"><span className="num">01</span><h3>Add your documents</h3><p>PDF, Word, text, web pages and images. Scans are read with OCR, and tables keep their rows and columns.</p></div>
          <div className="card"><span className="num">02</span><h3>Ask in plain English</h3><p>Keyword and meaning search find the right passage, and a re-ranker puts the best one first. Follow-up questions keep their context.</p></div>
          <div className="card"><span className="num">03</span><h3>Check the source</h3><p>Every claim links to its file and page. If the documents don’t say it, DocMind says “I couldn’t find that”.</p></div>
        </div>
      </section>

      <section id="features" className="dark-band">
        <div className="wrap band">
          <h2 className="h2">Built for answers people will act on.</h2>
          <div className="feature-grid">
            <div><h3>Page-level citations</h3><p>Open the page beside the answer, with the sentence highlighted.</p></div>
            <div><h3>Honest when it doesn’t know</h3><p>No guessing. Questions built on a wrong fact get corrected from the source.</p></div>
            <div><h3>Reads tables, scans and charts</h3><p>Price tables, scanned notices and bar charts become searchable text.</p></div>
            <div><h3>Hybrid search + re-ranking</h3><p>Exact terms like IDs and names, plus meaning, merged and re-ranked.</p></div>
            <div><h3>Workspaces that remember</h3><p>Documents and answers are saved per workspace, so you can come back to them.</p></div>
            <div><h3>Export with sources</h3><p>Download a workspace’s questions and answers with every citation attached.</p></div>
          </div>
        </div>
      </section>

      <section id="security" className="wrap band">
        <div className="security">
          <div>
            <h2 className="h2">Your documents stay yours.</h2>
            <p>[YOUR DATA POLICY: where documents are stored, which AI provider processes them, and how long they are kept.]</p>
          </div>
          <ul className="checks">
            <li><b>✓</b><span>Answers use only the documents in that workspace</span></li>
            <li><b>✓</b><span>Separate accounts; only a hash of each sign-in token is stored</span></li>
            <li><b>✓</b><span>Delete a workspace and its files and index are deleted with it</span></li>
            <li><b>✓</b><span>[YOUR COMPLIANCE ITEM, e.g. hosting region]</span></li>
          </ul>
        </div>
      </section>

      <section id="pricing" className="wrap" style={{ paddingBottom: 96, display: "flex", flexDirection: "column", gap: 40 }}>
        <h2 className="h2" style={{ textAlign: "center", alignSelf: "center" }}>Simple pricing</h2>
        <div className="prices">
          <div className="card price"><h3>Free</h3><span className="amount">$0</span><p>One workspace to try it with your own files. [YOUR LIMITS]</p><Link className="btn" to={start}>Start free</Link></div>
          <div className="card price featured">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}><h3>Pro</h3><span className="pill" style={{ background: "var(--accent)", color: "#fff" }}>Most teams</span></div>
            <span className="amount">[YOUR PRICE]</span><p>Unlimited workspaces, OCR, chart reading and exports. [YOUR LIMITS]</p>
            <Link className="btn" to={start} style={{ background: "var(--side-text)", borderColor: "var(--side-text)" }}>Upgrade to Pro</Link>
          </div>
          <div className="card price"><h3>Team</h3><span className="amount">[YOUR PRICE]</span><p>Shared workspaces, roles and private hosting. [YOUR LIMITS]</p><a className="btn" href="mailto:[YOUR EMAIL]">Talk to us</a></div>
        </div>
      </section>

      <section className="wrap" style={{ paddingBottom: 96 }}>
        <div className="cta-band">
          <h2>Stop searching PDFs. Start checking answers.</h2>
          <Link className="btn btn-lg" to={start}>Start free</Link>
        </div>
      </section>

      <footer className="site-foot">
        <div className="wrap">
          <span className="serif" style={{ fontSize: 22, color: "var(--ink)" }}>DocMind</span>
          <span>Built by Mustafa Amjad</span>
          <div className="links"><a href="#">Privacy</a><a href="#">Terms</a><a href="mailto:[YOUR EMAIL]">Contact</a></div>
        </div>
      </footer>
    </div>
  );
}
