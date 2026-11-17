// The DocMind wordmark: "Doc" speaks from an emerald chat bubble and its "o" is a magnifying glass that
// finds the line; the "i" in Mind carries an orange idea spark. Built from the page font, so it scales with
// font-size (set it on the parent or pass `size`). Same design as the Streamlit app (assets/docmind.css).
type Props = { size?: number; className?: string };

export function Wordmark({ size, className }: Props) {
  return (
    <span className={"dm-word" + (className ? ` ${className}` : "")} style={size ? { fontSize: size } : undefined}
      role="img" aria-label="DocMind">
      <span className="dm-bubble" aria-hidden="true">
        D
        <svg className="dm-lens" viewBox="0 0 26 26">
          <circle cx="10" cy="10" r="8.2" fill="none" stroke="#FAFAF7" strokeWidth="3.8" />
          <rect x="5.2" y="8.4" width="9.6" height="3.2" rx="1.6" fill="#E8692B" />
          <path d="M16.4 16.4 22.5 22.5" stroke="#E8692B" strokeWidth="4.2" strokeLinecap="round" />
        </svg>
        c
      </span>
      <span className="dm-mind" aria-hidden="true">
        M
        <span className="dm-i">
          ı
          <svg className="dm-spark" viewBox="0 0 10 10">
            <path d="M5 0C5.6 3.4 6.6 4.4 10 5 6.6 5.6 5.6 6.6 5 10 4.4 6.6 3.4 5.6 0 5 3.4 4.4 4.4 3.4 5 0z" fill="#E8692B" />
          </svg>
        </span>
        nd
      </span>
    </span>
  );
}
