// Stroke icons (decorative: aria-hidden; buttons carry their own labels).
type P = { size?: number };
const base = (size = 16) => ({
  width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor",
  strokeWidth: 2, strokeLinecap: "round" as const, strokeLinejoin: "round" as const, "aria-hidden": true,
});

export const Chat = ({ size }: P) => <svg {...base(size)}><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" /></svg>;
export const Book = ({ size }: P) => <svg {...base(size)}><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20V3H6.5A2.5 2.5 0 0 0 4 5.5z" /><path d="M4 19.5A2.5 2.5 0 0 0 6.5 22H20v-5" /></svg>;
export const Gear = ({ size }: P) => <svg {...base(size)}><circle cx="12" cy="12" r="3" /><path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M5 19l2-2M17 7l2-2" /></svg>;
export const Home = ({ size }: P) => <svg {...base(size)}><path d="M3 11 12 4l9 7" /><path d="M5 10v10h14V10" /></svg>;
export const Chevron = ({ size }: P) => <svg {...base(size)}><path d="m7 10 5 5 5-5" /></svg>;
export const Up = ({ size }: P) => <svg {...base(size)} strokeWidth={2.4}><path d="M12 19V5M6 11l6-6 6 6" /></svg>;
export const Check = ({ size }: P) => <svg {...base(size)} strokeWidth={2.6}><path d="M20 6 9 17l-5-5" /></svg>;
export const Doc = ({ size }: P) => <svg {...base(size)}><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" /><path d="M14 3v5h5" /></svg>;
export const UploadIcon = ({ size }: P) => <svg {...base(size)}><path d="M12 16V4M6 10l6-6 6 6M4 20h16" /></svg>;
export const Close = ({ size }: P) => <svg {...base(size)}><path d="M6 6l12 12M18 6 6 18" /></svg>;
export const LogOut = ({ size }: P) => <svg {...base(size)}><path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4" /><path d="M10 17l-5-5 5-5M5 12h12" /></svg>;
export const Plus = ({ size }: P) => <svg {...base(size)} strokeWidth={2.2}><path d="M12 5v14M5 12h14" /></svg>;
export const Search = ({ size }: P) => <svg {...base(size)}><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></svg>;
