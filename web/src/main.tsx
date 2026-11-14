import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import "./styles.css";
import { RequireUser, SessionProvider } from "./session";
import Landing from "./pages/Landing";
import { Login, Signup } from "./pages/Auth";
import Shell from "./pages/Shell";
import Home from "./pages/Home";
import Ask from "./pages/Ask";
import Library from "./pages/Library";
import SettingsPage from "./pages/Settings";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <SessionProvider>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/app" element={<RequireUser><Shell /></RequireUser>}>
            <Route index element={<Home />} />
            <Route path="w/:id" element={<Ask />} />
            <Route path="w/:id/library" element={<Library />} />
            <Route path="settings" element={<SettingsPage />} />
          </Route>
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </SessionProvider>
    </BrowserRouter>
  </StrictMode>,
);
