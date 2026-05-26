import React from "react";

export const TerminalHUD: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <div
      style={{
        width: "100vw",
        height: "100vh",
        backgroundColor: "#050505",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontFamily: "'Courier New', Courier, monospace",
        color: "#00ffcc",
        overflow: "hidden",
        position: "relative",
      }}
    >
      {/* CRT Scanline Overlay */}
      <div
        style={{
          position: "absolute",
          top: 0, left: 0, right: 0, bottom: 0,
          background: "linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06))",
          backgroundSize: "100% 2px, 3px 100%",
          pointerEvents: "none",
          zIndex: 50,
        }}
      />
      {/* Vignette */}
      <div
        style={{
          position: "absolute",
          top: 0, left: 0, right: 0, bottom: 0,
          background: "radial-gradient(circle, rgba(0,0,0,0) 60%, rgba(0,0,0,0.6) 100%)",
          pointerEvents: "none",
          zIndex: 40,
        }}
      />
      
      <div style={{ position: "relative", zIndex: 10, display: "flex", gap: "2rem", alignItems: "flex-start" }}>
        {children}
      </div>
    </div>
  );
};
