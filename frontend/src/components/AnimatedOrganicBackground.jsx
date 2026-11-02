import React from 'react';

/**
 * AnimatedOrganicBackground
 * Premium Modern "Light-Dark" Slate-Emerald Ambient Environment:
 * - Balanced medium-dark tone (rich slate & deep forest canvas, neither blinding light nor pitch black)
 * - Fluid multi-layered organic aurora orbs drifting with continuous smooth movement
 * - Gentle bioluminescent ambient pulses and soft floating particles
 * - Clean, natural, human-designed aesthetic (No tacky sci-fi laser scans or fake HUD boxes)
 */
export default function AnimatedOrganicBackground() {
  return (
    <div className="fixed inset-0 pointer-events-none -z-10 overflow-hidden select-none bg-gradient-to-b from-[#0d181e] via-[#0f1d24] to-[#091116]">
      {/* 1. Natural Ambient Radial Atmosphere */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_90%_70%_at_50%_0%,rgba(16,185,129,0.22),transparent_70%)]" />
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_70%_60%_at_85%_75%,rgba(20,184,166,0.18),transparent_65%)]" />
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_60%_50%_at_15%_55%,rgba(14,165,233,0.14),transparent_60%)]" />

      {/* 2. Soft Studio Micro-Dot Texture (Clean & tactile, not a harsh cyber grid) */}
      <div
        className="absolute inset-0 opacity-[0.06]"
        style={{
          backgroundImage: 'radial-gradient(rgba(255, 255, 255, 0.4) 1px, transparent 1px)',
          backgroundSize: '32px 32px',
        }}
      />

      {/* 3. Living Fluid Aurora Orbs (Smooth, continuous organic movement) */}
      {/* Primary Emerald/Jade Drifting Orb */}
      <div
        className="absolute -top-[10%] left-[10%] w-[58vw] h-[58vw] rounded-full bg-gradient-to-tr from-emerald-500/25 via-teal-400/20 to-emerald-600/10 blur-[130px] animate-blob-drift-1"
      />

      {/* Secondary Cyan/Ocean Teal Drifting Orb */}
      <div
        className="absolute top-[35%] right-[5%] w-[52vw] h-[52vw] rounded-full bg-gradient-to-bl from-teal-400/20 via-emerald-500/15 to-cyan-500/15 blur-[140px] animate-blob-drift-2"
      />

      {/* Tertiary Soft Indigo/Sky Accent Orb */}
      <div
        className="absolute bottom-[5%] left-[20%] w-[48vw] h-[48vw] rounded-full bg-gradient-to-tr from-sky-500/15 via-teal-500/15 to-emerald-400/10 blur-[130px] animate-blob-drift-3"
      />

      {/* Subtle Center Breathing Ambient Halo */}
      <div
        className="absolute top-[20%] left-[35%] w-[35vw] h-[35vw] rounded-full bg-emerald-400/10 blur-[120px] animate-pulse"
        style={{ animationDuration: '9s' }}
      />

      {/* 4. Soft Organic Drifting Light Particles (Gentle stardust, calm and natural) */}
      <div className="absolute top-[20%] left-[22%] w-2 h-2 rounded-full bg-emerald-300/40 blur-[1px] animate-float-slow" />
      <div className="absolute top-[38%] left-[78%] w-2.5 h-2.5 rounded-full bg-teal-300/35 blur-[1px] animate-float-delayed" />
      <div className="absolute top-[62%] left-[32%] w-2 h-2 rounded-full bg-emerald-200/30 blur-[1px] animate-float-slow" />
      <div className="absolute top-[78%] left-[68%] w-3 h-3 rounded-full bg-teal-200/25 blur-[1.5px] animate-float-delayed" />
      <div className="absolute top-[15%] right-[28%] w-2 h-2 rounded-full bg-cyan-300/30 blur-[1px] animate-float-slow" />
      <div className="absolute bottom-[25%] left-[15%] w-2.5 h-2.5 rounded-full bg-emerald-400/30 blur-[1px] animate-float-delayed" />
    </div>
  );
}
