// import React from 'react';

// const LOGO_SRC = '/assets/branding/sentinel_e.png';

// export default function SentinelIdentity({
//   size = 40,
//   label = 'Sentinel-E',
//   showLabel = false,
//   pulse = false,
//   className = '',
// }) {
//   return (
//     <div className={`sentinel-identity ${pulse ? 'sentinel-identity--pulse' : ''} ${className}`}>
//       <div
//         className="sentinel-mark"
//         style={{ width: size, height: size, borderRadius: Math.max(12, size * 0.28) }}
//         aria-hidden="true"
//       >
//         <img
//           src={LOGO_SRC}
//           alt="Sentinel-E"
//           className="sentinel-logo"
//           style={{ width: size * 0.64, height: size * 0.64 }}
//         />
//       </div>
//       {showLabel && (
//         <div className="sentinel-label">
//           <span>{label}</span>
//           <small>cognitive runtime</small>
//         </div>
//       )}
//     </div>
//   );
// }


import React from 'react';

export default function SentinelIdentity({
  size = 36,
  className = '',
  rounded = '20px',
}) {
  return (
    <div
      className={className}
      style={{
        width: size,
        height: size,

        borderRadius: rounded,

        overflow: 'hidden',

        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',

        background:
          'rgba(255,255,255,0.72)',

        border:
          '1px solid rgba(255,255,255,0.12)',

        backdropFilter:
          'blur(20px)',

        boxShadow:
          '0 8px 24px rgba(0,0,0,0.08)',
      }}
    >
      <img
        src="/assets/branding/sentinel_e.png"
        alt="Sentinel-E"
        draggable={false}
        style={{
          width: '100%',
          height: '100%',
          objectFit: 'cover',
          userSelect: 'none',
        }}
      />
    </div>
  );
}