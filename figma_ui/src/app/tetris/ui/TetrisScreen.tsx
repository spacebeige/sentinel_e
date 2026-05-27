// // // import React from "react"

// // // interface TetrisScreenProps {
// // //   children?: React.ReactNode
// // // }

// // // export default function TetrisScreen({
// // //   children,
// // // }: TetrisScreenProps) {
// // //   return (
// // //     <div
// // //       className="
// // //         relative
// // //         w-full
// // //         h-full
// // //         overflow-hidden
// // //         rounded-[42px]
// // //       "
// // //       style={{
// // //         background: `
// // //           radial-gradient(
// // //             circle at center,
// // //             #0a2b14 0%,
// // //             #05150c 35%,
// // //             #020705 70%,
// // //             #000000 100%
// // //           )
// // //         `,
// // //         border: "2px solid rgba(0,255,120,0.12)",
// // //         boxShadow: `
// // //           inset 0 0 180px rgba(0,0,0,1),
// // //           inset 0 0 120px rgba(0,255,120,0.08),
// // //           inset 0 0 40px rgba(0,255,120,0.06),
// // //           0 0 40px rgba(0,180,80,0.02)
// // //         `,
// // //       }}
// // //     >
// // //       {/* OUTER VIGNETTE */}
// // //       <div
// // //         className="absolute inset-0 z-[1] pointer-events-none"
// // //         style={{
// // //           background: `
// // //             radial-gradient(
// // //               circle at center,
// // //               transparent 40%,
// // //               rgba(0,0,0,0.15) 58%,
// // //               rgba(0,0,0,0.42) 78%,
// // //               rgba(0,0,0,0.92) 100%
// // //             )
// // //           `,
// // //         }}
// // //       />

// // //       {/* CRT CURVE SHADING */}
// // //       <div
// // //         className="absolute inset-0 z-[2] pointer-events-none"
// // //         style={{
// // //           background: `
// // //             radial-gradient(
// // //               ellipse at center,
// // //               rgba(0,180,80,0.02) 0%,
// // //               rgba(0,255,120,0.02) 40%,
// // //               rgba(0,0,0,0.18) 75%,
// // //               rgba(0,0,0,0.55) 100%
// // //             )
// // //           `,
// // //           transform: "scale(1.04)",
// // //           filter: "blur(8px)",
// // //         }}
// // //       />

// // //       {/* PHOSPHOR BLOOM */}
// // //       <div
// // //         className="absolute inset-0 z-[3] opacity-70 pointer-events-none"
// // //         style={{
// // //           background: `
// // //             radial-gradient(
// // //               circle at center,
// // //               rgba(0,255,120,0.10) 0%,
// // //               rgba(0,180,80,0.02) 30%,
// // //               rgba(0,0,0,0) 75%
// // //             )
// // //           `,
// // //           mixBlendMode: "screen",
// // //         }}
// // //       />

// // //       {/* CRT GLASS REFLECTION */}
// // //       <div
// // //         className="absolute top-0 left-0 right-0 h-[42%] z-[4] pointer-events-none"
// // //         style={{
// // //           background: `
// // //             linear-gradient(
// // //               to bottom,
// // //               rgba(255,255,255,0.08) 0%,
// // //               rgba(255,255,255,0.03) 18%,
// // //               rgba(255,255,255,0.01) 32%,
// // //               transparent 100%
// // //             )
// // //           `,
// // //           filter: "blur(14px)",
// // //           opacity: 0.45,
// // //         }}
// // //       />

// // //       {/* LEFT SIDE GLASS REFLECTION */}
// // //       <div
// // //         className="absolute left-[-8%] top-[-10%] w-[40%] h-[120%] z-[5] pointer-events-none"
// // //         style={{
// // //           background: `
// // //             linear-gradient(
// // //               90deg,
// // //               rgba(255,255,255,0.08),
// // //               transparent
// // //             )
// // //           `,
// // //           transform: "rotate(8deg)",
// // //           filter: "blur(26px)",
// // //           opacity: 0.16,
// // //         }}
// // //       />

// // //       {/* GRID */}
// // //       <div
// // //         className="absolute inset-0 z-[6] opacity-[0.22]"
// // //         style={{
// // //           backgroundImage: `
// // //             linear-gradient(rgba(0,255,120,0.10) 1px, transparent 1px),
// // //             linear-gradient(90deg, rgba(0,255,120,0.10) 1px, transparent 1px)
// // //           `,
// // //           backgroundSize: "48px 48px",
// // //         }}
// // //       />

// // //       {/* SUB GRID */}
// // //       <div
// // //         className="absolute inset-0 z-[7] opacity-[0.08]"
// // //         style={{
// // //           backgroundImage: `
// // //             linear-gradient(rgba(0,255,120,0.06) 1px, transparent 1px),
// // //             linear-gradient(90deg, rgba(0,255,120,0.06) 1px, transparent 1px)
// // //           `,
// // //           backgroundSize: "12px 12px",
// // //         }}
// // //       />

// // //       {/* SCANLINES */}
// // //       <div
// // //         className="absolute inset-0 z-[8] pointer-events-none"
// // //         style={{
// // //           backgroundImage: `
// // //             linear-gradient(
// // //               rgba(0,0,0,0.22) 1px,
// // //               transparent 2px
// // //             )
// // //           `,
// // //           backgroundSize: "100% 4px",
// // //           opacity: 0.34,
// // //         }}
// // //       />

// // //       {/* HORIZONTAL TUBE SHADOW */}
// // //       <div
// // //         className="absolute inset-0 z-[9] pointer-events-none"
// // //         style={{
// // //           background: `
// // //             linear-gradient(
// // //               to bottom,
// // //               rgba(0,0,0,0.35),
// // //               transparent 12%,
// // //               transparent 88%,
// // //               rgba(0,0,0,0.45)
// // //             )
// // //           `,
// // //         }}
// // //       />

// // //       {/* VERTICAL TUBE SHADOW */}
// // //       <div
// // //         className="absolute inset-0 z-[10] pointer-events-none"
// // //         style={{
// // //           background: `
// // //             linear-gradient(
// // //               90deg,
// // //               rgba(0,0,0,0.45),
// // //               transparent 10%,
// // //               transparent 90%,
// // //               rgba(0,0,0,0.55)
// // //             )
// // //           `,
// // //         }}
// // //       />

// // //       {/* NOISE / PHOSPHOR TEXTURE */}
// // //       <div
// // //         className="absolute inset-0 z-[11] opacity-[0.05] mix-blend-screen pointer-events-none"
// // //         style={{
// // //           backgroundImage:
// // //             "url('https://www.transparenttextures.com/patterns/asfalt-dark.png')",
// // //         }}
// // //       />

// // //       {/* SCREEN DUST */}
// // //       <div
// // //         className="absolute inset-0 z-[12] opacity-[0.03] pointer-events-none"
// // //         style={{
// // //           backgroundImage:
// // //             "url('https://www.transparenttextures.com/patterns/noise-pattern-with-subtle-cross-lines.png')",
// // //           mixBlendMode: "screen",
// // //         }}
// // //       />

// // //       {/* LIVE CONTENT */}
// // //       <div className="relative z-[20] w-full h-full">
// // //         {children}
// // //       </div>
// // //     </div>
// // //   )
// // // }
// // import React from "react"

// // interface Props {
// //   children?: React.ReactNode
// // }

// // export default function TetrisScreen({
// //   children,
// // }: Props) {
// //   return (
// //     <div
// //       className="
// //         crt-screen
// //         relative
// //         w-full
// //         h-full
// //         overflow-hidden
// //         rounded-[40px]
// //       "
// //       style={{
// //         background: `
// //           radial-gradient(
// //             circle at center,
// //             #071a0d 0%,
// //             #020705 60%,
// //             #000000 100%
// //           )
// //         `,

// //         border: "1px solid rgba(0,255,120,0.06)",

// //         boxShadow: `
// //           inset 0 0 140px rgba(0,0,0,1),
// //           inset 0 0 50px rgba(0,255,120,0.05),
// //           0 0 30px rgba(0,0,0,0.9)
// //         `,

// //         transform: `
// //           perspective(1800px)
// //           rotateX(1deg)
// //           scaleY(1.02)
// //         `,
// //       }}
// //     >
// //       {/* CENTER PHOSPHOR */}
// //       <div
// //         className="absolute inset-0 pointer-events-none"
// //         style={{
// //           background: `
// //             radial-gradient(
// //               circle at center,
// //               rgba(0,255,120,0.05),
// //               transparent 70%
// //             )
// //           `,
// //         }}
// //       />

// //       {/* GLASS REFLECTION */}
// //       <div
// //         className="
// //           absolute
// //           top-0
// //           left-0
// //           right-0
// //           h-[34%]
// //           pointer-events-none
// //         "
// //         style={{
// //           background: `
// //             linear-gradient(
// //               to bottom,
// //               rgba(255,255,255,0.07),
// //               transparent
// //             )
// //           `,
// //           filter: "blur(12px)",
// //           opacity: 0.45,
// //         }}
// //       />

// //       {/* SIDE REFLECTION */}
// //       <div
// //         className="
// //           absolute
// //           left-[-10%]
// //           top-[-10%]
// //           w-[40%]
// //           h-[120%]
// //           pointer-events-none
// //         "
// //         style={{
// //           background: `
// //             linear-gradient(
// //               90deg,
// //               rgba(255,255,255,0.05),
// //               transparent
// //             )
// //           `,
// //           transform: "rotate(8deg)",
// //           filter: "blur(20px)",
// //         }}
// //       />

// //       {/* CONTENT */}
// //       <div className="relative z-20 w-full h-full">
// //         {children}
// //       </div>
// //     </div>
// //   )
// // }

// import React from "react"

// export default function TetrisScreen({
//   children,
// }: {
//   children?: React.ReactNode
// }) {
//   return (
//     <div
//       className="
//         relative
//         w-full
//         h-full
//         overflow-hidden
//       "
//       style={{
//         background: `
//           radial-gradient(
//             circle at center,
//             rgba(0,60,25,0.45) 0%,
//             rgba(0,18,7,0.9) 40%,
//             #000000 100%
//           )
//         `,
//       }}
//     >
//       {/* CRT CURVATURE */}
//       <div
//         className="absolute inset-0"
//         style={{
//           borderRadius: "40px",

//           boxShadow: `
//             inset 0 0 120px rgba(0,0,0,0.95),
//             inset 0 0 40px rgba(0,180,80,0.02)
//           `,
//         }}
//       />

//       {/* SIDE DARKEN */}
//       <div
//         className="absolute inset-0"
//         style={{
//           background: `
//             radial-gradient(
//               ellipse at center,
//               transparent 35%,
//               rgba(0,0,0,0.45) 75%,
//               rgba(0,0,0,0.95) 100%
//             )
//           `,
//         }}
//       />

//       {/* CRT REFLECTION */}
//       <div
//         className="
//           absolute
//           top-0
//           left-0
//           w-full
//           h-[25%]
//           opacity-[0.12]
//         "
//         style={{
//           background: `
//             linear-gradient(
//               to bottom,
//               rgba(255,255,255,0.18),
//               transparent
//             )
//           `,
//           filter: "blur(10px)",
//         }}
//       />

//       {/* PHOSPHOR */}
//       <div
//         className="
//           absolute
//           inset-0
//           opacity-[0.07]
//         "
//         style={{
//           background: `
//             radial-gradient(
//               circle at center,
//               rgba(0,255,120,0.5),
//               transparent 60%
//             )
//           `,
//         }}
//       />

//       {/* SCANLINES */}
//       <div
//         className="
//           absolute
//           inset-0
//           opacity-[0.08]
//         "
//         style={{
//           backgroundImage: `
//             repeating-linear-gradient(
//               to bottom,
//               rgba(0,0,0,0.45) 0px,
//               rgba(0,0,0,0.45) 1px,
//               transparent 3px,
//               transparent 4px
//             )
//           `,
//         }}
//       />

//       <div className="relative z-10 w-full h-full">
//         {children}
//       </div>
//     </div>
//   )
// }

import "./styles/crt.css"

export default function TetrisScreen({
  children,
}: {
  children?: React.ReactNode
}) {
  return (
    <div
      className="
        crt-screen
        relative
        w-full
        h-full
        rounded-[42px]
        overflow-hidden
      "
      style={{
        background: `
          radial-gradient(
            circle at center,
            #031209 0%,
            #010804 45%,
            #000000 100%
          )
        `,

        boxShadow: `
          inset 0 0 140px rgba(0,0,0,1),
          inset 0 0 50px rgba(0,180,80,0.02)
        `,
      }}
    >
      {/* PHOSPHOR */}
      <div
        className="absolute inset-0 opacity-[0.16]"
        style={{
          background: `
            radial-gradient(
              circle at center,
              rgba(0,180,80,0.06),
              transparent 65%
            )
          `,
          filter: "blur(40px)",
        }}
      />

      {/* GLASS REFLECTION */}
      <div
        className="
          absolute
          top-0
          left-0
          right-0
          h-[34%]
          opacity-[0.12]
        "
        style={{
          background: `
            linear-gradient(
              to bottom,
              rgba(255,255,255,0.18),
              transparent
            )
          `,
          filter: "blur(16px)",
        }}
      />

      {/* CRT NOISE */}
      <div
        className="
          absolute
          inset-0
          opacity-[0.04]
          mix-blend-screen
        "
        style={{
          backgroundImage:
            "url('/src/app/tetris/assets/textures/SmudgesLarge001/SmudgesLarge001_OVERLAY_VAR1_4K.jpg')",
        }}
      />

      <div className="relative z-20 w-full h-full">
        {children}
      </div>
    </div>
  )
}