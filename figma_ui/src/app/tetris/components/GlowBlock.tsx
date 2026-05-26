// // // export default function GlowBlock({
// // // color,
// // // }: {
// // // color: string
// // // }) {
// // // return (
// // // <div
// // // className="
// // // w-[48px]
// // // h-[48px]
// // // border
// // // border-black/30
// // // "
// // // style={{
// // // background: color,
// // // boxShadow: `0 0 16px ${color}`
// // // }}
// // // />
// // // )
// // // }

// // import React from "react"

// // const COLORS: Record<string, string> = {
// //   cyan: "#7efcff",
// //   green: "#7dff8a",
// //   yellow: "#ffe16b",
// //   red: "#ff7a7a",
// //   blue: "#8da2ff",
// //   purple: "#c28cff",
// // }

// // export default function GlowBlock({
// //   color,
// //   active = true,
// // }: {
// //   color: string
// //   active?: boolean
// // }) {
// //   if (!active) {
// //     return (
// //       <div
// //         className="
// //           relative
// //           w-[46px]
// //           h-[46px]
// //           rounded-[2px]
// //           overflow-hidden
// //         "
// //         style={{
// //           background: `
// //             linear-gradient(
// //               145deg,
// //               rgba(0,20,10,0.85),
// //               rgba(0,0,0,1)
// //             )
// //           `,
// //           border: "1px solid rgba(0,255,120,0.04)",
// //         }}
// //       />
// //     )
// //   }

// //   const glow = COLORS[color] || "#7efcff"

// //   return (
// //     <div
// //       className="
// //         relative
// //         w-[46px]
// //         h-[46px]
// //         rounded-[3px]
// //         overflow-hidden
// //       "
// //       style={{
// //         background: `
// //           linear-gradient(
// //             145deg,
// //             rgba(255,255,255,0.22),
// //             ${glow} 18%,
// //             ${glow} 45%,
// //             rgba(0,0,0,0.82) 100%
// //           )
// //         `,

// //         border: `1px solid ${glow}22`,

// //         boxShadow: `
// //           0 0 10px ${glow}55,
// //           0 0 22px ${glow}22,
// //           inset 0 1px 2px rgba(255,255,255,0.28),
// //           inset 0 -8px 12px rgba(0,0,0,0.45)
// //         `,
// //       }}
// //     >
// //       {/* phosphor core */}
// //       <div
// //         className="
// //           absolute
// //           inset-[8px]
// //           rounded-[2px]
// //         "
// //         style={{
// //           background: `
// //             radial-gradient(
// //               circle at center,
// //               rgba(255,255,255,0.85),
// //               ${glow} 40%,
// //               rgba(0,0,0,0.15) 100%
// //             )
// //           `,
// //           filter: "blur(2px)",
// //           opacity: 0.9,
// //         }}
// //       />

// //       {/* glass reflection */}
// //       <div
// //         className="
// //           absolute
// //           left-[4px]
// //           right-[4px]
// //           top-[3px]
// //           h-[28%]
// //           rounded-full
// //         "
// //         style={{
// //           background:
// //             "linear-gradient(to bottom, rgba(255,255,255,0.35), transparent)",
// //           filter: "blur(3px)",
// //           opacity: 0.7,
// //         }}
// //       />

// //       {/* edge darkening */}
// //       <div
// //         className="absolute inset-0"
// //         style={{
// //           boxShadow: `
// //             inset 0 0 10px rgba(0,0,0,0.45)
// //           `,
// //         }}
// //       />
// //     </div>
// //   )
// // }



// import React from "react"

// const COLORS: Record<string, string> = {
//   cyan: "#8cf7ff",
//   green: "#84ff7c",
//   yellow: "#ffe27d",
//   red: "#ff8484",
//   blue: "#8da0ff",
//   purple: "#d193ff",
// }

// export default function GlowBlock({
//   color,
//   active = true,
// }: {
//   color: string
//   active?: boolean
// }) {
//   if (!active) {
//     return (
//       <div
//         className="
//           relative
//           w-[46px]
//           h-[46px]
//           overflow-hidden
//         "
//         style={{
//           background: `
//             linear-gradient(
//               145deg,
//               rgba(0,15,5,0.7),
//               rgba(0,0,0,1)
//             )
//           `,
//           border: "1px solid rgba(0,255,120,0.03)",
//         }}
//       />
//     )
//   }

//   const glow = COLORS[color] || "#8cf7ff"

//   return (
//     <div
//       className="
//         relative
//         w-[46px]
//         h-[46px]
//         overflow-hidden
//       "
//       style={{
//         background: `
//           linear-gradient(
//             145deg,
//             rgba(255,255,255,0.18),
//             ${glow} 18%,
//             rgba(0,0,0,0.4) 100%
//           )
//         `,

//         border: `1px solid ${glow}22`,

//         boxShadow: `
//           0 0 8px ${glow}55,
//           0 0 18px ${glow}22,
//           inset 0 1px 2px rgba(255,255,255,0.22),
//           inset 0 -10px 14px rgba(0,0,0,0.55)
//         `,

//         opacity:
//           Math.random() > 0.985
//             ? 0.72
//             : 1
//       }}
//     >
//       {/* PHOSPHOR CORE */}
//       <div
//         className="
//           absolute
//           inset-[9px]
//         "
//         style={{
//           background: `
//             radial-gradient(
//               circle at center,
//               rgba(255,255,255,0.92),
//               ${glow} 45%,
//               transparent 100%
//             )
//           `,
//           filter: "blur(2px)",
//           opacity: 0.9,
//         }}
//       />

//       {/* GLASS REFLECTION */}
//       <div
//         className="
//           absolute
//           top-[3px]
//           left-[4px]
//           right-[4px]
//           h-[28%]
//         "
//         style={{
//           background:
//             "linear-gradient(to bottom, rgba(255,255,255,0.22), transparent)",
//           filter: "blur(3px)",
//         }}
//       />

//       {/* EDGE DARKEN */}
//       <div
//         className="absolute inset-0"
//         style={{
//           boxShadow: `
//             inset 0 0 12px rgba(0,0,0,0.45)
//           `,
//         }}
//       />
//     </div>
//   )
// }

const COLORS: Record<string, string> = {
  cyan: "#89f7ff",
  green: "#8cff74",
  yellow: "#ffd95e",
  red: "#ff6868",
  blue: "#7ea7ff",
  purple: "#cc88ff",
}

export default function GlowBlock({
  color,
  active = true,
}: {
  color: string
  active?: boolean
}) {
  if (!active) {
    return (
      <div
        className="w-[42px] h-[42px]"
        style={{
          background: `
            linear-gradient(
              145deg,
              rgba(0,18,8,0.55),
              rgba(0,0,0,1)
            )
          `,
          border: "1px solid rgba(0,255,120,0.03)",
        }}
      />
    )
  }

  const glow = COLORS[color]

  return (
    <div
      className="
        relative
        w-[42px]
        h-[42px]
      "
      style={{
        background: `
          linear-gradient(
            145deg,
            rgba(255,255,255,0.16),
            ${glow} 18%,
            rgba(0,0,0,0.4) 100%
          )
        `,

        border: `1px solid ${glow}44`,

        boxShadow: `
          inset 0 2px 4px rgba(255,255,255,0.24),
          inset 0 -10px 20px rgba(0,0,0,0.55),
          0 0 10px ${glow}33
        `,
      }}
    >
      <div
        className="
          absolute
          top-[4px]
          left-[5px]
          right-[5px]
          h-[25%]
        "
        style={{
          background:
            "linear-gradient(to bottom, rgba(255,255,255,0.24), transparent)",
          filter: "blur(2px)",
        }}
      />
    </div>
  )
}