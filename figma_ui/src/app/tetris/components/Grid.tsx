// // import GlowBlock from "./GlowBlock"

// // const colors = [
// // "transparent",
// // "#28f0ff",
// // "#7dff32",
// // "#ff4040",
// // "#ffb300",
// // "#6f7dff",
// // ]

// // export default function Grid({
// // matrix,
// // }: {
// // matrix: number[][]
// // }) {
// // return ( <div className="grid grid-cols-10 gap-[2px]">

// // ```
// //   {matrix.flat().map((cell, i) => (
// //     <GlowBlock
// //       key={i}
// //       color={colors[cell]}
// //     />
// //   ))}

// // </div>

// // )
// // }

// import React from "react"
// import GlowBlock from "./GlowBlock"

// const COLORS: Record<number, string> = {
//   0: "none",
//   1: "cyan",
//   2: "green",
//   3: "yellow",
//   4: "red",
//   5: "blue",
//   6: "purple",
// }

// export default function Grid({
//   matrix,
// }: {
//   matrix: number[][]
// }) {
//   return (
//     <div
//       className="
//         relative
//         grid
//         grid-cols-10
//         gap-[3px]
//         p-[10px]
//       "
//       style={{
//         background: `
//           linear-gradient(
//             145deg,
//             rgba(0,0,0,0.92),
//             rgba(0,20,10,0.55)
//           )
//         `,

//         border: "1px solid rgba(0,255,120,0.06)",

//         boxShadow: `
//           inset 0 0 60px rgba(0,255,120,0.04),
//           inset 0 0 120px rgba(0,0,0,1)
//         `,
//       }}
//     >
//       {matrix.flat().map((cell, i) => (
//         <GlowBlock
//           key={i}
//           active={cell !== 0}
//           color={COLORS[cell]}
//         />
//       ))}

//       {/* CRT phosphor haze */}
//       <div
//         className="
//           absolute
//           inset-0
//           pointer-events-none
//         "
//         style={{
//           background: `
//             radial-gradient(
//               circle at center,
//               rgba(0,255,120,0.06),
//               transparent 70%
//             )
//           `,
//           mixBlendMode: "screen",
//         }}
//       />
//     </div>
//   )
// }

import React from "react"
import GlowBlock from "./GlowBlock"

const COLORS: Record<number, string> = {
  0: "none",
  1: "cyan",
  2: "green",
  3: "yellow",
  4: "red",
  5: "blue",
  6: "purple",
}

export default function Grid({
  matrix,
}: {
  matrix: number[][]
}) {
  return (
    <div
      className="
        relative
        grid
        grid-cols-10
        gap-[2px]
        p-[12px]
        rounded-[8px]
      "
      style={{
        background: `
          linear-gradient(
            145deg,
            rgba(0,0,0,0.96),
            rgba(0,20,10,0.45)
          )
        `,

        border: "1px solid rgba(0,255,120,0.04)",

        boxShadow: `
          inset 0 0 80px rgba(0,255,120,0.04),
          inset 0 0 220px rgba(0,0,0,1)
        `,
      }}
    >
      {matrix.flat().map((cell, i) => (
        <GlowBlock
          key={i}
          active={cell !== 0}
          color={COLORS[cell]}
        />
      ))}

      {/* CENTER TUBE GLOW */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: `
            radial-gradient(
              circle at center,
              rgba(0,255,120,0.04),
              transparent 72%
            )
          `,
        }}
      />
    </div>
  )
}