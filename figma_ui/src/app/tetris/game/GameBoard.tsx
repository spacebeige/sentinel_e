// import React from 'react';

// export default function GameBoard() {
//   return (
//     <div
//       className="
//         w-full
//         h-full
//         grid
//         grid-cols-10
//         grid-rows-20
//         gap-[1px]
//         p-[2px]
//       "
//     >
//       {Array.from({ length: 200 }).map((_, i) => (
//         <div
//           key={i}
//           className="
//             bg-[#0f220f]
//             border
//             border-green-900/10
//           "
//         />
//       ))}
//     </div>
//   )
// }

// FILE:
// figma_ui/src/app/tetris/game/GameBoard.tsx

// import Grid from "../components/Grid"

// const matrix = Array.from({
// length: 20
// }).map(() =>
// Array(10).fill(0)
// )

// matrix[18][4] = 1
// matrix[17][4] = 1
// matrix[16][4] = 1
// matrix[16][5] = 1

// export default function GameBoard() {
// return ( <div className="absolute left-[80px] top-[120px]"> <Grid matrix={matrix} /> </div>
// )
// }


import React, { useMemo } from "react"
import Grid from "../components/Grid"

export default function GameBoard() {
  const matrix = useMemo(() => {
    const board = Array.from({ length: 20 }, () =>
      Array(10).fill(0)
    )

    // LEFT CYAN STACK
    board[19][0] = 1
    board[18][0] = 1
    board[17][0] = 1
    board[16][0] = 1

    board[19][1] = 1
    board[18][1] = 1
    board[17][1] = 1
    board[16][1] = 1
    board[15][1] = 1

    // PURPLE CLUSTER
    board[19][2] = 2
    board[18][2] = 2
    board[17][2] = 2

    board[18][3] = 2

    // YELLOW TOWER
    board[19][4] = 3
    board[18][4] = 3
    board[17][4] = 3
    board[16][4] = 3
    board[15][4] = 3

    board[18][5] = 3
    board[17][5] = 3

    // GREEN BASE
    board[19][5] = 4
    board[19][6] = 4
    board[18][6] = 4
    board[19][7] = 4

    // BLUE RIGHT SIDE
    board[19][8] = 5
    board[18][8] = 5
    board[17][8] = 5

    board[16][9] = 5
    board[17][9] = 5
    board[18][9] = 5
    board[19][9] = 5

    // FLOATING ACTIVE PIECE
    board[8][4] = 6
    board[8][5] = 6
    board[9][5] = 6
    board[10][5] = 6

    return board
  }, [])

  return (
    <div
      className="
        absolute
        left-[170px]
        top-[110px]
        w-[560px]
        h-[620px]
      "
    >
      {/* SCREEN HEADER */}
      <div
        className="
          absolute
          top-[-38px]
          left-0
          right-0
          flex
          items-center
          justify-between
          text-[#00ff99]
          text-[18px]
          tracking-[2px]
          font-mono
          opacity-90
        "
      >
        <span>SENTINEL COGNITION TERMINAL v2.7</span>

        <div className="flex items-center gap-3">
          <span>ONLINE</span>

          <div
            className="
              w-3
              h-3
              rounded-full
              bg-[#00ff88]
              shadow-[0_0_12px_#00ff88]
            "
          />
        </div>
      </div>

      {/* MAIN GRID */}
      <div
        className="
          relative
          w-full
          h-full
          border
          border-[#00ff88]/40
          overflow-hidden
        "
        style={{
          background: `
            radial-gradient(
              circle at center,
              rgba(0,255,120,0.10) 0%,
              rgba(0,255,120,0.03) 30%,
              rgba(0,0,0,0.82) 100%
            )
          `,
          boxShadow: `
            inset 0 0 80px rgba(0,255,120,0.08),
            inset 0 0 160px rgba(0,0,0,0.95),
            0 0 20px rgba(0,255,120,0.08)
          `,
        }}
      >
        {/* LEFT NUMBERS */}
        <div
  className="
    absolute
    left-[150px]
    top-[120px]
  "
  style={{
    transform: `
      perspective(1400px)
      rotateX(1deg)
      scaleY(1.01)
    `,
  }}
>
  <Grid matrix={matrix} />
</div>

        {/* CRT CENTER GLOW */}
        <div
          className="
            absolute
            inset-0
            pointer-events-none
          "
          style={{
            background: `
              radial-gradient(
                circle at center,
                rgba(0,255,120,0.12),
                transparent 65%
              )
            `,
            mixBlendMode: "screen",
          }}
        />
      </div>

      {/* BOTTOM STATUS */}
      <div
        className="
          absolute
          bottom-[-42px]
          left-0
          right-0
          flex
          items-center
          justify-between
          text-[#00ff88]
          text-[16px]
          tracking-[2px]
          font-mono
          opacity-90
        "
      >
        <span>GRID: 10x20</span>

        <span>CYCLE: 00:01:42</span>
      </div>
    </div>
  )
}