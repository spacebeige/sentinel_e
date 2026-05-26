import TetrisScreen from "./TetrisScreen"
import SystemPanel from "./SystemPanel"
import ControlButtons from "./ControlButtons"

const matrix = [
[0,0,0,0,0,0,0,0,0,0],
[1,2,3,4,4,4,4,0,0,0],
[1,2,3,4,4,5,4,0,0,0],
[1,2,3,4,5,5,0,0,0,0],
[1,2,3,4,0,0,0,0,0,0],
[1,2,3,0,0,0,0,0,0,0],
[1,1,0,0,0,0,0,0,0,0],
]

const COLORS: Record<number,string> = {
0: "transparent",
1: "#28f0ff",
2: "#6f7dff",
3: "#ad5cff",
4: "#7dff32",
5: "#ffb300",
}

export default function TetrisConsole() {
return ( <div
   className="
     min-h-screen
     w-full
     bg-black
     flex
     items-center
     justify-center
     p-10
   "
 >


  <div
    className="
      relative
      w-[1400px]
      h-[980px]
      rounded-[42px]
      overflow-hidden
      border
      border-[#403c33]
    "
    style={{
      background: `
        linear-gradient(
          145deg,
          #302c26,
          #171511,
          #090909
        )
      `,
      boxShadow: `
        inset 0 0 120px rgba(255,255,255,0.03),
        inset 0 -120px 180px rgba(0,0,0,0.95),
        0 0 80px rgba(0,255,120,0.08)
      `
    }}
  >

    <div
      className="absolute inset-0 opacity-[0.08]"
      style={{
        backgroundImage:
          "url('https://www.transparenttextures.com/patterns/asfalt-dark.png')"
      }}
    />

    <div className="flex h-full p-8 gap-8">

      {/* LEFT CRT */}
      <div className="flex-1 flex flex-col">

        <div
          className="
            flex-1
            rounded-[40px]
            p-8
            border
            border-[#2d2d2d]
          "
          style={{
            background: `
              linear-gradient(
                145deg,
                #040404,
                #0a0a0a
              )
            `,
            boxShadow: `
              inset 0 0 60px rgba(255,255,255,0.03),
              inset 0 -50px 120px rgba(0,0,0,1)
            `
          }}
        >

          <TetrisScreen>

            <div className="absolute top-8 left-8 text-[#0aff84] font-mono text-[26px] tracking-widest">
              SENTINEL COGNITION TERMINAL v2.7
            </div>

            <div className="absolute top-8 right-8 text-[#0aff84] font-mono text-[26px]">
              ONLINE ●
            </div>

            <div
              className="
                absolute
                left-[80px]
                top-[110px]
                grid
                grid-cols-10
                gap-[2px]
              "
            >
              {Array.from({ length: 20 }).map((_, y) =>
                Array.from({ length: 10 }).map((_, x) => {
                  const row = matrix[19 - y]
                  const value = row?.[x] ?? 0

                  return (
                    <div
                      key={`${x}-${y}`}
                      className="w-[48px] h-[48px] border border-[#073a1d]"
                      style={{
                        background:
                          value === 0
                            ? "rgba(0,255,120,0.03)"
                            : COLORS[value],
                        boxShadow:
                          value === 0
                            ? "none"
                            : `0 0 16px ${COLORS[value]}`
                      }}
                    />
                  )
                })
              )}
            </div>

          </TetrisScreen>
        </div>

        {/* BOTTOM CONTROLS */}
        <div className="h-[240px] mt-6 flex justify-between items-center px-8">

          <ControlButtons />

          <div
            className="
              w-[320px]
              h-[160px]
              rounded-[20px]
              border
              border-[#2f2b25]
              bg-[#111]
              flex
              flex-col
              justify-center
              items-center
            "
            style={{
              boxShadow: `
                inset 0 0 20px rgba(255,255,255,0.03),
                inset 0 -20px 40px rgba(0,0,0,1)
              `
            }}
          >
            <div className="text-[#7c735f] text-[40px] tracking-widest">
              SENTINEL
            </div>

            <div className="text-[#5b5647] text-[22px] mt-2">
              ORCHESTRATION UNIT
            </div>
          </div>

        </div>
      </div>

      {/* RIGHT PANELS */}
      <div className="w-[360px] flex flex-col gap-5">

        <SystemPanel title="COHERENCE">
          <div className="flex gap-2">
            {Array.from({ length: 20 }).map((_, i) => (
              <div
                key={i}
                className="w-[12px] h-[40px]"
                style={{
                  background:
                    i < 14
                      ? "#0aff84"
                      : "#062d18",
                  boxShadow:
                    i < 14
                      ? "0 0 12px #0aff84"
                      : "none"
                }}
              />
            ))}
          </div>
        </SystemPanel>

        <SystemPanel title="MEMORY">
          <div className="text-[#0aff84] text-[42px] font-mono">
            12
          </div>
        </SystemPanel>

        <SystemPanel title="DIVERGENCE">
          <div className="text-[#0aff84] text-[42px] font-mono">
            03%
          </div>
        </SystemPanel>

        <SystemPanel title="EVENT LOG">
          <div className="space-y-3 text-[#0aff84] font-mono text-[18px]">
            <div>Routing stable</div>
            <div>Memory pipeline synced</div>
            <div>Arbitration complete</div>
            <div>Divergence low</div>
          </div>
        </SystemPanel>

      </div>
    </div>
  </div>
</div>


)
}
