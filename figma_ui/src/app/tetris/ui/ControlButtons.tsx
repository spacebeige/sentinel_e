// export function RedButton({
// label,
// }: {
// label: string
// }) {
// return ( <div className="flex flex-col items-center gap-3">


//   <button
//     className="
//       relative
//       w-[120px]
//       h-[120px]
//       rounded-full
//       overflow-hidden
//     "
//     style={{
//   background: `
//     radial-gradient(
//       circle at 35% 30%,
//       #ffb3b3,
//       #d10000 30%,
//       #3a0000 70%,
//       #120000 100%
//     )
//   `,

//   boxShadow: `
//     inset 0 8px 18px rgba(255,255,255,0.18),
//     inset 0 -20px 30px rgba(0,0,0,0.85),
//     0 0 18px rgba(255,0,0,0.35),
//     0 8px 30px rgba(0,0,0,0.8)
//   `,

//   border: "2px solid rgba(255,255,255,0.08)"
// }}
//   >

//     <div
//       className="
//         absolute
//         top-3
//         left-5
//         right-5
//         h-[35%]
//         rounded-full
//       "
//       style={{
//         background: "rgba(255,255,255,0.15)",
//         filter: "blur(10px)"
//       }}
//     />
//   </button>

//   <span className="text-[#86795f] text-[44px]">
//     {label}
//   </span>
// </div>


// )
// }

// export function DPad() {
// return ( <div className="relative w-[220px] h-[220px]">

//   <div
//     className="
//       absolute
//       left-[82px]
//       top-0
//       w-[56px]
//       h-full
//       rounded-xl
//     "
//     style={{
//       background: `
//         linear-gradient(
//           to right,
//           #050505,
//           #161616,
//           #050505
//         )
//       `,
//       border: "2px solid #2d2d2d",
//       boxShadow: `
//         inset 0 2px 8px rgba(255,255,255,0.05),
//         inset 0 -10px 18px rgba(0,0,0,1),
//         0 0 20px rgba(0,0,0,0.8)
//       `
//     }}
//   />

//   <div
//     className="
//       absolute
//       top-[82px]
//       left-0
//       h-[56px]
//       w-full
//       rounded-xl
//     "
//     style={{
//       background: `
//         linear-gradient(
//           to bottom,
//           #050505,
//           #161616,
//           #050505
//         )
//       `,
//       border: "2px solid #2d2d2d",
//       boxShadow: `
//         inset 0 2px 8px rgba(255,255,255,0.05),
//         inset 0 -10px 18px rgba(0,0,0,1),
//         0 0 20px rgba(0,0,0,0.8)
//       `
//     }}
//   />
// </div>


// )
// }

let audioUrl: string | null = null;
try {
  // Safe dynamic asset resolution fallback
  // audioUrl = new URL("../assets/sounds/button_click.mp3", import.meta.url).href;
  console.warn("[ASSET SYSTEM] Audio file missing. Initializing safe fallback.");
} catch (e) {
  console.warn("[ASSET SYSTEM] Failed to load audio:", e);
}
const audio = audioUrl ? new Audio(audioUrl) : null;

function Button({
  label,
}: {
  label: string
}) {
  return (
    <button
      onClick={() => {
        if (audio) {
          audio.currentTime = 0
          audio.play().catch(e => console.warn(e))
        }
      }}
      className="
        relative
        w-[120px]
        h-[120px]
        rounded-full
      "
      style={{
        background: `
          radial-gradient(circle at top, #3a0c0c, #110000)
        `,

        boxShadow: `
          inset 0 8px 14px rgba(255,255,255,0.05), inset 0 -16px 28px rgba(0,0,0,0.85), 0 0 16px rgba(0,0,0,0.9), 0 0 40px rgba(0,0,0,0.9)
        `,
      }}
    >
      {/* GLASS REFLECTION */}
      <div
        className="
          absolute
          top-[12px]
          left-[18px]
          w-[50%]
          h-[28%]
          rounded-full
        "
        style={{
          background:
            "linear-gradient(to bottom, rgba(255,255,255,0.42), transparent)",

          filter: "blur(4px)",
        }}
      />

      <div
        className="
          absolute
          -bottom-[58px]
          left-1/2
          -translate-x-1/2
        "
        style={{
          color: "#4a5c4a",
          fontSize: "52px",
          fontFamily: '"VT323", "ShareTechMono", monospace',
        }}
      >
        {label}
      </div>
    </button>
  )
}

export default function ControlButtons() {
  return (
    <>
      <div className="absolute right-[210px] bottom-[120px]">
        <Button label="B" />
      </div>

      <div className="absolute right-[60px] bottom-[100px]">
        <Button label="A" />
      </div>
    </>
  )
}