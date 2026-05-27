
// export default function SystemPanel({
// title,
// children,
// }: {
// title: string
// children: React.ReactNode
// }) {
// return (
// <div
// className="
// relative
// rounded-[24px]
// overflow-hidden
// border
// border-[#18472e]
// p-5
// "
// style={{
// background: `           linear-gradient(
//             145deg,             #071207,             #020502
//           )
//         `,
// boxShadow: `           inset 0 0 40px rgba(0,255,120,0.05),
//           inset 0 -20px 40px rgba(0,0,0,0.95),
//           0 0 20px rgba(0,255,120,0.03)
//         `
// }}
// >

//   <div
//     className="absolute inset-0 opacity-[0.08]"
//     style={{
//       background: `
//         radial-gradient(circle at top,
//           rgba(0,255,120,0.16),
//           transparent 60%)
//       `
//     }}
//   />

//   <div className="relative z-10">

//     <div
//       className="
//         text-[30px]
//         tracking-wide
//         mb-5
//         font-mono
//       "
//       style={{
//         color: "#0aff84",
//         textShadow: "0 0 12px rgba(0,255,120,0.5)"
//       }}
//     >
//       {title}
//     </div>

//     {children}
//   </div>
// </div>


// )
// }




export default function SystemPanel({
  title,
  children,
}: {
  title: string
  children?: React.ReactNode
}) {
  return (
    <div
      className="
        rounded-[18px]
        p-5
        relative
        overflow-hidden
      "
      style={{
        background: `
          linear-gradient(
            145deg,
            #060606,
            #010101
          )
        `,

        border: "1px solid rgba(0,180,80,0.04)",

        boxShadow: `
          inset 0 0 30px rgba(0,255,120,0.02),
          inset 0 0 80px rgba(0,0,0,1)
        `,
      }}
    >
      {/* DUST */}
      <div
        className="absolute inset-0 opacity-[0.05]"
        style={{
          backgroundImage:
            "url('/src/app/tetris/assets/textures/SmudgesLarge001/SmudgesLarge001_OVERLAY_VAR1_4K.jpg')",
        }}
      />

      <div
        style={{
          color: "#00aa55",
          fontFamily: "VT323",
          fontSize: "28px",
          letterSpacing: "2px",

          textShadow: `
            0 0 10px rgba(0,180,80,0.1)
          `,
        }}
      >
        {title}
      </div>

      <div className="mt-5 relative z-10">
        {children}
      </div>
    </div>
  )
}