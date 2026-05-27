import './styles/crt.css';
// // // src/app/tetris/ui/TerminalShell.tsx

// // import React from "react"

// // type TerminalShellProps = {
// //   children: React.ReactNode
// // }

// // export default function TerminalShell({
// //   children,
// // }: TerminalShellProps) {
// //   return (
// //     <div className="w-screen h-screen bg-black overflow-hidden flex items-center justify-center p-8">

// //       {/* OUTER MACHINE */}
// //       <div
// //         className="
// //           relative
// //           w-[1450px]
// //           h-[980px]
// //           rounded-[40px]
// //           border
// //           border-[#283128]
// //           bg-[#111411]
// //           shadow-[0_0_120px_rgba(0,255,120,0.08)]
// //           overflow-hidden
// //         "
// //       >

// //         {/* METAL TEXTURE OVERLAY */}
// //         <div
// //           className="
// //             absolute
// //             inset-0
// //             opacity-[0.04]
// //             pointer-events-none
// //             mix-blend-overlay
// //           "
// //           style={{
// //             backgroundImage:
// //               "radial-gradient(circle at 20% 20%, white 1px, transparent 1px)",
// //             backgroundSize: "8px 8px",
// //           }}
// //         />

// //         {/* INNER LAYOUT */}
// //         <div className="flex w-full h-full p-8 gap-8">

// //           {/* LEFT CRT DISPLAY */}
// //           <div
// //             className="
// //               relative
// //               flex-1
// //               rounded-[28px]
// //               border
// //               border-[#1d2a1d]
// //               bg-[071207]
// //               overflow-hidden
// //               shadow-inner
// //             "
// //           >

// //             {/* CRT Glow */}
// //             <div
// //               className="
// //                 absolute
// //                 inset-0
// //                 pointer-events-none
// //                 opacity-20
// //                 bg-[radial-gradient(circle_at_center,rgba(0,255,120,0.18),transparent_70%)]
// //               "
// //             />

// //             {/* Scanlines */}
// //             <div
// //               className="
// //                 absolute
// //                 inset-0
// //                 pointer-events-none
// //                 opacity-10
// //                 bg-[linear-gradient(rgba(0,255,100,0.08)_1px,transparent_1px)]
// //                 bg-[size:100%_4px]
// //               "
// //             />

// //             {/* CRT Curvature */}
// //             <div
// //               className="
// //                 absolute
// //                 inset-0
// //                 pointer-events-none
// //                 rounded-[28px]
// //                 shadow-[inset_0_0_80px_rgba(0,0,0,0.9)]
// //               "
// //             />

// //             {/* HEADER */}
// //             <div className="absolute top-6 left-8 right-8 flex justify-between text-green-400 font-mono text-xl z-30 tracking-widest">
// //               <span>SENTINEL COGNITION TERMINAL v2.7</span>

// //               <div className="flex items-center gap-3">
// //                 <span>ONLINE</span>

// //                 <div className="w-3 h-3 rounded-full bg-green-400 animate-pulse" />
// //               </div>
// //             </div>

// //             {/* GRID AREA */}
// //             <div className="absolute inset-0 pt-24 pb-12 px-10 z-20">
// //               {children}
// //             </div>

// //             {/* FOOTER */}
// //             <div className="absolute bottom-4 left-8 right-8 flex justify-between text-green-500 text-lg font-mono z-30">
// //               <span>GRID: 10x20</span>
// //               <span>CYCLE: 00:01:42</span>
// //             </div>
// //           </div>

// //           {/* RIGHT SIDEBAR */}
// //           <div className="w-[360px] flex flex-col gap-5">

// //             {/* COHERENCE */}
// //             <Panel title="COHERENCE">
// //               <ProgressBar value={14} max={20} />
// //               <div className="mt-3 text-green-400 text-4xl font-mono">
// //                 0002574
// //               </div>
// //             </Panel>

// //             {/* MEMORY */}
// //             <Panel title="MEMORY">
// //               <ProgressBar value={5} max={20} />
// //               <div className="mt-3 flex justify-between items-center">
// //                 <span className="text-green-400 text-4xl font-mono">12</span>

// //                 <span className="text-green-700 text-xl font-mono">
// //                   /20
// //                 </span>
// //               </div>
// //             </Panel>

// //             {/* DIVERGENCE */}
// //             <Panel title="DIVERGENCE">
// //               <ProgressBar value={2} max={20} />

// //               <div className="mt-3 flex justify-between items-center">
// //                 <span className="text-green-400 text-4xl font-mono">03</span>

// //                 <span className="text-green-700 text-xl font-mono">
// //                   %
// //                 </span>
// //               </div>
// //             </Panel>

// //             {/* NEXT NODE */}
// //             <Panel title="NEXT NODE">
// //               <div className="w-full h-[140px] flex items-center justify-center">

// //                 {/* T PIECE */}
// //                 <div className="grid grid-cols-3 gap-[4px]">
// //                   <Block />
// //                   <Block />
// //                   <Block />

// //                   <div />
// //                   <Block />
// //                   <div />
// //                 </div>
// //               </div>
// //             </Panel>

// //             {/* STATUS */}
// //             <Panel title="SYSTEM STATUS">
// //               <div className="space-y-3 text-green-400 text-lg font-mono">
// //                 <StatusRow label="[ROUTING]" value="Stable" />
// //                 <StatusRow label="[COUNCIL]" value="Active" />
// //                 <StatusRow label="[MEMORY]" value="Synced" />
// //                 <StatusRow label="[ARBITER]" value="Online" />
// //               </div>
// //             </Panel>

// //             {/* EVENT LOG */}
// //             <Panel title="EVENT LOG">
// //               <div className="space-y-2 text-[14px] font-mono text-green-500 h-[140px] overflow-hidden">
// //                 <LogRow time="00:01:41" text="Routing stable" />
// //                 <LogRow time="00:01:39" text="Memory pipeline synced" />
// //                 <LogRow time="00:01:36" text="Arbitration complete" />
// //                 <LogRow time="00:01:32" text="Divergence low" />
// //                 <LogRow time="00:01:29" text="Council consensus" />
// //               </div>
// //             </Panel>
// //           </div>
// //         </div>

// //         {/* LOWER CONTROLS */}
// //         <div className="absolute bottom-8 left-8 right-8 flex justify-between items-end">

// //           {/* DPAD */}
// //           <div className="relative w-[180px] h-[180px]">

// //             <div className="absolute left-[64px] top-0 w-[52px] h-full rounded-xl bg-[#090d09] border border-[#1d311d]" />

// //             <div className="absolute top-[64px] left-0 h-[52px] w-full rounded-xl bg-[#090d09] border border-[#1d311d]" />

// //             <div className="absolute inset-0 flex items-center justify-center text-green-700 text-2xl font-bold">
// //               +
// //             </div>
// //           </div>

// //           {/* CENTER PANEL */}
// //           <div className="flex flex-col items-center gap-6">

// //             <div
// //               className="
// //                 w-[340px]
// //                 h-[130px]
// //                 rounded-2xl
// //                 border
// //                 border-[#2d322d]
// //                 bg-[#0b0d0b]
// //                 flex
// //                 flex-col
// //                 items-center
// //                 justify-center
// //                 text-green-700
// //                 font-mono
// //               "
// //             >
// //               <div className="text-3xl tracking-widest">
// //                 SENTINEL
// //               </div>

// //               <div className="text-xl opacity-70">
// //                 ORCHESTRATION UNIT
// //               </div>

// //               <div className="mt-4 text-sm opacity-50">
// //                 MODEL: ST-7
// //               </div>

// //               <div className="text-sm opacity-50">
// //                 SERIAL: 8847-AI
// //               </div>
// //             </div>

// //             {/* SELECT START */}
// //             <div className="flex gap-8">
// //               <ControlButton label="SELECT" />
// //               <ControlButton label="START" />
// //             </div>
// //           </div>

// //           {/* AB BUTTONS */}
// //           <div className="flex gap-10 mb-8">

// //             <ActionButton label="B" />

// //             <ActionButton label="A" />
// //           </div>
// //         </div>
// //       </div>
// //     </div>
// //   )
// // }

// // /* -------------------------------- */
// // /* COMPONENTS */
// // /* -------------------------------- */

// // function Panel({
// //   title,
// //   children,
// // }: {
// //   title: string
// //   children: React.ReactNode
// // }) {
// //   return (
// //     <div
// //       className="
// //         rounded-2xl
// //         border
// //         border-[#183018]
// //         bg-[071207]
// //         p-5
// //         shadow-inner
// //       "
// //     >
// //       <div className="text-green-400 text-3xl font-mono mb-4 tracking-wide">
// //         {title}
// //       </div>

// //       {children}
// //     </div>
// //   )
// // }

// // function ProgressBar({
// //   value,
// //   max,
// // }: {
// //   value: number
// //   max: number
// // }) {
// //   return (
// //     <div className="grid grid-cols-20 gap-[3px]">
// //       {Array.from({ length: max }).map((_, i) => (
// //         <div
// //           key={i}
// //           className={`
// //             h-10
// //             border
// //             border-[#164216]
// //             ${
// //               i < value
// //                 ? "bg-green-400 shadow-[0_0_14px_rgba(0,255,120,0.8)]"
// //                 : "bg-[#061106]"
// //             }
// //           `}
// //         />
// //       ))}
// //     </div>
// //   )
// // }

// // function Block() {
// //   return (
// //     <div
// //       className="
// //         w-14
// //         h-14
// //         border
// //         border-green-400
// //         bg-green-500/70
// //         shadow-[0_0_20px_rgba(180,100,255,0.8)]
// //       "
// //     />
// //   )
// // }

// // function StatusRow({
// //   label,
// //   value,
// // }: {
// //   label: string
// //   value: string
// // }) {
// //   return (
// //     <div className="flex justify-between">
// //       <span>{label}</span>
// //       <span>{value}</span>
// //     </div>
// //   )
// // }

// // function LogRow({
// //   time,
// //   text,
// // }: {
// //   time: string
// //   text: string
// // }) {
// //   return (
// //     <div className="flex gap-4">
// //       <span className="text-green-700">{time}</span>
// //       <span>{text}</span>
// //     </div>
// //   )
// // }

// // function ControlButton({
// //   label,
// // }: {
// //   label: string
// // }) {
// //   return (
// //     <div className="flex flex-col items-center gap-2">
// //       <button
// //         className="
// //           w-[110px]
// //           h-[28px]
// //           rounded-full
// //           bg-[#050805]
// //           border
// //           border-[#173017]
// //           shadow-inner
// //         "
// //       />

// //       <span className="text-green-700 font-mono tracking-widest">
// //         {label}
// //       </span>
// //     </div>
// //   )
// // }

// // function ActionButton({
// //   label,
// // }: {
// //   label: string
// // }) {
// //   return (
// //     <div className="flex flex-col items-center gap-3">
// //       <button
// //         className="
// //           w-[110px]
// //           h-[110px]
// //           rounded-full
// //           bg-red-900
// //           border
// //           border-red-700
// //           shadow-[0_0_40px_rgba(255,0,0,0.35)]
// //           active:translate-y-[4px]
// //           transition-all
// //         "
// //       />

// //       <span className="text-green-700 text-4xl font-mono">
// //         {label}
// //       </span>
// //     </div>
// //   )
// // }
// // src/app/tetris/ui/TerminalShell.tsx

// import React from "react"

// type Props = {
//   children: React.ReactNode
// }

// export default function TerminalShell({ children }: Props) {
//   return (
//     <div className="w-screen h-screen bg-black flex items-center justify-center overflow-hidden">

//       {/* MACHINE BODY */}
//       <div
//         className="
//           relative
//           w-[1380px]
//           h-[980px]
//           rounded-[42px]
//           overflow-hidden
//         "
//         style={{
//           background: `
//             radial-gradient(circle at top,
//             rgba(120,100,70,0.14),
//             rgba(15,15,15,1) 28%),

//             linear-gradient(
//               145deg,
//               #34312c 0%,
//               #181818 18%,
//               #050505 55%,
//               #1d1d1d 100%
//             )
//           `,

//           boxShadow: `
//             inset 0 0 0 2px rgba(255,255,255,0.03),
//             inset 0 0 40px rgba(255,255,255,0.02),
//             inset 0 0 140px rgba(0,0,0,0.95),
//             inset 0 -40px 120px rgba(0,0,0,0.9),
//             0 0 120px rgba(0,0,0,0.95)
//           `
//         }}
//       >

//         {/* METAL GRAIN */}
//         <div
//           className="absolute inset-0 opacity-[0.05] pointer-events-none"
//           style={{
//             backgroundImage: `
//               repeating-linear-gradient(
//                 90deg,
//                 rgba(255,255,255,0.02) 0px,
//                 rgba(255,255,255,0.02) 1px,
//                 transparent 2px,
//                 transparent 4px
//               )
//             `
//           }}
//         />

//         {/* SCREWS */}
//         {[
//           "top-5 left-5",
//           "top-5 right-5",
//           "bottom-5 left-5",
//           "bottom-5 right-5",
//         ].map((pos) => (
//           <div
//             key={pos}
//             className={`absolute ${pos} w-8 h-8 rounded-full`}
//             style={{
//               background: `
//                 radial-gradient(circle at 30% 30%,
//                 #555,
//                 #1b1b1b 55%,
//                 #050505 100%)
//               `,
//               boxShadow: `
//                 inset 0 2px 4px rgba(255,255,255,0.1),
//                 inset 0 -4px 8px rgba(0,0,0,0.95)
//               `
//             }}
//           />
//         ))}

//         {/* MAIN */}
//         <div className="flex h-full p-10 gap-8">

//           {/* CRT */}
//           <div
//             className="
//               relative
//               flex-1
//               rounded-[40px]
//               overflow-hidden
//             "
//             style={{
//               background: `
//                 linear-gradient(
//                   145deg,
//                   #060606,
//                   #000000 55%,
//                   #121212
//                 )
//               `,

//               boxShadow: `
//                 inset 0 0 0 2px rgba(255,255,255,0.03),
//                 inset 0 0 80px rgba(0,0,0,1),
//                 inset 0 0 140px rgba(0,0,0,1),
//                 0 0 40px rgba(0,0,0,0.95)
//               `
//             }}
//           >

//             {/* INNER CRT */}
//             <div
//               className="
//                 absolute
//                 left-[28px]
//                 top-[28px]
//                 right-[28px]
//                 bottom-[28px]
//                 rounded-[30px]
//                 overflow-hidden
//               "
//               style={{
//                 background: `
//                   radial-gradient(
//                     circle at center,
//                     rgba(0,255,120,0.12),
//                     rgba(0,0,0,0.98) 72%
//                   )
//                 `,

//                 border: "1px solid rgba(80,255,180,0.08)",

//                 transform: `
//                   perspective(1400px)
//                   rotateX(1deg)
//                 `,

//                 boxShadow: `
//                   inset 0 0 40px rgba(0,255,120,0.06),
//                   inset 0 0 120px rgba(0,0,0,1)
//                 `
//               }}
//             >

//               {/* CRT GLOW */}
//               <div
//                 className="absolute inset-0 pointer-events-none"
//                 style={{
//                   background: `
//                     radial-gradient(
//                       circle at center,
//                       rgba(0,255,120,0.08),
//                       transparent 60%
//                     )
//                   `
//                 }}
//               />

//               {/* SCANLINES */}
//               <div
//                 className="absolute inset-0 opacity-[0.18] pointer-events-none"
//                 style={{
//                   background: `
//                     repeating-linear-gradient(
//                       to bottom,
//                       rgba(0,255,120,0.04) 0px,
//                       rgba(0,255,120,0.04) 1px,
//                       transparent 2px,
//                       transparent 4px
//                     )
//                   `
//                 }}
//               />

//               {/* VIGNETTE */}
//               <div
//                 className="absolute inset-0 pointer-events-none"
//                 style={{
//                   boxShadow: `
//                     inset 0 0 120px rgba(0,0,0,1)
//                   `
//                 }}
//               />

//               {/* HEADER */}
//               <div className="absolute top-6 left-8 right-8 flex justify-between z-50 text-[#22ff88] font-mono text-[28px] tracking-wider">

//                 <span>
//                   SENTINEL COGNITION TERMINAL v2.7
//                 </span>

//                 <div className="flex items-center gap-3">
//                   <span>ONLINE</span>

//                   <div
//                     className="w-4 h-4 rounded-full"
//                     style={{
//                       background: "#22ff88",
//                       boxShadow: "0 0 14px #22ff88"
//                     }}
//                   />
//                 </div>
//               </div>

//               {/* CONTENT */}
//               <div className="absolute inset-0 pt-24 pb-20 px-12">
//                 {children}
//               </div>

//               {/* FOOTER */}
//               <div className="absolute bottom-5 left-8 right-8 flex justify-between text-[#18cc66] text-xl font-mono">
//                 <span>GRID: 10x20</span>
//                 <span>CYCLE: 00:01:42</span>
//               </div>
//             </div>
//           </div>

//           {/* RIGHT PANELS */}
//           <div className="w-[340px] flex flex-col gap-5">

//             <SidePanel title="COHERENCE">
//               <Meter value={14} />
//               <div className="mt-4 text-[#22ff88] text-5xl font-mono">
//                 0002574
//               </div>
//             </SidePanel>

//             <SidePanel title="MEMORY">
//               <Meter value={5} />

//               <div className="mt-4 flex justify-between">
//                 <div className="text-[#22ff88] text-5xl font-mono">
//                   12
//                 </div>

//                 <div className="text-[#1c7f44] text-2xl font-mono">
//                   /20
//                 </div>
//               </div>
//             </SidePanel>

//             <SidePanel title="DIVERGENCE">
//               <Meter value={2} />

//               <div className="mt-4 flex justify-between">
//                 <div className="text-[#22ff88] text-5xl font-mono">
//                   03
//                 </div>

//                 <div className="text-[#1c7f44] text-2xl font-mono">
//                   %
//                 </div>
//               </div>
//             </SidePanel>

//             <SidePanel title="EVENT LOG">
//               <div className="space-y-3 text-[#22ff88] text-sm font-mono">
//                 <div>00:01:41 Routing stable</div>
//                 <div>00:01:39 Memory synced</div>
//                 <div>00:01:36 Arbitration complete</div>
//                 <div>00:01:32 Divergence low</div>
//               </div>
//             </SidePanel>
//           </div>
//         </div>

//         {/* BOTTOM */}
//         <div className="absolute bottom-10 left-10 right-10 flex justify-between items-end">

//           {/* DPAD */}
//           <div className="relative w-[220px] h-[220px]">

//             <div
//               className="absolute left-[82px] top-0 w-[56px] h-full rounded-xl"
//               style={{
//                 background: `
//                   linear-gradient(
//                     to right,
//                     #020202,
//                     #151515,
//                     #020202
//                   )
//                 `,
//                 boxShadow: `
//                   inset 0 4px 8px rgba(255,255,255,0.06),
//                   inset 0 -12px 18px rgba(0,0,0,0.95),
//                   0 8px 18px rgba(0,0,0,0.95)
//                 `
//               }}
//             />

//             <div
//               className="absolute top-[82px] left-0 h-[56px] w-full rounded-xl"
//               style={{
//                 background: `
//                   linear-gradient(
//                     to bottom,
//                     #020202,
//                     #151515,
//                     #020202
//                   )
//                 `,
//                 boxShadow: `
//                   inset 0 4px 8px rgba(255,255,255,0.06),
//                   inset 0 -12px 18px rgba(0,0,0,0.95),
//                   0 8px 18px rgba(0,0,0,0.95)
//                 `
//               }}
//             />
//           </div>

//           {/* CENTER */}
//           <div className="flex flex-col items-center gap-6">

//             <div
//               className="
//                 w-[360px]
//                 h-[160px]
//                 rounded-2xl
//                 relative
//                 overflow-hidden
//               "
//               style={{
//                 background: `
//                   linear-gradient(
//                     145deg,
//                     #0d0d0d,
//                     #020202
//                   )
//                 `,

//                 boxShadow: `
//                   inset 0 0 0 1px rgba(255,255,255,0.03),
//                   inset 0 0 40px rgba(0,0,0,0.95)
//                 `
//               }}
//             >

//               <div className="absolute inset-0 opacity-[0.04] bg-[radial-gradient(circle_at_top,rgba(255,255,255,0.25),transparent_60%)]" />

//               <div className="relative z-10 h-full flex flex-col items-center justify-center">
//                 <div className="text-[#85795d] text-5xl tracking-wider">
//                   SENTINEL
//                 </div>

//                 <div className="text-[#746c56] text-2xl">
//                   ORCHESTRATION UNIT
//                 </div>

//                 <div className="mt-5 text-[#5c5646] text-lg">
//                   MODEL: ST-7
//                 </div>

//                 <div className="text-[#5c5646] text-lg">
//                   SERIAL: 8847-AI
//                 </div>
//               </div>
//             </div>

//             {/* BUTTONS */}
//             <div className="flex gap-10">
//               <SmallButton label="SELECT" />
//               <SmallButton label="START" />
//             </div>
//           </div>

//           {/* RED BUTTONS */}
//           <div className="flex gap-10 mb-8">
//             <RedButton label="B" />
//             <RedButton label="A" />
//           </div>
//         </div>
//       </div>
//     </div>
//   )
// }

// /* -------------------------------- */

// function SidePanel({
//   title,
//   children,
// }: {
//   title: string
//   children: React.ReactNode
// }) {
//   return (
//     <div
//       className="rounded-2xl p-5 relative overflow-hidden"
//       style={{
//         background: `
//           linear-gradient(
//             145deg,
//             #020202,
//             #081208
//           )
//         `,

//         boxShadow: `
//           inset 0 0 0 1px rgba(0,255,120,0.08),
//           inset 0 0 40px rgba(0,0,0,0.95)
//         `
//       }}
//     >
//       <div className="text-[#22ff88] text-3xl font-mono mb-5">
//         {title}
//       </div>

//       {children}
//     </div>
//   )
// }

// function Meter({ value }: { value: number }) {
//   return (
//     <div className="grid grid-cols-20 gap-[3px]">
//       {Array.from({ length: 20 }).map((_, i) => (
//         <div
//           key={i}
//           className="h-10"
//           style={{
//             background:
//               i < value
//                 ? "#22ff88"
//                 : "#071107",

//             boxShadow:
//               i < value
//                 ? "0 0 16px rgba(0,255,120,0.7)"
//                 : "none",

//             border:
//               "1px solid rgba(0,255,120,0.08)"
//           }}
//         />
//       ))}
//     </div>
//   )
// }

// function SmallButton({ label }: { label: string }) {
//   return (
//     <div className="flex flex-col items-center gap-2">
//       <button
//         className="w-[130px] h-[34px] rounded-full"
//         style={{
//           background: `
//             linear-gradient(
//               to bottom,
//               #1b1b1b,
//               #050505
//             )
//           `,

//           boxShadow: `
//             inset 0 2px 4px rgba(255,255,255,0.06),
//             inset 0 -8px 12px rgba(0,0,0,0.95)
//           `
//         }}
//       />

//       <span className="text-[#22ff88] text-xl tracking-widest">
//         {label}
//       </span>
//     </div>
//   )
// }

// function RedButton({ label }: { label: string }) {
//   return (
//     <div className="flex flex-col items-center gap-3">

//       <button
//         className="relative w-[120px] h-[120px] rounded-full"
//         style={{
//           background: `
//             radial-gradient(
//               circle at 35% 30%,
//               #ffb3b3,
//               #d10000 30%,
//               #3a0000 70%,
//               #120000 100%
//             )
//           `,

//           boxShadow: `
//             inset 0 8px 18px rgba(255,255,255,0.18),
//             inset 0 -20px 30px rgba(0,0,0,0.85),
//             0 0 18px rgba(255,0,0,0.35),
//             0 8px 30px rgba(0,0,0,0.8)
//           `,

//           border: "2px solid rgba(255,255,255,0.08)"
//         }}
//       />

//       <span className="text-[#8b8269] text-5xl">
//         {label}
//       </span>
//     </div>
//   )
// }


import React from "react"
import TetrisScreen from "./TetrisScreen"
import SystemPanel from "./SystemPanel"
import ControlButtons from "./ControlButtons"

export default function TerminalShell({
  children,
}: {
  children?: React.ReactNode
}) {
  return (
    <div
      className="
        w-screen
        h-screen
        flex
        items-center
        justify-center
        bg-black
        overflow-hidden
      "
      style={{
        background: `
          radial-gradient(
            circle at center,
            #07110a 0%,
            #020403 45%,
            #000000 100%
          )
        `,
      }}
    >
      <div
        className="
          relative
          rounded-[44px]
          overflow-hidden
        "
        style={{
          width: "1460px",
          height: "1040px",

          background: `
            linear-gradient(
              145deg,
              #3f3a33 0%,
              #1b1a18 12%,
              #090909 26%,
              #1a1815 42%,
              #050505 58%,
              #22201c 76%,
              #3a352f 100%
            )
          `,

          border: "2px solid #3e392f",

          boxShadow: `
            inset 0 0 0 2px rgba(255,255,255,0.03),
            inset 0 0 30px rgba(255,255,255,0.04),
            inset 0 -40px 80px rgba(0,0,0,0.95),
            inset 0 40px 80px rgba(255,255,255,0.03),
            0 0 120px rgba(0,0,0,1)
          `,
        }}
      >
        {/* METAL NOISE */}
        <div
          className="absolute inset-0 opacity-[0.12]"
          style={{
            backgroundImage:
              "url(https://www.transparenttextures.com/patterns/brushed-alum-dark.png)",
            mixBlendMode: "overlay",
          }}
        />

        {/* TOP HIGHLIGHT */}
        <div
          className="
            absolute
            top-0
            left-0
            right-0
            h-[120px]
            pointer-events-none
          "
          style={{
            background: `
              linear-gradient(
                to bottom,
                rgba(255,255,255,0.08),
                transparent
              )
            `,
            filter: "blur(18px)",
          }}
        />

        {/* MAIN BODY */}
        <div className="absolute inset-[38px]">
          {/* SCREEN CHAMBER */}
          <div
            className="
              absolute
              left-[40px]
              top-[40px]
              rounded-[42px]
              overflow-hidden
            "
            style={{
              width: "920px",
              height: "700px",

              background: `
                linear-gradient(
                  145deg,
                  #050505,
                  #0f0f0f 30%,
                  #020202 100%
                )
              `,

              border: "1px solid #2f2f2f",

              boxShadow: `
                inset 0 0 40px rgba(255,255,255,0.03),
                inset 0 0 140px rgba(0,0,0,1),
                0 12px 30px rgba(0,0,0,0.85)
              `,
            }}
          >
            {/* INNER CRT */}
            <div
              className="absolute inset-[28px]"
              style={{
                borderRadius: "34px",
                overflow: "hidden",
              }}
            >
              <TetrisScreen>
                {children}
              </TetrisScreen>
            </div>
          </div>

          {/* SIDE PANELS */}
          <div className="absolute right-[38px] top-[40px]">
            <SystemPanel />
          </div>

          {/* DPAD */}
          <div
            className="
              absolute
              left-[70px]
              bottom-[70px]
            "
          >
            <ControlButtons />
          </div>

          {/* CENTER PLATE */}
          <div
            className="
              absolute
              left-[370px]
              bottom-[78px]
              rounded-[18px]
            "
            style={{
              width: "300px",
              height: "170px",

              background: `
                linear-gradient(
                  145deg,
                  #161616,
                  #050505 55%,
                  #1d1d1d
                )
              `,

              border: "1px solid #2d2d2d",

              boxShadow: `
                inset 0 0 20px rgba(255,255,255,0.03),
                inset 0 -20px 40px rgba(0,0,0,0.8),
                0 10px 20px rgba(0,0,0,0.7)
              `,
            }}
          >
            <div
              className="
                absolute
                left-[28px]
                top-[28px]
                text-[#8d846f]
                font-mono
              "
            >
              <div className="text-[54px] font-bold tracking-wider">
                SENTINEL
              </div>

              <div className="text-[20px] opacity-70 mt-2">
                ORCHESTRATION UNIT
              </div>

              <div className="text-[14px] opacity-50 mt-6 leading-7">
                MODEL: ST-7
                <br />
                SERIAL: 8847-AI
              </div>
            </div>
          </div>

          {/* RED BUTTONS */}
          <div
            className="
              absolute
              right-[90px]
              bottom-[90px]
              flex
              gap-[60px]
            "
          >
            {["B", "A"].map((label) => (
              <div
                key={label}
                className="flex flex-col items-center"
              >
                <div
                  className="rounded-full"
                  style={{
                    width: "120px",
                    height: "120px",

                    background: `
                      radial-gradient(
                        circle at 35% 30%,
                        #ffd2d2 0%,
                        #ff5252 22%,
                        #8b0000 65%,
                        #180000 100%
                      )
                    `,

                    border: "3px solid #421313",

                    boxShadow: `
                      inset -10px -18px 30px rgba(0,0,0,0.65),
                      inset 10px 10px 20px rgba(255,255,255,0.12),
                      0 0 30px rgba(255,0,0,0.28),
                      0 8px 18px rgba(0,0,0,0.9)
                    `,
                  }}
                />

                <div
                  className="
                    text-[#9d927c]
                    text-[56px]
                    mt-5
                    font-mono
                  "
                >
                  {label}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}