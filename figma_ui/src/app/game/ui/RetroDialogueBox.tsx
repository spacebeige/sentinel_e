// import React from "react";

// import { useDialogueStore } from "../store/dialogueStore";

// export function RetroDialogueBox() {
//   const {
//     activeDialogue,
//     currentIndex,
//     nextLine,
//   } = useDialogueStore();

//   if (!activeDialogue) return null;

//   const line =
//     activeDialogue.lines[currentIndex];

//   return (
//     <div
//       onClick={nextLine}
//       style={{
//         position: "fixed",
//         bottom: 24,
//         left: "50%",
//         transform: "translateX(-50%)",
//         width: "70%",
//         background: "#2f1f14",
//         border: "4px solid #c8a165",
//         borderRadius: 12,
//         padding: 20,
//         color: "#f5deb3",
//         fontFamily: "monospace",
//         zIndex: 999,
//         cursor: "pointer",
//         boxShadow:
//           "0 8px 30px rgba(0,0,0,0.45)",
//       }}
//     >
//       <div
//         style={{
//           fontWeight: "bold",
//           marginBottom: 12,
//           color: "#ffd27f",
//         }}
//       >
//         {activeDialogue.npcName}
//       </div>

//       <div
//         style={{
//           lineHeight: 1.7,
//           fontSize: 18,
//         }}
//       >
//         {line.text}
//       </div>

//       <div
//         style={{
//           marginTop: 12,
//           opacity: 0.7,
//           fontSize: 12,
//         }}
//       >
//         click to continue
//       </div>
//     </div>
//   );
// }

import React from "react";

import { useDialogueStore } from "../store/dialogueStore";

export function RetroDialogueBox() {
  const {
    activeDialogue,
    currentIndex,
    nextLine,
  } = useDialogueStore();

  if (!activeDialogue) return null;

  return (
    <div
      onClick={nextLine}
      style={{
        position: "fixed",
        left: "50%",
        bottom: 20,
        transform: "translateX(-50%)",
        width: "70%",
        background: "#2c1d14",
        border: "4px solid #d1a45f",
        borderRadius: 12,
        padding: 20,
        color: "#f7e2b4",
        fontFamily: "monospace",
        zIndex: 999,
        cursor: "pointer",
      }}
    >
      <div
        style={{
          marginBottom: 12,
          color: "#ffd27f",
          fontWeight: "bold",
        }}
      >
        {activeDialogue.npcName}
      </div>

      <div>
        {
          activeDialogue.lines[currentIndex]
            .text
        }
      </div>

      <div
        style={{
          marginTop: 12,
          opacity: 0.7,
          fontSize: 12,
        }}
      >
        click to continue
      </div>
    </div>
  );
}