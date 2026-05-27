// import { create } from "zustand";

// interface DialogueLine {
//   text: string;
// }

// interface DialogueState {
//   activeDialogue: {
//     npcName: string;
//     lines: DialogueLine[];
//   } | null;

//   currentIndex: number;

//   startDialogue: (
//     dialogue: {
//       npcName: string;
//       lines: DialogueLine[];
//     }
//   ) => void;

//   nextLine: () => void;

//   closeDialogue: () => void;
// }

// export const useDialogueStore =
//   create<DialogueState>((set, get) => ({
//     activeDialogue: null,

//     currentIndex: 0,

//     startDialogue: (dialogue) =>
//       set({
//         activeDialogue: dialogue,
//         currentIndex: 0,
//       }),

//     nextLine: () => {
//       const state = get();

//       if (!state.activeDialogue) return;

//       if (
//         state.currentIndex <
//         state.activeDialogue.lines.length - 1
//       ) {
//         set({
//           currentIndex:
//             state.currentIndex + 1,
//         });
//       } else {
//         set({
//           activeDialogue: null,
//           currentIndex: 0,
//         });
//       }
//     },

//     closeDialogue: () =>
//       set({
//         activeDialogue: null,
//         currentIndex: 0,
//       }),
//   }));




import { create } from "zustand";

interface DialogueLine {
  text: string;
}

interface DialogueState {
  activeDialogue: {
    npcName: string;
    lines: DialogueLine[];
  } | null;

  currentIndex: number;

  startDialogue: (
    dialogue: {
      npcName: string;
      lines: DialogueLine[];
    }
  ) => void;

  nextLine: () => void;
}

export const useDialogueStore =
  create<DialogueState>((set, get) => ({
    activeDialogue: null,

    currentIndex: 0,

    startDialogue: (dialogue) =>
      set({
        activeDialogue: dialogue,
        currentIndex: 0,
      }),

    nextLine: () => {
      const state = get();

      if (!state.activeDialogue) return;

      if (
        state.currentIndex <
        state.activeDialogue.lines.length - 1
      ) {
        set({
          currentIndex:
            state.currentIndex + 1,
        });
      } else {
        set({
          activeDialogue: null,
          currentIndex: 0,
        });
      }
    },
  }));