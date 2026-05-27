import { createMatrix } from "./Matrix"

export function createInitialState() {
return {
matrix: createMatrix(),
score: 0,
lines: 0,
level: 1,
gameOver: false,
}
}
