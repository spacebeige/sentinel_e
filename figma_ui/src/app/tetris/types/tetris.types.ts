export type CellValue =
| 0
| 1
| 2
| 3
| 4
| 5

export type Matrix = CellValue[][]

export type TetrominoType =
| "I"
| "O"
| "T"
| "S"
| "Z"
| "J"
| "L"

export interface Position {
x: number
y: number
}

export interface Tetromino {
type: TetrominoType
shape: number[][]
color: string
position: Position
}

export interface GameState {
matrix: Matrix
active: Tetromino | null
score: number
lines: number
level: number
gameOver: boolean
}
