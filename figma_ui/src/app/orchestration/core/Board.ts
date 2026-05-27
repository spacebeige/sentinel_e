export type Offset = { x: number; y: number };

export interface Brick {
  location: Offset;
}

export const MATRIX_WIDTH = 12;
export const MATRIX_HEIGHT = 24;

export function createBrick(x: number, y: number): Brick {
  return { location: { x, y } };
}

export function offsetBrick(brick: Brick, dx: number, dy: number): Brick {
  return { location: { x: brick.location.x + dx, y: brick.location.y + dy } };
}
