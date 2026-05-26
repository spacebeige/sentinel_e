import { Brick, Offset, MATRIX_WIDTH, MATRIX_HEIGHT } from "./Board";

export type CognitiveType = 
  | "MEMORY_CHAIN" 
  | "ARBITRATION_NODE" 
  | "EMBEDDING_CLUSTER" 
  | "DIVERGENCE_FRAGMENT" 
  | "ROUTING_PIPELINE";

export interface Spirit {
  shape: Offset[];
  offset: Offset;
  typeId?: string;
  cognitiveType?: CognitiveType;
}

export const EmptySpirit: Spirit = { shape: [], offset: { x: 0, y: 0 } };

export const SPIRIT_TYPES: { id: string; shape: Offset[]; cognitiveType: CognitiveType }[] = [
  { id: "Z", cognitiveType: "DIVERGENCE_FRAGMENT", shape: [{ x: 1, y: -1 }, { x: 1, y: 0 }, { x: 0, y: 0 }, { x: 0, y: 1 }] },
  { id: "S", cognitiveType: "DIVERGENCE_FRAGMENT", shape: [{ x: 0, y: -1 }, { x: 0, y: 0 }, { x: 1, y: 0 }, { x: 1, y: 1 }] },
  { id: "I", cognitiveType: "MEMORY_CHAIN", shape: [{ x: 0, y: -1 }, { x: 0, y: 0 }, { x: 0, y: 1 }, { x: 0, y: 2 }] },
  { id: "T", cognitiveType: "ARBITRATION_NODE", shape: [{ x: 0, y: 1 }, { x: 0, y: 0 }, { x: 0, y: -1 }, { x: 1, y: 0 }] },
  { id: "O", cognitiveType: "EMBEDDING_CLUSTER", shape: [{ x: 1, y: 0 }, { x: 0, y: 0 }, { x: 1, y: -1 }, { x: 0, y: -1 }] },
  { id: "L", cognitiveType: "ROUTING_PIPELINE", shape: [{ x: 0, y: -1 }, { x: 1, y: -1 }, { x: 1, y: 0 }, { x: 1, y: 1 }] },
  { id: "J", cognitiveType: "ROUTING_PIPELINE", shape: [{ x: 1, y: -1 }, { x: 0, y: -1 }, { x: 0, y: 0 }, { x: 0, y: 1 }] },
];

export function getSpiritLocation(spirit: Spirit): Offset[] {
  return spirit.shape.map((p) => ({ x: p.x + spirit.offset.x, y: p.y + spirit.offset.y }));
}

export function moveSpirit(spirit: Spirit, dx: number, dy: number): Spirit {
  return { ...spirit, offset: { x: spirit.offset.x + dx, y: spirit.offset.y + dy } };
}

export function rotateSpirit(spirit: Spirit): Spirit {
  const newShape = spirit.shape.map((p) => ({ x: p.y, y: -p.x }));
  return { ...spirit, shape: newShape };
}

export function adjustSpiritOffset(spirit: Spirit): Spirit {
  const loc = getSpiritLocation(spirit);
  let yOffset = 0;
  let xOffset = 0;

  const minY = Math.min(...loc.map((p) => p.y));
  if (minY < 0) yOffset += Math.abs(minY);
  
  const maxY = Math.max(...loc.map((p) => p.y));
  if (maxY > MATRIX_HEIGHT - 1) yOffset += (MATRIX_HEIGHT - 1 - maxY);

  const minX = Math.min(...loc.map((p) => p.x));
  if (minX < 0) xOffset += Math.abs(minX);

  const maxX = Math.max(...loc.map((p) => p.x));
  if (maxX > MATRIX_WIDTH - 1) xOffset += (MATRIX_WIDTH - 1 - maxX);

  return moveSpirit(spirit, xOffset, yOffset);
}

export function isValidInMatrix(spirit: Spirit, bricks: Brick[]): boolean {
  const loc = getSpiritLocation(spirit);
  return loc.every((p) => {
    if (p.x < 0 || p.x > MATRIX_WIDTH - 1 || p.y > MATRIX_HEIGHT - 1) return false;
    return !bricks.some((b) => b.location.x === p.x && b.location.y === p.y);
  });
}

export function generateSpiritReserve(): Spirit[] {
  // Shuffle the bag
  const shuffled = [...SPIRIT_TYPES].sort(() => Math.random() - 0.5);
  return shuffled.map((type) => {
    const s: Spirit = { 
      shape: type.shape, 
      offset: { x: Math.floor(Math.random() * (MATRIX_WIDTH - 1)), y: -1 }, 
      typeId: type.id,
      cognitiveType: type.cognitiveType 
    };
    // We only adjust X bounds initially so it doesn't spawn out of left/right walls. We don't adjust Y so it spawns at top.
    const loc = getSpiritLocation(s);
    let xOffset = 0;
    const minX = Math.min(...loc.map((p) => p.x));
    if (minX < 0) xOffset += Math.abs(minX);
    const maxX = Math.max(...loc.map((p) => p.x));
    if (maxX > MATRIX_WIDTH - 1) xOffset += (MATRIX_WIDTH - 1 - maxX);
    return moveSpirit(s, xOffset, 0);
  });
}
