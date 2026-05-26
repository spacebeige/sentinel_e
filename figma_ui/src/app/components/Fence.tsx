import React from "react";

export function Fence({
  position,
}: {
  position: [number, number, number];
}) {
  return (
    <mesh position={position}>
      <boxGeometry args={[2, 1, 0.3]} />

      <meshStandardMaterial color="#8b5a2b" />
    </mesh>
  );
}