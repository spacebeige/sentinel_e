import React from "react";

export function Rock({
  position,
}: {
  position: [number, number, number];
}) {
  return (
    <mesh position={position}>
      <dodecahedronGeometry args={[1, 0]} />

      <meshStandardMaterial color="#777777" />
    </mesh>
  );
}