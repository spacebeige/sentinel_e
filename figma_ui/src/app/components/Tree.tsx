import React from "react";

export function Tree({
  position,
}: {
  position: [number, number, number];
}) {
  const size = 2 + Math.random() * 2;

  return (
    <group position={position}>
      <mesh position={[0, 2, 0]}>
        <cylinderGeometry args={[0.5, 0.7, 4]} />

        <meshStandardMaterial color="#6b4423" />
      </mesh>

      <mesh position={[0, 5, 0]}>
        <coneGeometry args={[size, 6, 8]} />

        <meshStandardMaterial color="#2e6d39" />
      </mesh>
    </group>
  );
}