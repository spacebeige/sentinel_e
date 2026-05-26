import React from "react";

export function WaterTile({
  position,
  scale,
}: {
  position: [number, number, number];
  scale: [number, number, number];
}) {
  return (
    <mesh
      rotation={[-Math.PI / 2, 0, 0]}
      position={position}
      scale={scale}
    >
      <planeGeometry args={[1, 1]} />

      <meshStandardMaterial
        color="#5ca7d4"
        transparent
        opacity={0.9}
      />
    </mesh>
  );
}