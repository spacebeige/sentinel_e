import React from "react";

export function CropPatch({
  position,
}: {
  position: [number, number, number];
}) {
  return (
    <group position={position}>
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[6, 6]} />

        <meshStandardMaterial color="#7b5d32" />
      </mesh>

      {Array.from({ length: 9 }).map((_, i) => {
        const x = (i % 3) * 1.6 - 1.6;
        const z = Math.floor(i / 3) * 1.6 - 1.6;

        return (
          <mesh key={i} position={[x, 0.8, z]}>
            <boxGeometry args={[0.3, 1.5, 0.3]} />

            <meshStandardMaterial color="#4caf50" />
          </mesh>
        );
      })}
    </group>
  );
}