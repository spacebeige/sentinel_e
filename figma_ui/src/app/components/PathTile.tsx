import React from "react";

export function PathTile({
  position,
}: {
  position: [number, number, number];
}) {
  return (
    <mesh
      rotation={[-Math.PI / 2, 0, 0]}
      position={position}
    >
      <planeGeometry args={[2, 2]} />

      <meshStandardMaterial color="#d1a45f" />
    </mesh>
  );
}