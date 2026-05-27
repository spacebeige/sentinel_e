import React from "react";

export function House({
  position,
}: {
  position: [number, number, number];
}) {
  return (
    <group position={position}>
      {/* foundation */}

      <mesh position={[0, 0.4, 0]}>
        <boxGeometry args={[8, 0.8, 8]} />

        <meshStandardMaterial color="#75604f" />
      </mesh>

      {/* walls */}

      <mesh position={[0, 3, 0]}>
        <boxGeometry args={[7, 5, 7]} />

        <meshStandardMaterial color="#d7a45a" />
      </mesh>

      {/* roof */}

      <mesh position={[0, 6.5, 0]} rotation={[0, Math.PI / 4, 0]}>
        <coneGeometry args={[6, 4, 4]} />

        <meshStandardMaterial color="#b64d2f" />
      </mesh>

      {/* porch */}

      <mesh position={[0, 1, 4]}>
        <boxGeometry args={[5, 0.4, 2]} />

        <meshStandardMaterial color="#8b5a2b" />
      </mesh>
    </group>
  );
}