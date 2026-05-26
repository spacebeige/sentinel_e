import React from "react";

export function Lamp({
  position,
}: {
  position: [number, number, number];
}) {
  return (
    <group position={position}>
      <mesh position={[0, 2, 0]}>
        <cylinderGeometry args={[0.2, 0.2, 4]} />

        <meshStandardMaterial color="#5b4128" />
      </mesh>

      <mesh position={[0, 4.3, 0]}>
        <sphereGeometry args={[0.4, 8, 8]} />

        <meshStandardMaterial
          color="#ffd27f"
          emissive="#ffd27f"
          emissiveIntensity={0.5}
        />
      </mesh>

      <pointLight
        intensity={0.7}
        distance={12}
        color="#ffddaa"
      />
    </group>
  );
}