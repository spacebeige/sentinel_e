import React, { useRef } from "react";

import { useFrame } from "@react-three/fiber";

import { useKeyboardControls } from "@react-three/drei";

import * as THREE from "three";

export default function Player() {
  const playerRef = useRef<THREE.Group>(null);

  const [, getKeys] = useKeyboardControls();

  const velocity = new THREE.Vector3();

  const SPEED = 0.13;

  useFrame((state) => {
    if (!playerRef.current) return;

    const keys = getKeys();

    velocity.set(0, 0, 0);

    if (keys.forward) velocity.z -= SPEED;
    if (keys.backward) velocity.z += SPEED;
    if (keys.left) velocity.x -= SPEED;
    if (keys.right) velocity.x += SPEED;

    playerRef.current.position.x += velocity.x;
    playerRef.current.position.z += velocity.z;

    state.camera.position.lerp(
      new THREE.Vector3(
        playerRef.current.position.x,
        50,
        playerRef.current.position.z + 0.1
      ),
      0.08
    );

    state.camera.lookAt(
      playerRef.current.position.x,
      0,
      playerRef.current.position.z
    );
  });

  return (
    <group ref={playerRef} position={[0, 0.1, 0]}>
      {/* shadow */}

      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[0.7, 16]} />

        <meshBasicMaterial
          color="black"
          transparent
          opacity={0.2}
        />
      </mesh>

      {/* body */}

      <mesh position={[0, 1, 0]}>
        <planeGeometry args={[1.2, 2]} />

        <meshStandardMaterial color="#2f6db2" />
      </mesh>

      {/* hat */}

      <mesh position={[0, 2.1, 0.01]}>
        <planeGeometry args={[1.4, 0.4]} />

        <meshStandardMaterial color="#8b5a2b" />
      </mesh>
    </group>
  );
}