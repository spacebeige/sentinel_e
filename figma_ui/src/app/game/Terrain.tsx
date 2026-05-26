// import React, { useMemo, useRef, useEffect } from "react";
// import * as THREE from "three";
// import { InstancedRigidBodies } from "@react-three/rapier";
// import { createNoise2D } from "simplex-noise";

// /*
//   TERRAIN REDESIGN GOALS:
//   - Less abstract geometry chaos
//   - More Stardew Valley readability
//   - Cozy village layout
//   - Roads + rivers + districts
//   - Recognizable regions
//   - Reduced neon overload
//   - Better emotional navigation
// */

// const GRID_SIZE = 70;
// const SPACING = 3;

// const VILLAGE_RADIUS = 35;
// const FOREST_RADIUS = 80;

// function House({ position }: { position: [number, number, number] }) {
//   return (
//     <group position={position}>
//       {/* House Base */}
//       <mesh position={[0, 2, 0]} castShadow receiveShadow>
//         <boxGeometry args={[6, 4, 6]} />
//         <meshStandardMaterial color="#8b5a2b" />
//       </mesh>

//       {/* Roof */}
//       <mesh position={[0, 5, 0]} rotation={[0, Math.PI / 4, 0]}>
//         <coneGeometry args={[5, 3, 4]} />
//         <meshStandardMaterial color="#b7410e" />
//       </mesh>

//       {/* Door */}
//       <mesh position={[0, 1, 3.1]}>
//         <boxGeometry args={[1.2, 2, 0.2]} />
//         <meshStandardMaterial color="#3b2416" />
//       </mesh>
//     </group>
//   );
// }

// function Bridge({ position }: { position: [number, number, number] }) {
//   return (
//     <group position={position}>
//       <mesh receiveShadow>
//         <boxGeometry args={[10, 0.5, 4]} />
//         <meshStandardMaterial color="#7c5a3a" />
//       </mesh>

//       {/* Rails */}
//       <mesh position={[0, 1, -1.8]}>
//         <boxGeometry args={[10, 1, 0.2]} />
//         <meshStandardMaterial color="#5c4033" />
//       </mesh>

//       <mesh position={[0, 1, 1.8]}>
//         <boxGeometry args={[10, 1, 0.2]} />
//         <meshStandardMaterial color="#5c4033" />
//       </mesh>
//     </group>
//   );
// }

// export default function Terrain() {
//   const noise2D = useMemo(() => createNoise2D(), []);

//   const instancedMeshRef = useRef<THREE.InstancedMesh>(null);

//   const { positions, scales, colors } = useMemo(() => {
//     const pos: [number, number, number][] = [];
//     const sca: [number, number, number][] = [];
//     const col = new Float32Array(GRID_SIZE * GRID_SIZE * 3);

//     const offset = (GRID_SIZE * SPACING) / 2;
//     let idx = 0;

//     for (let x = 0; x < GRID_SIZE; x++) {
//       for (let z = 0; z < GRID_SIZE; z++) {
//         const worldX = x * SPACING - offset;
//         const worldZ = z * SPACING - offset;

//         const dist = Math.sqrt(worldX * worldX + worldZ * worldZ);

//         let height = 1;
//         let scale = SPACING;
//         const color = new THREE.Color();

//         if (dist < VILLAGE_RADIUS) {
//           height = 1;
//           if (Math.abs(worldX) < 3 || Math.abs(worldZ) < 3) {
//             color.set("#8b6b3f");
//           } else {
//             color.set("#5f8f52");
//           }
//         } else if (Math.abs(worldX + 25) < 5) {
//           height = -2;
//           color.set("#4ca7c8");
//           scale = SPACING * 0.9;
//         } else if (dist < FOREST_RADIUS) {
//           const noise = noise2D(worldX * 0.03, worldZ * 0.03);
//           height = noise * 2 + 1;
//           color.set(noise > 0.2 ? "#355e3b" : "#4f7942");
//         } else {
//           const noise = noise2D(worldX * 0.06, worldZ * 0.06);
//           height = noise * 8;
//           scale = SPACING * (0.5 + Math.random() * 0.3);
//           color.set(noise > 0 ? "#8b2e3c" : "#5e1f29");
//         }

//         pos.push([worldX, height / 2 - 1, worldZ]);
//         sca.push([scale, Math.max(1, Math.abs(height)), scale]);

//         col[idx * 3] = color.r;
//         col[idx * 3 + 1] = color.g;
//         col[idx * 3 + 2] = color.b;
//         idx++;
//       }
//     }
//     return { positions: pos, scales: sca, colors: col };
//   }, [noise2D]);

//   useEffect(() => {
//     if (instancedMeshRef.current) {
//       instancedMeshRef.current.instanceColor = new THREE.InstancedBufferAttribute(colors, 3);
//     }
//   }, [colors]);

//   return (
//     <>
//       <InstancedRigidBodies
//         positions={positions as any}
//         scales={scales as any}
//         colliders="cuboid"
//         type="fixed"
//       >
//         <instancedMesh
//           ref={instancedMeshRef as any}
//           args={[undefined, undefined, positions.length]}
//           castShadow
//           receiveShadow
//         >
//           <boxGeometry />
//           <meshStandardMaterial vertexColors roughness={1} metalness={0} />
//         </instancedMesh>
//       </InstancedRigidBodies>

//       <House position={[10, 0, 10]} />
//       <House position={[-12, 0, 8]} />
//       <House position={[15, 0, -10]} />
//       <House position={[-18, 0, -15]} />

//       <Bridge position={[-25, 0, 0]} />

//       {Array.from({ length: 50 }).map((_, i) => {
//         const x = (Math.random() - 0.5) * 160;
//         const z = (Math.random() - 0.5) * 160;

//         if (Math.sqrt(x * x + z * z) < 40) return null;

//         return (
//           <group key={i} position={[x, 0, z]}>
//             <mesh position={[0, 2, 0]}>
//               <cylinderGeometry args={[0.4, 0.6, 4]} />
//               <meshStandardMaterial color="#5b3a29" />
//             </mesh>
//             <mesh position={[0, 5, 0]}>
//               <sphereGeometry args={[2.5, 8, 8]} />
//               <meshStandardMaterial color="#3d6b3d" />
//             </mesh>
//           </group>
//         );
//       })}
//     </>
//   );
// }

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