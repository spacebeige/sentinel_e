import React from "react";

import { Canvas } from "@react-three/fiber";

import {
  KeyboardControls,
  Sky,
} from "@react-three/drei";

import {
  EffectComposer,
  Bloom,
} from "@react-three/postprocessing";

import Player from "./Player";
import Terrain from "./Terrain";

import { NPC } from "./entities/NPC";

import { RetroDialogueBox } from "./ui/RetroDialogueBox";

export default function GameScene() {
  const keyboardMap = [
    {
      name: "forward",
      keys: ["ArrowUp", "w", "W"],
    },
    {
      name: "backward",
      keys: ["ArrowDown", "s", "S"],
    },
    {
      name: "left",
      keys: ["ArrowLeft", "a", "A"],
    },
    {
      name: "right",
      keys: ["ArrowRight", "d", "D"],
    },
  ];

  return (
    <KeyboardControls map={keyboardMap}>
      <Canvas
        orthographic
        shadows="basic"
        gl={{ antialias: true }}
        camera={{
          zoom: 42,
          position: [0, 50, 0],
          near: 0.1,
          far: 1000,
        }}
        style={{
          width: "100vw",
          height: "100vh",
          background:
            "linear-gradient(to bottom, #86b35f, #3e5d35)",
        }}
      >
        <Sky
          distance={450000}
          sunPosition={[5, 1, 8]}
          inclination={0.52}
          azimuth={0.25}
        />

        <ambientLight intensity={0.75} />

        <directionalLight
          castShadow
          position={[30, 50, 20]}
          intensity={1}
          shadow-mapSize={[1024, 1024]}
        />

        <EffectComposer>
          <Bloom
            intensity={0.08}
            luminanceThreshold={0.92}
          />
        </EffectComposer>

        <Terrain />

        <Player />

        <NPC
          name="Archivist"
          position={[6, 0, -2]}
          color="#4a74c9"
          dialoguePools={[
            [
              "The river remembers every disagreement.",
              "Most people forget too quickly."
            ],
            [
              "The old council wanted certainty.",
              "That nearly destroyed the village."
            ]
          ]}
        />

        <NPC
          name="Drift Monk"
          position={[-10, 0, 8]}
          color="#d4a45d"
          dialoguePools={[
            [
              "The forest changes shape every season.",
              "Thoughts drift the same way."
            ],
            [
              "Walk slowly.",
              "The land speaks softly here."
            ]
          ]}
        />

        <NPC
          name="Council Watcher"
          position={[15, 0, 10]}
          color="#5c8f4e"
          dialoguePools={[
            [
              "One model answers quickly.",
              "Many models answer honestly."
            ],
            [
              "The council no longer seeks agreement.",
              "Only perspective."
            ]
          ]}
        />
      </Canvas>

      <RetroDialogueBox />
    </KeyboardControls>
  );
}